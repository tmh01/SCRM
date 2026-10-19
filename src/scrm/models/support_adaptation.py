# 保留原交叉拟合与适配计算，仅将检查点文件查找改成显式参数。
from __future__ import annotations
import json
import numpy as np
from scrm.config import ExperimentConfig
from scrm.models.self_anchor import support_prior_self_anchor
from scrm.training.trainer import _TorchPrefixDataset, _make_collate, _model_sequence_limit, _prediction_frame, _resolve_device
from scrm.training.transfer import _fit_epochs, instantiate_transfer_model, select_finetune_epoch

def _predict_and_encode(model, samples, vocab, scaler, batch_size: int, device):
    import torch
    from torch.utils.data import DataLoader
    if not samples:
        raise ValueError('SCRM representation extraction received no prefixes')
    if not hasattr(model, 'encode'):
        raise TypeError('SCRM requires a Transformer backbone exposing encode()')
    dataset = _TorchPrefixDataset(samples, vocab, scaler)
    loader = DataLoader(dataset, batch_size=min(int(batch_size), max(1, len(dataset))), shuffle=False, collate_fn=_make_collate(_model_sequence_limit(model)))
    predictions, representations = ([], [])
    model.eval()
    with torch.no_grad():
        for activity_ids, time_features, lengths, _labels, _indices in loader:
            activity_ids = activity_ids.to(device)
            time_features = time_features.to(device)
            lengths = lengths.to(device)
            representation = model.encode(activity_ids, time_features, lengths)
            output = model.head(representation).squeeze(-1)
            predictions.append(torch.expm1(output).clamp_min(0).cpu().numpy())
            representations.append(representation.cpu().numpy())
    return (np.concatenate(predictions).astype(float), np.concatenate(representations).astype(np.float32))

def residual_memory_predictions(support_embeddings: np.ndarray, support_case_ids: np.ndarray, support_remaining: np.ndarray, support_base_predictions: np.ndarray, query_embeddings: np.ndarray, query_base_predictions: np.ndarray, *, max_cases: int=20, batch_size: int=512, exclude_query_case_ids: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    support_embeddings = np.asarray(support_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    support_case_ids = np.asarray(support_case_ids)
    support_remaining = np.asarray(support_remaining, dtype=float)
    support_base_predictions = np.asarray(support_base_predictions, dtype=float)
    query_base_predictions = np.asarray(query_base_predictions, dtype=float)
    if support_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError('SCRM memory embeddings must be two-dimensional')
    if support_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError('Support/query embedding dimensions do not match')
    if len(support_embeddings) != len(support_case_ids) or len(support_embeddings) != len(support_remaining):
        raise ValueError('Support memory arrays are not aligned')
    if len(query_embeddings) != len(query_base_predictions):
        raise ValueError('Query representations and base predictions are not aligned')
    import torch
    support_norm = support_embeddings / np.maximum(np.sqrt(np.sum(np.square(support_embeddings), axis=1, keepdims=True)), 1e-08)
    query_norm = query_embeddings / np.maximum(np.sqrt(np.sum(np.square(query_embeddings), axis=1, keepdims=True)), 1e-08)
    support_norm_t = torch.from_numpy(np.ascontiguousarray(support_norm.T))
    cases = np.asarray(sorted(set(support_case_ids.tolist())), dtype=support_case_ids.dtype)
    groups = [np.flatnonzero(support_case_ids == case_id) for case_id in cases]
    residuals = support_remaining - support_base_predictions
    n_queries = len(query_embeddings)
    output = np.empty(n_queries, dtype=float)
    max_similarity = np.empty(n_queries, dtype=float)
    entropy = np.empty(n_queries, dtype=float)
    effective_neighbors = np.empty(n_queries, dtype=float)
    residual_dispersion = np.empty(n_queries, dtype=float)
    for start in range(0, n_queries, int(batch_size)):
        stop = min(start + int(batch_size), n_queries)
        similarity = torch.matmul(torch.from_numpy(np.ascontiguousarray(query_norm[start:stop])), support_norm_t).numpy()
        case_scores = np.empty((stop - start, len(cases)), dtype=float)
        case_residuals = np.empty_like(case_scores)
        for case_index, indices in enumerate(groups):
            local = similarity[:, indices]
            nearest = np.argmax(local, axis=1)
            selected = indices[nearest]
            case_scores[:, case_index] = local[np.arange(len(local)), nearest]
            case_residuals[:, case_index] = residuals[selected]
        if exclude_query_case_ids is not None:
            excluded = np.asarray(exclude_query_case_ids[start:stop])
            case_scores[cases[None, :] == excluded[:, None]] = -np.inf
        for row_index in range(stop - start):
            available = np.flatnonzero(np.isfinite(case_scores[row_index]))
            if not len(available):
                raise ValueError('Residual memory has no legal support case after exclusion')
            count = min(max(1, int(max_cases)), len(available))
            if count < len(available):
                local_order = np.argpartition(case_scores[row_index, available], -count)[-count:]
                chosen = available[local_order]
            else:
                chosen = available
            scores = case_scores[row_index, chosen]
            values = case_residuals[row_index, chosen]
            distances = np.maximum(2.0 - 2.0 * scores, 0.0)
            positive = distances[distances > 1e-08]
            bandwidth = float(np.median(positive)) if len(positive) else float(np.mean(distances) + 0.001)
            bandwidth = max(bandwidth, 0.001)
            logits = -distances / bandwidth
            logits -= float(np.max(logits))
            weights = np.exp(logits)
            weights /= max(float(weights.sum()), 1e-12)
            correction = float(np.sum(weights * values))
            prediction_index = start + row_index
            output[prediction_index] = max(query_base_predictions[prediction_index] + correction, 0.0)
            max_similarity[prediction_index] = float(np.max(scores))
            effective_neighbors[prediction_index] = float(1.0 / max(float(np.square(weights).sum()), 1e-12))
            entropy[prediction_index] = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum() / np.log(len(weights))) if len(weights) > 1 else 0.0
            center = float(np.sum(weights * values))
            residual_dispersion[prediction_index] = float(np.sqrt(np.sum(weights * np.square(values - center))))
    return (output, {'max_similarity': max_similarity, 'entropy': entropy, 'effective_neighbors': effective_neighbors, 'residual_dispersion': residual_dispersion, 'available_cases': np.full(n_queries, len(cases), dtype=float)})

def robust_prior_predictions(support_samples, query_samples, *, min_activity_cases: int=2, exclude_query_case_ids: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    duration_by_case: dict[int, float] = {}
    activity_by_case: dict[str, dict[int, list[float]]] = {}
    for sample in support_samples:
        case_id = int(sample.case_id)
        duration_by_case[case_id] = float(sample.case_duration_hours)
        activity = str(sample.activity_names[-1]) if sample.activity_names else '<MISSING_ACTIVITY>'
        activity_by_case.setdefault(activity, {}).setdefault(case_id, []).append(float(sample.remaining_hours))
    if not duration_by_case:
        raise ValueError('SCRM prior received no support cases')
    predictions = np.empty(len(query_samples), dtype=float)
    coverage = np.empty(len(query_samples), dtype=float)
    for index, sample in enumerate(query_samples):
        excluded = int(exclude_query_case_ids[index]) if exclude_query_case_ids is not None else None
        legal_cases = [case_id for case_id in duration_by_case if case_id != excluded]
        if not legal_cases:
            raise ValueError('SCRM prior has no legal support case after exclusion')
        total = max(float(np.median([duration_by_case[case_id] for case_id in legal_cases])) - float(sample.elapsed_hours), 0.0)
        activity = str(sample.activity_names[-1]) if sample.activity_names else '<MISSING_ACTIVITY>'
        activity_cases = activity_by_case.get(activity, {})
        values = [float(np.median(activity_cases[case_id])) for case_id in legal_cases if case_id in activity_cases]
        if len(values) >= int(min_activity_cases):
            predictions[index] = max(float(np.median(values)), 0.0)
            coverage[index] = len(values) / len(legal_cases)
        else:
            predictions[index] = total
            coverage[index] = 0.0
    return (predictions, coverage)

def _case_balanced_log_mae(y_true, y_pred, case_ids) -> float:
    import pandas as pd
    frame = pd.DataFrame({'case': np.asarray(case_ids), 'error': np.abs(np.log1p(np.asarray(y_true, float)) - np.log1p(np.asarray(y_pred, float)))})
    return float(frame.groupby('case', sort=False)['error'].mean().mean())

def support_crossfit_bias_profile(y_true: np.ndarray, base_prediction: np.ndarray, prior_prediction: np.ndarray, case_ids: np.ndarray, *, bias_grid: np.ndarray | None=None) -> dict:
    """仅用支持集折外预测选择 Self-Anchor 的稳健对数偏置。

    同时记录逐前缀、逐案例和逐案例对数误差的最优值，最终偏置取三者
    的中位数，避免极少量长轨迹或超长时长案例单独主导选择。
    """
    target = np.asarray(y_true, dtype=float)
    base = np.asarray(base_prediction, dtype=float)
    prior = np.asarray(prior_prediction, dtype=float)
    cases = np.asarray(case_ids)
    if not target.shape == base.shape == prior.shape == cases.shape:
        raise ValueError('支持集偏置选择的输入形状必须一致')
    if bias_grid is None:
        bias_grid = np.round(np.arange(-0.8, 0.201, 0.02), 10)
    unique_cases = np.asarray(sorted(set(cases.tolist())))
    scores: list[dict[str, float]] = []
    for bias in np.asarray(bias_grid, dtype=float):
        prediction = support_prior_self_anchor(base, prior, global_log_bias=float(bias))
        absolute = np.abs(target - prediction)
        log_absolute = np.abs(np.log1p(np.maximum(target, 0.0)) - np.log1p(np.maximum(prediction, 0.0)))
        case_mae = [float(np.mean(absolute[cases == case_id])) for case_id in unique_cases]
        case_log_mae = [float(np.mean(log_absolute[cases == case_id])) for case_id in unique_cases]
        scores.append({'bias': float(bias), 'prefix_mae': float(np.mean(absolute)), 'case_macro_mae': float(np.mean(case_mae)), 'case_macro_log_mae': float(np.mean(case_log_mae))})

    def best(metric: str) -> dict[str, float]:
        return min(scores, key=lambda row: (row[metric], abs(row['bias'] + 0.08)))
    prefix_best = best('prefix_mae')
    case_best = best('case_macro_mae')
    log_best = best('case_macro_log_mae')
    consensus_bias = float(np.median([prefix_best['bias'], case_best['bias'], log_best['bias']]))
    consensus = min(scores, key=lambda row: abs(row['bias'] - consensus_bias))
    fixed = min(scores, key=lambda row: abs(row['bias'] + 0.08))
    return {'selection_scope': 'target_k_shot_support_crossfit_only', 'grid_min': float(np.min(bias_grid)), 'grid_max': float(np.max(bias_grid)), 'grid_step': 0.02, 'prefix_mae_best': prefix_best, 'case_macro_mae_best': case_best, 'case_macro_log_mae_best': log_best, 'consensus_bias': consensus_bias, 'consensus_scores': consensus, 'fixed_minus_0_08_scores': fixed, 'test_labels_used': False}

def _query_confidence(memory_prediction: np.ndarray, memory_meta: dict[str, np.ndarray], activity_coverage: np.ndarray) -> np.ndarray:
    proximity = np.clip((np.asarray(memory_meta['max_similarity'], float) + 1.0) / 2.0, 0.05, 1.0)
    available = np.maximum(np.asarray(memory_meta['available_cases'], float), 1.0)
    effective = np.asarray(memory_meta['effective_neighbors'], float)
    neighbor_strength = np.sqrt(np.clip(effective / np.minimum(available, 5.0), 0.04, 1.0))
    scale = np.maximum(np.abs(np.asarray(memory_prediction, float)), 1.0)
    dispersion_strength = 1.0 / (1.0 + np.asarray(memory_meta['residual_dispersion'], float) / scale)
    memory_confidence = np.clip(proximity * neighbor_strength * dispersion_strength, 0.05, 1.0)
    prior_confidence = np.clip(0.35 + 0.65 * np.asarray(activity_coverage, float), 0.35, 1.0)
    return np.column_stack([np.ones(len(memory_prediction), dtype=float), memory_confidence, prior_confidence])

def optimize_support_gate(y_true: np.ndarray, base_prediction: np.ndarray, memory_prediction: np.ndarray, prior_prediction: np.ndarray, case_ids: np.ndarray, memory_meta: dict[str, np.ndarray], activity_coverage: np.ndarray, *, grid_step: float=0.05, regularization: float=0.01) -> tuple[np.ndarray, dict]:
    step = float(grid_step)
    if not 0 < step <= 0.5:
        raise ValueError('SCRM gate grid_step must be in (0, 0.5]')
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError('SCRM gate grid_step must divide 1.0 exactly')
    experts = np.column_stack([base_prediction, memory_prediction, prior_prediction]).astype(float)
    confidence = _query_confidence(memory_prediction, memory_meta, activity_coverage)
    cases = np.asarray(case_ids)
    unique_cases = np.asarray(sorted(set(cases.tolist())))
    target = np.asarray(y_true, dtype=float)
    prior_center = np.asarray([0.5, 0.25, 0.25], dtype=float)
    best_objective = float('inf')
    best_weights = prior_center.copy()
    best_cv_mae = float('inf')
    for base_units in range(units + 1):
        for memory_units in range(units - base_units + 1):
            prior_units = units - base_units - memory_units
            global_weights = np.asarray([base_units / units, memory_units / units, prior_units / units], dtype=float)
            query_weights = global_weights[None, :] * confidence
            denominator = query_weights.sum(axis=1, keepdims=True)
            query_weights = np.divide(query_weights, np.maximum(denominator, 1e-12), out=np.tile(prior_center, (len(query_weights), 1)))
            prediction = np.maximum(np.sum(query_weights * experts, axis=1), 0.0)
            errors = np.abs(np.log1p(target) - np.log1p(prediction))
            cv_mae = float(np.mean([errors[cases == case_id].mean() for case_id in unique_cases]))
            objective = cv_mae + float(regularization) * float(np.square(global_weights - prior_center).sum())
            tie_break = (objective, -global_weights[0], -global_weights[1])
            current = (best_objective, -best_weights[0], -best_weights[1])
            if tie_break < current:
                best_objective = objective
                best_weights = global_weights
                best_cv_mae = cv_mae
    return (best_weights, {'crossfit_case_macro_log_mae': float(best_cv_mae), 'regularized_objective': float(best_objective), 'grid_step': step, 'n_crossfit_cases': int(len(unique_cases))})

def reliability_gate(base_prediction: np.ndarray, memory_prediction: np.ndarray, prior_prediction: np.ndarray, memory_meta: dict[str, np.ndarray], activity_coverage: np.ndarray, calibration_risks: np.ndarray, *, risk_floor: float=0.05, gate_power: float=2.0, global_weights: np.ndarray | None=None, correction_ratio_cap: float | None=None) -> tuple[np.ndarray, np.ndarray]:
    risks = np.asarray(calibration_risks, dtype=float)
    if risks.shape != (3,) or not np.isfinite(risks).all():
        raise ValueError('SCRM gate requires three finite support-CV risks')
    if global_weights is None:
        reliability = np.power(1.0 / np.maximum(risks + float(risk_floor), 1e-06), float(gate_power))
    else:
        reliability = np.asarray(global_weights, dtype=float)
        if reliability.shape != (3,) or np.any(reliability < 0) or (not np.isclose(reliability.sum(), 1.0)):
            raise ValueError('SCRM global gate weights must be a non-negative length-3 simplex')
    confidence = _query_confidence(memory_prediction, memory_meta, activity_coverage)
    raw_weights = confidence * reliability[None, :]
    weights = raw_weights / np.maximum(raw_weights.sum(axis=1, keepdims=True), 1e-12)
    experts = np.column_stack([base_prediction, memory_prediction, prior_prediction])
    prediction = np.maximum(np.sum(weights * experts, axis=1), 0.0)
    if correction_ratio_cap is not None:
        ratio_cap = float(correction_ratio_cap)
        if ratio_cap <= 1.0:
            raise ValueError('SCRM correction_ratio_cap must be greater than 1')
        base_shifted = np.asarray(base_prediction, dtype=float) + 1.0
        lower = np.maximum(base_shifted / ratio_cap - 1.0, 0.0)
        upper = base_shifted * ratio_cap - 1.0
        prediction = np.clip(prediction, lower, upper)
    return (prediction, weights)

def _fit_adapted_model(checkpoint, samples, training_cfg, epochs: int, seed: int, device):
    model, vocab, scaler = instantiate_transfer_model(checkpoint, 'transformer', samples, training_cfg, init_seed=int(seed))
    model = _fit_epochs(model, samples, vocab, scaler, int(epochs), training_cfg, int(seed), device, training_cfg.finetune_learning_rate)
    return (model, vocab, scaler)

def run_support_adaptation(cfg: ExperimentConfig, domain: str, bundle, *, checkpoint, source_metadata=None, device: str='auto', quick: bool=False, max_test_prefixes: int | None=None) -> tuple[dict[str, object], dict]:
    source_meta = dict(source_metadata or {})
    training_cfg = cfg.training_config(quick=quick)
    resolved_device = _resolve_device(device)
    best_epoch, inner_val_mae = select_finetune_epoch(checkpoint, 'transformer', bundle.inner_train_samples, bundle.inner_val_samples, training_cfg, int(bundle.seed), device_str=str(resolved_device))
    support_case_ids = np.asarray(sorted(set((sample.case_id for sample in bundle.support_samples))))
    calibration_rng = np.random.default_rng(int(bundle.seed) + 510011)
    calibration_rng.shuffle(support_case_ids)
    n_gate_folds = min(int(cfg.gate_folds), len(support_case_ids))
    if n_gate_folds < 2:
        raise ValueError('SCRM support cross-fitting requires at least two support cases')
    cross_true_parts: list[np.ndarray] = []
    cross_case_parts: list[np.ndarray] = []
    cross_base_parts: list[np.ndarray] = []
    cross_memory_parts: list[np.ndarray] = []
    cross_prior_parts: list[np.ndarray] = []
    cross_coverage_parts: list[np.ndarray] = []
    cross_memory_meta_parts: dict[str, list[np.ndarray]] = {}
    for fold_index, validation_ids in enumerate(np.array_split(support_case_ids, n_gate_folds)):
        validation_set = set(validation_ids.tolist())
        fold_train_samples = [sample for sample in bundle.support_samples if sample.case_id not in validation_set]
        fold_validation_samples = [sample for sample in bundle.support_samples if sample.case_id in validation_set]
        fold_model, fold_vocab, fold_scaler = _fit_adapted_model(checkpoint, fold_train_samples, training_cfg, best_epoch, int(bundle.seed) + 510011 + fold_index * 1009, resolved_device)
        fold_train_base, fold_train_embedding = _predict_and_encode(fold_model, fold_train_samples, fold_vocab, fold_scaler, training_cfg.batch_size, resolved_device)
        fold_validation_base, fold_validation_embedding = _predict_and_encode(fold_model, fold_validation_samples, fold_vocab, fold_scaler, training_cfg.batch_size, resolved_device)
        fold_memory, fold_memory_meta = residual_memory_predictions(fold_train_embedding, np.asarray([sample.case_id for sample in fold_train_samples]), np.asarray([sample.remaining_hours for sample in fold_train_samples]), fold_train_base, fold_validation_embedding, fold_validation_base, max_cases=cfg.memory_max_cases, batch_size=cfg.memory_batch_size)
        fold_prior, fold_coverage = robust_prior_predictions(fold_train_samples, fold_validation_samples, min_activity_cases=cfg.min_activity_prior_cases)
        cross_true_parts.append(np.asarray([sample.remaining_hours for sample in fold_validation_samples], dtype=float))
        cross_case_parts.append(np.asarray([sample.case_id for sample in fold_validation_samples]))
        cross_base_parts.append(fold_validation_base)
        cross_memory_parts.append(fold_memory)
        cross_prior_parts.append(fold_prior)
        cross_coverage_parts.append(fold_coverage)
        for key, values in fold_memory_meta.items():
            cross_memory_meta_parts.setdefault(key, []).append(values)
    cal_true = np.concatenate(cross_true_parts)
    cal_cases = np.concatenate(cross_case_parts)
    cal_val_base = np.concatenate(cross_base_parts)
    cal_memory = np.concatenate(cross_memory_parts)
    cal_prior = np.concatenate(cross_prior_parts)
    cal_coverage = np.concatenate(cross_coverage_parts)
    cal_memory_meta = {key: np.concatenate(parts) for key, parts in cross_memory_meta_parts.items()}
    risks = np.asarray([_case_balanced_log_mae(cal_true, cal_val_base, cal_cases), _case_balanced_log_mae(cal_true, cal_memory, cal_cases), _case_balanced_log_mae(cal_true, cal_prior, cal_cases)], dtype=float)
    gate_global_weights, gate_crossfit_meta = optimize_support_gate(cal_true, cal_val_base, cal_memory, cal_prior, cal_cases, cal_memory_meta, cal_coverage, grid_step=cfg.gate_grid_step, regularization=cfg.gate_regularization)
    self_anchor_bias_profile = support_crossfit_bias_profile(cal_true, cal_val_base, cal_prior, cal_cases)
    final_seed = int(bundle.seed) + 400009
    final_model, final_vocab, final_scaler = _fit_adapted_model(checkpoint, bundle.support_samples, training_cfg, best_epoch, final_seed, resolved_device)
    support_base, support_embedding = _predict_and_encode(final_model, bundle.support_samples, final_vocab, final_scaler, training_cfg.batch_size, resolved_device)
    test_samples = list(bundle.test_samples)
    if max_test_prefixes is not None:
        test_samples = test_samples[:int(max_test_prefixes)]
    test_base, test_embedding = _predict_and_encode(final_model, test_samples, final_vocab, final_scaler, training_cfg.batch_size, resolved_device)
    test_memory, test_memory_meta = residual_memory_predictions(support_embedding, np.asarray([sample.case_id for sample in bundle.support_samples]), np.asarray([sample.remaining_hours for sample in bundle.support_samples]), support_base, test_embedding, test_base, max_cases=cfg.memory_max_cases, batch_size=cfg.memory_batch_size)
    test_prior, test_coverage = robust_prior_predictions(bundle.support_samples, test_samples, min_activity_cases=cfg.min_activity_prior_cases)
    test_full, weights = reliability_gate(test_base, test_memory, test_prior, test_memory_meta, test_coverage, risks, risk_floor=cfg.gate_risk_floor, gate_power=cfg.gate_power, global_weights=gate_global_weights, correction_ratio_cap=cfg.gate_correction_ratio_cap)
    predictions = {'base': _prediction_frame(test_samples, test_base), 'memory': _prediction_frame(test_samples, test_memory), 'prior': _prediction_frame(test_samples, test_prior), 'diagnostic_mixture': _prediction_frame(test_samples, test_full)}
    meta = {**source_meta, 'architecture': 'shared_transformer+case_balanced_residual_memory+activity_duration_prior+case_crossfit_simplex_gate+log_trust_region', 'finetune_best_epoch': int(best_epoch), 'target_inner_val_mae_h': float(inner_val_mae), 'inner_split_protocol': 'private_seed_plus_target_id' if domain == 'private' else 'phase4b_seed_plus_target_char_sum', 'inner_split_seed': int(bundle.seed) + int(bundle.target) if domain == 'private' else int(bundle.seed) + sum((ord(character) for character in str(bundle.target))), 'calibration_log_mae': {'base': float(risks[0]), 'memory': float(risks[1]), 'prior': float(risks[2])}, 'gate_crossfit_folds': int(n_gate_folds), 'gate_global_weights': {'base': float(gate_global_weights[0]), 'memory': float(gate_global_weights[1]), 'prior': float(gate_global_weights[2])}, 'gate_crossfit': gate_crossfit_meta, 'self_anchor_crossfit_bias': self_anchor_bias_profile, 'gate_correction_ratio_cap': float(cfg.gate_correction_ratio_cap), 'mean_gate_weights': {'base': float(weights[:, 0].mean()), 'memory': float(weights[:, 1].mean()), 'prior': float(weights[:, 2].mean())}, 'mean_memory_max_similarity': float(np.mean(test_memory_meta['max_similarity'])), 'mean_memory_effective_neighbors': float(np.mean(test_memory_meta['effective_neighbors'])), 'mean_activity_prior_coverage': float(np.mean(test_coverage)), 'support_cases': int(len(set((sample.case_id for sample in bundle.support_samples)))), 'support_prefixes': int(len(bundle.support_samples)), 'test_prefixes': int(len(test_samples)), 'target_parameter_updates': True, 'test_labels_used_by_predictor': False, 'device': str(resolved_device), 'quick': bool(quick)}
    json.dumps(meta, ensure_ascii=False, sort_keys=True)
    return (predictions, meta)
