from __future__ import annotations

"""框架实验入口：预处理、元开发选源、目标适配和最终预测。"""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scrm.config import ExperimentConfig, load_settings
from scrm.data.event_logs import (
    PublicSchema, build_public_prefix_samples, build_public_signature,
    chronological_case_split, compare_public_signatures, normalize_public_log,
    read_raw_public_log, select_support_case_ids, split_support_case_ids_generic,
    truncate_at_last_terminal_event,
)
from scrm.data.records import ExperimentBundle
from scrm.evaluation.metrics import regression_metrics
from scrm.models.predictor import predict_final
from scrm.models.support_adaptation import run_support_adaptation
from scrm.selection.learned_loto import _choose_score, _weighted_score, build_loto_weights
from scrm.training.trainer import fit_target_reference
from scrm.training.transfer import (
    load_source_checkpoint, run_transfer_finetune, save_source_checkpoint, train_source_checkpoint,
)


def _hash(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temporary, index=False)
    temporary.replace(path)


def _local_path(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _safe_name(value) -> str:
    value = str(value)
    if not value or any(character in value for character in ('/', '\\', '..')):
        raise ValueError("日志和分组名称不能包含路径分隔符或上级目录")
    return value


def _make_demo_inputs(settings: dict, root: Path) -> None:
    """演示数据由程序生成，不使用或分发任何实验数据。"""
    for group_index, group in enumerate(settings["groups"]):
        for log_index, log in enumerate(group["logs"]):
            path = _local_path(root, log["path"])
            rows = []
            rng = np.random.default_rng(41 + group_index * 11 + log_index)
            for case_index in range(20):
                start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=case_index * 3)
                times = np.cumsum([0.0, *rng.uniform(0.5, 2.0 + log_index, 3)])
                for activity, hours in zip(("Register", "Review", "Approve", "Complete"), times):
                    rows.append({"case_id": f"case-{case_index}", "activity": activity,
                                 "timestamp": (start + pd.Timedelta(hours=float(hours))).strftime("%Y-%m-%d %H:%M:%S.%f")})
            _csv(path, pd.DataFrame(rows))


def prepare_group(group: dict, root: Path, output: Path) -> dict:
    prepared = {}
    for log in group["logs"]:
        name = _safe_name(log["name"])
        path = _local_path(root, log["path"])
        if not path.is_file():
            raise FileNotFoundError(f"请在 YAML 中配置实际日志路径：{path}")
        raw = read_raw_public_log(path)
        schema = PublicSchema(**log["columns"]) if log.get("columns") else None
        raw, completion_audit = truncate_at_last_terminal_event(raw, log.get("terminal_activities", ()), schema)
        events, cases, audit = normalize_public_log(raw, name, schema, activity_lifecycle=False)
        split = chronological_case_split(cases)
        signature_ids = split.loc[split.split.isin(("train", "val")), "case_uid"].astype(int).tolist()
        prepared[name] = {"events": events, "cases": cases, "split": split,
                          "signature": build_public_signature(events, cases, signature_ids)}
        for suffix, frame in (("events", events), ("cases", cases), ("split", split)):
            _csv(output / "prepared" / f"{name}_{suffix}.csv", frame)
        _json(output / "prepared" / f"{name}_audit.json", {**audit, **completion_audit})
    if len(prepared) < 3:
        raise ValueError("Learned-LOTO 至少需要三个流程日志；只有两个日志时无法比较候选源")
    return prepared


def make_bundle(name: str, log: dict, k: int, seed: int, cfg: ExperimentConfig, output: Path):
    split, events, cases = log["split"], log["events"], log["cases"]
    support_ids = select_support_case_ids(split, name, seed, k)
    if len(support_ids) != k:
        raise ValueError(f"{name} 的完整训练案例不足 K={k}")
    inner_train, inner_val = split_support_case_ids_generic(
        support_ids, seed + sum(map(ord, name)), cfg.inner_val_ratio)
    test_ids = split.loc[split.split.eq("test"), "case_uid"].astype(int).tolist()
    if set(support_ids) & set(test_ids):
        raise AssertionError("支持案例与查询案例发生重叠")
    manifest = output / "support" / name / f"k{k}_seed{seed}.csv"
    _csv(manifest, pd.DataFrame({"case_uid": support_ids}))
    build = lambda ids: build_public_prefix_samples(events, cases, ids, cfg.min_prefix_len)
    return ExperimentBundle(
        domain="public", target=name, k=k, seed=seed,
        support_ids=support_ids, inner_train_ids=inner_train, inner_val_ids=inner_val, test_ids=test_ids,
        support_samples=build(support_ids), inner_train_samples=build(inner_train),
        inner_val_samples=build(inner_val), test_samples=build(test_ids), traces={},
        manifest_path=manifest, manifest_sha256=_hash(manifest),
        split_path=output / "prepared" / f"{name}_split.csv",
    )


def _source_checkpoint(name, log, seed, cfg, output, device, resume):
    path = output / "checkpoints" / name / f"seed{seed}.pt"
    if resume and path.exists():
        return load_source_checkpoint(path)
    split, events, cases = log["split"], log["events"], log["cases"]
    train = split.loc[split.split.eq("train"), "case_uid"].astype(int).tolist()
    val = split.loc[split.split.eq("val"), "case_uid"].astype(int).tolist()
    checkpoint = train_source_checkpoint(
        "transformer", build_public_prefix_samples(events, cases, train),
        build_public_prefix_samples(events, cases, val), cfg, seed, device_str=device)
    save_source_checkpoint(checkpoint, path)
    return checkpoint


def _develop_sources(group, prepared, bundles, checkpoints, cfg, output, device, resume):
    """辅助目标的迁移结果用于元开发；当前评估目标的结果不进入其权重学习。"""
    path = output / "source_development.csv"
    rows = pd.read_csv(path, dtype={"k_label": str}).to_dict("records") if resume and path.exists() else []
    completed = {(r["target_process"], str(r["k_label"]), int(r["seed"]), r["source_label"]) for r in rows}
    for (target, k, seed), bundle in bundles.items():
        sources = [name for name in prepared if name != target]
        if all((target, str(k), seed, source) in completed for source in sources):
            continue
        # 目标内参考模型只提供元开发归一化分母，不作为最终预测组件。
        reference, _ = fit_target_reference(
            "transformer", bundle.support_samples, bundle.inner_train_samples,
            bundle.inner_val_samples, bundle.test_samples, cfg, seed, device_str=device)
        reference_mae = regression_metrics(reference)["mae_h"]
        target_signature = build_public_signature(prepared[target]["events"], prepared[target]["cases"], bundle.support_ids)
        for source in sources:
            key = (target, str(k), seed, source)
            if key in completed:
                continue
            print(f"源开发 {source} -> {target}, K={k}, seed={seed}", flush=True)
            prediction, _ = run_transfer_finetune(
                checkpoints[(source, seed)], "transformer", bundle.support_samples,
                bundle.inner_train_samples, bundle.inner_val_samples, bundle.test_samples, cfg, seed, device_str=device)
            mae = regression_metrics(prediction)["mae_h"]
            rows.append({"family": group, "target_process": target, "k_label": str(k), "seed": seed,
                         "source_label": source, "model": "transformer", "mae_h": mae,
                         "transfer_gain_pct": 100 * (1 - mae / max(reference_mae, 1e-12)),
                         "n_source_signature_cases": prepared[source]["signature"]["n_cases"],
                         **compare_public_signatures(prepared[source]["signature"], target_signature)})
            _csv(path, pd.DataFrame(rows))
            completed.add(key)
    return pd.DataFrame(rows)


def run_experiment(config_path: Path, *, root: Path, quick=False, device="auto", resume=True):
    import torch

    root = Path(root).resolve()
    settings = load_settings(config_path)
    if settings.get("synthetic"):
        _make_demo_inputs(settings, root)
    if quick:
        torch.set_num_threads(1)
    base_cfg = ExperimentConfig(**settings["training"])
    cfg = base_cfg.training_config(quick)
    if quick:
        cfg = replace(cfg, source_weight_grid_step=1.0)
    output = root / "outputs" / _safe_name(settings.get("experiment", "experiment"))
    input_hashes = {str(log["path"]): _hash(_local_path(root, log["path"]))
                    for group in settings["groups"] for log in group["logs"]}
    fingerprint_input = {"settings": settings, "config": asdict(cfg), "quick": bool(quick), "device": device,
                         "inputs": input_hashes,
                         "code": {str(p.relative_to(Path(__file__).parents[1])): _hash(p)
                                  for p in Path(__file__).parents[1].rglob("*.py")}}
    fingerprint = sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    manifest = output / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("配置、数据或代码已改变，请使用新的 experiment 名称，避免复用旧缓存")
    _json(manifest, {"fingerprint": fingerprint, "quick_check": bool(quick), "training": asdict(cfg),
                     "K": settings["K"], "seeds": settings["seeds"], "input_sha256": input_hashes,
                     "selector_scope": "original_meta_target_exclusion"})
    results_path = output / "results.csv"
    results = pd.read_csv(results_path).to_dict("records") if resume and results_path.exists() else []
    completed = {(r["group"], r["target"], int(r["K"]), int(r["seed"])) for r in results}
    for group in settings["groups"]:
        family = _safe_name(group["name"])
        group_output = output / family
        prepared = prepare_group(group, root, group_output)
        bundles = {(name, int(k), int(seed)): make_bundle(name, log, int(k), int(seed), cfg, group_output)
                   for name, log in prepared.items() for k in settings["K"] for seed in settings["seeds"]}
        checkpoints = {(name, int(seed)): _source_checkpoint(name, log, int(seed), cfg, group_output, device, resume)
                       for name, log in prepared.items() for seed in settings["seeds"]}
        development = _develop_sources(family, prepared, bundles, checkpoints, cfg, group_output, device, resume)
        weights = build_loto_weights(development, {"families": [family], "models": ["transformer"],
                                                  "weight_grid_step": cfg.source_weight_grid_step})
        _csv(group_output / "selection_weights.csv", weights)
        for (target, k, seed), bundle in bundles.items():
            cell = (family, target, k, seed)
            prediction_path = group_output / "predictions" / target / f"k{k}_seed{seed}.csv"
            if resume and cell in completed and prediction_path.exists():
                continue
            candidates = development.loc[development.target_process.eq(target)
                & development.k_label.astype(str).eq(str(k)) & development.seed.eq(seed)].copy()
            weight_row = weights.loc[weights.heldout_target.eq(target)].iloc[0]
            from scrm.data.event_logs import SIM_COMPONENTS
            frozen_weights = np.asarray([weight_row[f"w_{view}"] for view in SIM_COMPONENTS])
            candidates["selection_score"] = _weighted_score(candidates, frozen_weights)
            chosen = _choose_score(candidates, "selection_score")
            if chosen is None:
                raise ValueError(f"没有可用的候选源：{family}/{target}")
            source = str(chosen.source_label)
            print(f"SCRM {family}/{target}, K={k}, seed={seed}, source={source}", flush=True)
            heads, metadata = run_support_adaptation(
                cfg, family, bundle, checkpoint=checkpoints[(source, seed)],
                source_metadata={"selected_source": source}, device=device, quick=quick)
            final_prediction, final_metadata = predict_final(
                heads["base"].y_pred_hours.to_numpy(float), heads["prior"].y_pred_hours.to_numpy(float),
                metadata, bundle.support_samples, bundle.test_samples)
            prediction = heads["base"].copy()
            prediction["y_pred_hours"] = final_prediction
            _csv(prediction_path, prediction)
            _json(prediction_path.with_suffix(".json"), {"adaptation": metadata, "calibration": final_metadata})
            results = [r for r in results if (r["group"], r["target"], int(r["K"]), int(r["seed"])) != cell]
            results.append({"group": family, "target": target, "K": k, "seed": seed,
                            "selected_source": source, "model": "scrm_self_anchor",
                            "quick_check": bool(quick), **regression_metrics(prediction)})
            _csv(results_path, pd.DataFrame(results))
    frame = pd.DataFrame(results)
    per_k = frame.groupby(["group", "target", "K"], as_index=False).agg(mae_h=("mae_h", "mean"))
    summary = per_k.groupby(["group", "target"], as_index=False).agg(mae_h=("mae_h", "mean"))
    _csv(output / "summary.csv", summary)
    print(summary.to_string(index=False))
    return frame
