# 支持集条件生存估计及可靠性融合。
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence
import numpy as np

@dataclass(frozen=True)
class SupportSurvivalChoice:
    """支持集条件生存头的配置。"""
    compatibility: str
    quantile: float
    min_compatible_cases: int = 2

def _case_summary(support_samples: Sequence) -> dict:
    if not support_samples:
        raise ValueError('条件生存头至少需要一个支持前缀')
    by_case: dict[int, list] = {}
    for sample in support_samples:
        by_case.setdefault(int(sample.case_id), []).append(sample)
    case_ids = np.asarray(sorted(by_case), dtype=np.int64)
    durations, activities, transitions, trace_lengths = ([], [], [], [])
    for case_id in case_ids:
        samples = by_case[int(case_id)]
        last = max(samples, key=lambda sample: int(sample.prefix_len))
        names = tuple((str(value) for value in last.activity_names))
        durations.append(float(last.case_duration_hours))
        activities.append(set(names))
        transitions.append(set(zip(names[:-1], names[1:])))
        trace_lengths.append(int(last.prefix_len) + 1)
    return {'case_ids': case_ids, 'durations': np.asarray(durations, dtype=float), 'activities': activities, 'transitions': transitions, 'trace_lengths': np.asarray(trace_lengths, dtype=float)}

def _remaining_table(summary: dict, query_samples: Sequence, compatibility: str, *, min_compatible_cases: int, exclude_same_case: bool) -> np.ndarray:
    elapsed = np.asarray([max(float(sample.elapsed_hours), 0.0) for sample in query_samples])
    remaining = summary['durations'][None, :] - elapsed[:, None]
    legal = remaining > 0.0
    if exclude_same_case:
        query_cases = np.asarray([int(sample.case_id) for sample in query_samples], dtype=np.int64)
        legal &= query_cases[:, None] != summary['case_ids'][None, :]
    else:
        query_cases = None
    empty = ~legal.any(axis=1)
    if np.any(empty):
        if exclude_same_case:
            legal[empty] = query_cases[empty, None] != summary['case_ids'][None, :]
        else:
            legal[empty] = True
    compatible = legal.copy()
    if compatibility == 'activity':
        compatible &= np.asarray([[str(sample.activity_names[-1]) in activities if sample.activity_names else False for activities in summary['activities']] for sample in query_samples], dtype=bool)
    elif compatibility == 'transition':
        compatible &= np.asarray([[len(sample.activity_names) > 1 and (str(sample.activity_names[-2]), str(sample.activity_names[-1])) in transitions for transitions in summary['transitions']] for sample in query_samples], dtype=bool)
    elif compatibility != 'all':
        raise ValueError(f'未知的生存兼容模式: {compatibility}')
    enough = compatible.sum(axis=1) >= int(min_compatible_cases)
    selected = np.where(enough[:, None], compatible, legal)
    return np.where(selected, np.maximum(remaining, 0.0), np.nan)

def _quantile_prediction(table: np.ndarray, quantile: float) -> np.ndarray:
    prediction = np.nanquantile(table, float(quantile), axis=1)
    if not np.isfinite(prediction).all():
        raise ValueError('条件生存头产生了非有限预测')
    return np.maximum(np.asarray(prediction, dtype=float), 0.0)

def fit_support_survival(support_samples: Sequence, *, compatibilities: Iterable[str]=('all', 'activity', 'transition'), quantiles: Iterable[float]=(0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75), min_compatible_cases: int=2) -> tuple[SupportSurvivalChoice, dict]:
    """用案例留一 MAE 选择条件和分位点，全程不读取测试标签。"""
    samples = tuple(support_samples)
    summary = _case_summary(samples)
    if len(summary['case_ids']) < 2:
        raise ValueError('条件生存头留一选择至少需要两个支持案例')
    target = np.asarray([float(sample.remaining_hours) for sample in samples], dtype=float)
    best: tuple[tuple[float, float, int], SupportSurvivalChoice, np.ndarray] | None = None
    mode_order = {'all': 0, 'activity': 1, 'transition': 2}
    mode_best: dict[str, float] = {}
    for compatibility in tuple(compatibilities):
        table = _remaining_table(summary, samples, compatibility, min_compatible_cases=int(min_compatible_cases), exclude_same_case=True)
        local_best = float('inf')
        for quantile in tuple(quantiles):
            prediction = _quantile_prediction(table, float(quantile))
            mae = float(np.mean(np.abs(target - prediction)))
            key = (mae, abs(float(quantile) - 0.5), mode_order[compatibility])
            choice = SupportSurvivalChoice(compatibility=str(compatibility), quantile=float(quantile), min_compatible_cases=int(min_compatible_cases))
            if best is None or key < best[0]:
                best = (key, choice, prediction)
            local_best = min(local_best, mae)
        mode_best[str(compatibility)] = float(local_best)
    if best is None:
        raise RuntimeError('条件生存头没有产生有效候选')
    durations = summary['durations']
    duration_mean = max(float(np.mean(durations)), 1e-12)
    duration_cv = float(np.std(durations) / duration_mean)
    return (best[1], {'support_crossfit_mae': float(best[0][0]), 'mode_best_mae': mode_best, 'choice': asdict(best[1]), 'duration_cv': duration_cv, 'duration_median_hours': float(np.median(durations)), 'mean_trace_length': float(np.mean(summary['trace_lengths'])), 'median_hours_per_event': float(np.median(durations) / max(float(np.mean(summary['trace_lengths'])), 1.0)), 'support_cases': int(len(summary['case_ids'])), 'support_prefixes': int(len(samples)), 'test_labels_used': False})

def support_survival_predictions(support_samples: Sequence, query_samples: Sequence, choice: SupportSurvivalChoice, *, batch_size: int=4096) -> np.ndarray:
    """用选定的条件生存头预测查询前缀。"""
    summary = _case_summary(tuple(support_samples))
    samples = tuple(query_samples)
    outputs = []
    for start in range(0, len(samples), int(batch_size)):
        batch = samples[start:start + int(batch_size)]
        table = _remaining_table(summary, batch, choice.compatibility, min_compatible_cases=choice.min_compatible_cases, exclude_same_case=False)
        outputs.append(_quantile_prediction(table, choice.quantile))
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=float)

def survival_reliability_weight(meta: dict) -> float:
    """根据支持案例时长稳定性决定生存头是否足够可信。"""
    duration_cv = float(meta['duration_cv'])
    mean_trace_length = float(meta['mean_trace_length'])
    if 'median_hours_per_event' in meta:
        hours_per_event = float(meta['median_hours_per_event'])
    else:
        hours_per_event = float(meta['duration_median_hours']) / max(mean_trace_length, 1.0)
    duration_score = float(np.clip((0.3 - duration_cv) / 0.08, 0.0, 1.0))
    trace_score = float(np.clip((9.0 - mean_trace_length) / 3.0, 0.0, 1.0))
    horizon_score = float(np.clip((hours_per_event - 80.0) / 40.0, 0.0, 1.0))
    return duration_score * trace_score * horizon_score

def adaptive_survival_anchor(anchor_prediction: np.ndarray, survival_prediction: np.ndarray, survival_meta: dict) -> tuple[np.ndarray, float]:
    """以支持集可靠性为门控，在自锚定主头与条件生存头间平滑切换。"""
    anchor = np.asarray(anchor_prediction, dtype=float)
    survival = np.asarray(survival_prediction, dtype=float)
    if anchor.shape != survival.shape:
        raise ValueError('自锚定预测与条件生存预测形状不一致')
    weight = survival_reliability_weight(survival_meta)
    output = (1.0 - weight) * anchor + weight * survival
    return (np.maximum(output, 0.0), float(weight))
