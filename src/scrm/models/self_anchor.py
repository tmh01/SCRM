# 对数空间分歧门控与有界自锚定校正。
from __future__ import annotations
import numpy as np

def support_prior_self_anchor(base_prediction: np.ndarray, support_prior_prediction: np.ndarray, *, max_support_prior_log_weight: float=1.0, disagreement_temperature: float=1.25, global_log_bias: float=-0.08, trust_ratio_cap: float=2.0) -> np.ndarray:
    base = np.asarray(base_prediction, dtype=float)
    prior = np.asarray(support_prior_prediction, dtype=float)
    max_prior_weight = float(max_support_prior_log_weight)
    temperature = float(disagreement_temperature)
    log_bias = float(global_log_bias)
    ratio_cap = float(trust_ratio_cap)
    if base.shape != prior.shape:
        raise ValueError('SCRM self-anchor inputs must be aligned')
    if base.ndim != 1:
        raise ValueError('SCRM self-anchor expects one-dimensional predictions')
    if not np.isfinite(base).all() or not np.isfinite(prior).all():
        raise ValueError('SCRM self-anchor received a non-finite prediction')
    if np.any(base < 0.0) or np.any(prior < 0.0):
        raise ValueError('SCRM self-anchor predictions must be non-negative')
    if not 0.0 <= max_prior_weight <= 1.0:
        raise ValueError('SCRM self-anchor maximum prior weight must be in [0, 1]')
    if temperature <= 0.0:
        raise ValueError('SCRM self-anchor disagreement temperature must be positive')
    if not np.isfinite(log_bias):
        raise ValueError('SCRM self-anchor global log bias must be finite')
    if ratio_cap <= 1.0:
        raise ValueError('SCRM self-anchor trust ratio must be greater than one')
    base_log = np.log1p(base)
    prior_log = np.log1p(prior)
    disagreement = np.abs(prior_log - base_log)
    prior_weight = max_prior_weight / (1.0 + disagreement / temperature)
    corrected_log = base_log + prior_weight * (prior_log - base_log) + log_bias
    radius = np.log(ratio_cap)
    corrected_log = np.clip(corrected_log, base_log - radius, base_log + radius)
    return np.maximum(np.expm1(corrected_log), 0.0)
