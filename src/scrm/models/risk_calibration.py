# 支持集风险、自适应偏置和动态信任域。
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scrm.models.self_anchor import support_prior_self_anchor

@dataclass(frozen=True)
class RiskAnchorChoice:
    """由目标支持集折外风险决定的 Self-Anchor 参数。"""
    activated: bool
    crossfit_log_risk: float
    global_log_bias: float
    trust_ratio_cap: float
    risk_threshold: float

def select_risk_anchor(support_metadata: dict, *, risk_threshold: float=1.0, default_log_bias: float=-0.08, default_trust_ratio_cap: float=2.0, high_risk_trust_ratio_cap: float=3.0) -> tuple[RiskAnchorChoice, dict]:
    """只根据 K-shot 支持集的折外误差启用风险校准。

    低风险单元严格保留原 Self-Anchor；高风险单元使用支持集三种目标
    共同确定的中位偏置，并适度放宽信任域，让纠偏不再被固定边界截断。
    """
    gate_meta = support_metadata.get('gate_crossfit', {})
    risk = float(gate_meta.get('crossfit_case_macro_log_mae', np.nan))
    if not np.isfinite(risk):
        raise ValueError('Phase 10 元信息缺少有效的支持集折外风险')
    activated = bool(risk >= float(risk_threshold))
    if activated:
        profile = support_metadata.get('self_anchor_crossfit_bias')
        if not isinstance(profile, dict):
            raise ValueError('高风险单元缺少支持集 Self-Anchor 偏置画像')
        bias = float(profile['consensus_bias'])
        trust_cap = float(high_risk_trust_ratio_cap)
    else:
        profile = None
        bias = float(default_log_bias)
        trust_cap = float(default_trust_ratio_cap)
    choice = RiskAnchorChoice(activated=activated, crossfit_log_risk=risk, global_log_bias=bias, trust_ratio_cap=trust_cap, risk_threshold=float(risk_threshold))
    return (choice, {'choice': asdict(choice), 'bias_profile': profile, 'selection_scope': 'target_k_shot_support_crossfit_only', 'external_model_predictions_used': [], 'test_labels_used': False})

def risk_calibrated_anchor_predictions(base_prediction: np.ndarray, prior_prediction: np.ndarray, choice: RiskAnchorChoice, *, max_support_prior_log_weight: float=1.0, disagreement_temperature: float=1.25) -> np.ndarray:
    """应用已经由支持集选定的风险偏置和信任域。"""
    return support_prior_self_anchor(base_prediction, prior_prediction, max_support_prior_log_weight=max_support_prior_log_weight, disagreement_temperature=disagreement_temperature, global_log_bias=choice.global_log_bias, trust_ratio_cap=choice.trust_ratio_cap)
