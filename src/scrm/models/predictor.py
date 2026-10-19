from __future__ import annotations

"""最终预测头：风险校准锚点与支持集条件生存估计。"""

import numpy as np

from .conditional_survival import adaptive_survival_anchor, fit_support_survival, support_survival_predictions
from .risk_calibration import risk_calibrated_anchor_predictions, select_risk_anchor


def predict_final(base_prediction, prior_prediction, support_metadata, support_samples, query_samples):
    choice, risk_metadata = select_risk_anchor(support_metadata)
    anchor = risk_calibrated_anchor_predictions(base_prediction, prior_prediction, choice)
    survival_choice, survival_metadata = fit_support_survival(support_samples)
    survival = support_survival_predictions(support_samples, query_samples, survival_choice)
    prediction, weight = adaptive_survival_anchor(anchor, survival, survival_metadata)
    return np.asarray(prediction), {
        "risk_calibration": risk_metadata,
        "survival": survival_metadata,
        "survival_weight": float(weight),
        "query_labels_used": False,
    }
