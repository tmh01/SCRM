# 小时尺度误差与案例宏平均指标。
from __future__ import annotations
'Lightweight RTP regression metrics.\n\nThis module intentionally depends only on NumPy/Pandas so analysis-only phases\n(e.g. Phase 1.5) do not import PyTorch/XGBoost and their OpenMP runtimes.\n'
import numpy as np
import pandas as pd

def regression_metrics(pred_df: pd.DataFrame, eps: float=1e-08) -> dict:
    y = pred_df['y_true_hours'].to_numpy(float)
    p = pred_df['y_pred_hours'].to_numpy(float)
    ae = np.abs(y - p)
    mae = float(np.mean(ae)) if len(ae) else float('nan')
    medae = float(np.median(ae)) if len(ae) else float('nan')
    rmse = float(np.sqrt(np.mean((y - p) ** 2))) if len(ae) else float('nan')
    nmae = float(mae / max(float(np.mean(y)), eps)) if len(ae) else float('nan')
    tmp = pred_df.copy()
    tmp['ae'] = np.abs(tmp['y_true_hours'] - tmp['y_pred_hours'])
    by_case = tmp.groupby('case_id', sort=False).agg(case_mae_h=('ae', 'mean'), case_duration_h=('case_duration_hours', 'first'))
    case_macro_mae = float(by_case['case_mae_h'].mean()) if len(by_case) else float('nan')
    case_macro_nmae = float((by_case['case_mae_h'] / by_case['case_duration_h'].clip(lower=eps)).mean()) if len(by_case) else float('nan')
    return {'mae_h': mae, 'median_ae_h': medae, 'rmse_h': rmse, 'nmae': nmae, 'case_macro_mae_h': case_macro_mae, 'case_macro_nmae': case_macro_nmae, 'n_test_prefixes': int(len(pred_df)), 'n_test_cases': int(pred_df['case_id'].nunique())}

def bootstrap_case_macro_mae_ci(pred_df: pd.DataFrame, seed: int, n_boot: int=1000) -> tuple[float, float]:
    tmp = pred_df.copy()
    tmp['ae'] = np.abs(tmp['y_true_hours'] - tmp['y_pred_hours'])
    case_mae = tmp.groupby('case_id')['ae'].mean().to_numpy(float)
    if len(case_mae) <= 1:
        x = float(case_mae[0]) if len(case_mae) else float('nan')
        return (x, x)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals[i] = rng.choice(case_mae, size=len(case_mae), replace=True).mean()
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
