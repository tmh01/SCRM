# 保留原实验的选源评分、LOTO 权重学习与同分规则；未加入后续作废修复。
from __future__ import annotations
import hashlib
import math
import numpy as np
import pandas as pd
from scrm.data.event_logs import SIM_COMPONENTS

def _simplex_weights(n: int, step: float):
    units = int(round(1.0 / float(step)))
    if units <= 0 or abs(units * step - 1.0) > 1e-08:
        raise ValueError('weight_grid_step must divide 1 exactly')

    def rec(prefix, remaining, slots):
        if slots == 1:
            yield (prefix + [remaining])
            return
        for v in range(remaining + 1):
            yield from rec(prefix + [v], remaining - v, slots - 1)
    for ints in rec([], units, n):
        yield (np.asarray(ints, float) / units)

def _weighted_score(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    X = df[list(SIM_COMPONENTS)].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    mask = np.isfinite(X)
    z = np.where(mask, X, 0.0) * w.reshape(1, -1)
    den = (mask * w.reshape(1, -1)).sum(axis=1)
    return np.divide(z.sum(axis=1), den, out=np.full(len(df), np.nan), where=den > 0)

def _choose_score(g: pd.DataFrame, score_col: str):
    x = g.dropna(subset=[score_col]).copy()
    if x.empty:
        return None
    return x.sort_values([score_col, 'n_source_signature_cases', 'source_label'], ascending=[False, False, True], kind='mergesort').iloc[0]

def _score_weight_meta(meta: pd.DataFrame, w: np.ndarray):
    x = meta.copy()
    x['_score'] = _weighted_score(x, w)
    regrets = []
    matches = []
    for _, g in x.groupby(['family', 'target_process', 'k_label', 'seed', 'model'], sort=False):
        g = g.dropna(subset=['_score', 'transfer_gain_pct'])
        if len(g) < 2:
            continue
        ch = _choose_score(g, '_score')
        oracle = g.loc[pd.to_numeric(g['transfer_gain_pct'], errors='coerce').idxmax()]
        regrets.append((str(g['target_process'].iloc[0]), float(oracle['transfer_gain_pct'] - ch['transfer_gain_pct'])))
        matches.append((str(g['target_process'].iloc[0]), int(str(ch['source_label']) == str(oracle['source_label']))))
    if not regrets:
        return (np.inf, 0.0, np.inf)
    rd = pd.DataFrame(regrets, columns=['target', 'regret']).groupby('target')['regret'].mean()
    md = pd.DataFrame(matches, columns=['target', 'match']).groupby('target')['match'].mean()
    l2 = float(((w - 1 / len(w)) ** 2).sum())
    return (float(rd.mean()), float(md.mean()), l2)

def fit_loto(sim: pd.DataFrame, family: str, heldout: str, model: str, step: float):
    meta = sim[(sim['family'].astype(str) == str(family)) & (sim['target_process'].astype(str) != str(heldout)) & (sim['model'].astype(str) == str(model))].copy()
    if meta['target_process'].nunique() < 2:
        w = np.repeat(1 / len(SIM_COMPONENTS), len(SIM_COMPONENTS))
        return (w, np.nan, np.nan, True)
    best = None
    for w in _simplex_weights(len(SIM_COMPONENTS), step):
        regret, match, l2 = _score_weight_meta(meta, w)
        key = (round(regret, 12), -round(match, 12), round(l2, 12), tuple(w.tolist()))
        if best is None or key < best[0]:
            best = (key, w.copy(), regret, match)
    return (best[1], best[2], best[3], False)

def build_loto_weights(sim: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    rows = []
    step = float(protocol.get('weight_grid_step', 0.1))
    for family in protocol['families']:
        targets = sorted(sim[sim['family'].astype(str) == str(family)]['target_process'].astype(str).unique())
        for model in protocol['models']:
            for target in targets:
                w, regret, match, fallback = fit_loto(sim, family, target, model, step)
                row = {'family': family, 'heldout_target': target, 'model': model, 'weight_grid_step': step, 'meta_mean_regret_pct': regret, 'meta_oracle_match_rate': match, 'fallback_equal6': fallback}
                row.update({f'w_{c}': float(v) for c, v in zip(SIM_COMPONENTS, w)})
                rows.append(row)
    return pd.DataFrame(rows)
