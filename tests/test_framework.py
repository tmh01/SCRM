from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from scrm.config import ExperimentConfig
from scrm.data.event_logs import (
    build_public_prefix_samples, chronological_case_split, normalize_public_log,
    select_support_case_ids,
)
from scrm.models.backbones import TransformerRegressor
from scrm.models.conditional_survival import fit_support_survival, support_survival_predictions
from scrm.models.predictor import predict_final
from scrm.models.risk_calibration import select_risk_anchor
from scrm.models.self_anchor import support_prior_self_anchor
from scrm.models.support_adaptation import residual_memory_predictions, robust_prior_predictions
from scrm.selection.learned_loto import _choose_score, _weighted_score, fit_loto


def _samples():
    from scrm.data.records import PrefixSample

    samples = []
    for case_id, duration in ((1, 10.), (2, 12.), (3, 11.)):
        for length, elapsed in ((1, 0.), (2, 4.)):
            samples.append(PrefixSample(case_id, str(case_id), 1, ["A", "B"][:length],
                np.zeros((length, 6), np.float32), length, elapsed, elapsed,
                duration - elapsed, duration, pd.Timestamp("2024-01-01") + pd.Timedelta(hours=elapsed)))
    return samples


def test_original_hyperparameters_are_retained():
    cfg = ExperimentConfig()
    assert (cfg.embedding_dim, cfg.hidden_dim, cfg.transformer_layers) == (32, 64, 2)
    assert (cfg.gate_folds, cfg.finetune_learning_rate) == (3, 3e-4)


def test_transformer_outputs_a_shared_prefix_representation():
    torch.manual_seed(42)
    model = TransformerRegressor(5).eval()
    tokens = torch.tensor([[2, 3, 0], [2, 3, 4]])
    times = torch.zeros(2, 3, 6)
    lengths = torch.tensor([2, 3])
    with torch.no_grad():
        representation = model.encode(tokens, times, lengths)
        assert representation.shape == (2, 64)
        assert torch.allclose(model(tokens, times, lengths), model.head(representation).squeeze(-1))


def test_self_anchor_is_nonnegative_and_within_trust_region():
    base = np.array([0., 5., 50.])
    result = support_prior_self_anchor(base, np.array([100., 1000., 0.]))
    assert np.all(result >= 0)
    assert np.all((result + 1) / (base + 1) <= 2 + 1e-10)
    assert np.all((result + 1) / (base + 1) >= .5 - 1e-10)


@pytest.mark.parametrize("risk,activated,bias,cap", [(0.5, False, -.08, 2), (1.2, True, -.3, 3)])
def test_risk_controls_bias_and_dynamic_trust(risk, activated, bias, cap):
    choice, meta = select_risk_anchor({"gate_crossfit": {"crossfit_case_macro_log_mae": risk},
                                      "self_anchor_crossfit_bias": {"consensus_bias": -.3}})
    assert (choice.activated, choice.global_log_bias, choice.trust_ratio_cap) == (activated, bias, cap)
    assert not meta["test_labels_used"]


def test_final_head_does_not_use_query_outcomes():
    support = _samples()
    query = [_samples()[0]]
    from dataclasses import replace
    query = [replace(query[0], case_id=99)]
    changed = [replace(query[0], remaining_hours=1e9, case_duration_hours=1e9)]
    meta = {"gate_crossfit": {"crossfit_case_macro_log_mae": 1.2},
            "self_anchor_crossfit_bias": {"consensus_bias": -.2}}
    first, _ = predict_final(np.array([10.]), np.array([11.]), meta, support, query)
    second, _ = predict_final(np.array([10.]), np.array([11.]), meta, support, changed)
    assert np.array_equal(first, second)


def test_support_prior_and_survival_only_fit_completed_support():
    support = _samples()
    choice, meta = fit_support_survival(support)
    prior, _ = robust_prior_predictions(support, support[:2])
    prediction = support_survival_predictions(support, support[:2], choice)
    assert prior.shape == prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert not meta["test_labels_used"]
    assert "query_labels" not in inspect.signature(support_survival_predictions).parameters


def test_data_splits_are_case_level_and_support_excludes_test():
    rows = [{"case_id": str(case), "activity": f"A{event}",
             "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=case, hours=event)}
            for case in range(12) for event in range(3)]
    events, cases, _ = normalize_public_log(pd.DataFrame(rows), "demo")
    split = chronological_case_split(cases)
    support = select_support_case_ids(split, "demo", 42, 2)
    test = set(split.loc[split.split.eq("test"), "case_uid"])
    assert len(support) == 2 and not set(support) & test
    samples = build_public_prefix_samples(events, cases, support)
    assert samples and all(sample.remaining_hours > 0 for sample in samples)


def test_original_uninformative_view_tie_rule_is_preserved():
    from scrm.data.event_logs import SIM_COMPONENTS
    frame = pd.DataFrame([{**dict.fromkeys(SIM_COMPONENTS, 0.), "source_label": "small", "n_source_signature_cases": 1},
                          {**dict.fromkeys(SIM_COMPONENTS, 0.), "source_label": "large", "n_source_signature_cases": 100}])
    frame["score"] = _weighted_score(frame, np.ones(6) / 6)
    assert _choose_score(frame, "score").source_label == "large"


def test_synthetic_pipeline_runs_without_research_repository(tmp_path):
    from pathlib import Path
    from scrm.experiments.runner import run_experiment

    configuration = Path(__file__).resolve().parents[1] / "configs/demo.yaml"
    result = run_experiment(configuration, root=tmp_path, quick=True, device="cpu")
    assert len(result) == 3
    assert result.model.eq("scrm_self_anchor").all()
    assert np.isfinite(result.mae_h).all()
    assert result.quick_check.all()
    resumed = run_experiment(configuration, root=tmp_path, quick=True, device="cpu")
    assert len(resumed) == len(result)
