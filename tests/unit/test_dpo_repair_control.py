import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def control(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    verify = importlib.import_module("compare_dpo_repair").verify_control
    original = {"config": {"learning_rate": 1e-4, "max_steps": 2, "experiment_id": "pure"}}
    for key in ("model", "resolved_revision", "baseline_manifest_sha256", "data_manifest",
                "initial_policy_sha256", "initial_reference_sha256", "before", "training_token_counts"):
        original[key] = "fixed"
    for key in ("reference_unchanged", "reference_no_grad", "policy_frozen_parameters_unchanged",
                "policy_updated", "initial_policy_reference_match", "reload_logits_match"):
        original[key] = True
    repaired = copy.deepcopy(original)
    repaired["config"].update(chosen_nll_weight=1.0, experiment_id="repair")
    history = [{"step": step, "pair_index": step, "pair_id": str(step), "epoch": 1,
                "chosen_response_tokens": 5, "rejected_response_tokens": 6,
                "input_tokens": 20, "padded_input_tokens": 22, "learning_rate": 1e-4} for step in (1, 2)]
    return verify, original, repaired, history


def test_repair_allows_only_declared_objective_change(control):
    verify, original, repaired, history = control
    verify(original, repaired, history, history)
    repaired["config"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="setup"):
        verify(original, repaired, history, history)


def test_repair_weight_must_match_the_declared_experiment(control):
    verify, original, repaired, history = control
    repaired["config"]["chosen_nll_weight"] = 5.0
    with pytest.raises(ValueError, match="declared repair"):
        verify(original, repaired, history, history)
    verify(original, repaired, history, history, expected_anchor_weight=5.0)


def test_learning_rate_control_preserves_anchor_and_all_other_settings(control):
    verify, original, repaired, history = control
    original["config"]["chosen_nll_weight"] = 5.0
    repaired["config"].update(chosen_nll_weight=5.0, learning_rate=2e-5)
    changed = copy.deepcopy(history)
    for row in changed:
        row["learning_rate"] = 2e-5
    verify(original, repaired, history, changed, 5.0, 2e-5)
    repaired["config"]["chosen_nll_weight"] = 1.0
    with pytest.raises(ValueError, match="setup"):
        verify(original, repaired, history, changed, 5.0, 2e-5)


def test_learning_rate_control_checks_the_actual_step_log(control):
    verify, original, repaired, history = control
    original["config"]["chosen_nll_weight"] = 5.0
    repaired["config"].update(chosen_nll_weight=5.0, learning_rate=2e-5)
    with pytest.raises(ValueError, match="training learning rate"):
        verify(original, repaired, history, history, 5.0, 2e-5)


def test_repair_rejects_changed_exposure_order(control):
    verify, original, repaired, history = control
    changed = copy.deepcopy(history)
    changed[0]["pair_index"] = 9
    with pytest.raises(ValueError, match="exposure"):
        verify(original, repaired, history, changed)


def test_repair_rejects_changed_start_or_failed_reference_invariant(control):
    verify, original, repaired, history = control
    repaired["initial_policy_sha256"] = "other"
    with pytest.raises(ValueError, match="initial_policy"):
        verify(original, repaired, history, history)
    repaired["initial_policy_sha256"] = original["initial_policy_sha256"]
    repaired["reference_unchanged"] = False
    with pytest.raises(ValueError, match="reference_unchanged"):
        verify(original, repaired, history, history)
