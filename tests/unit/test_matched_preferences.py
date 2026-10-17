import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_post_training.data.balanced_preferences import build_balanced_preferences


@pytest.fixture
def design(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("build_matched_preferences")
    original = build_balanced_preferences("fixed")["train"][:2]
    candidates = [replace(row, rejected="{broken}", provenance={**row.provenance,
        "rejected_source": "frozen_sft_greedy_training_error", "original_synthetic_rejected": row.rejected}) for row in original]
    return module, original, candidates


def test_all_arms_keep_same_input_chosen_and_order(design):
    module, original, candidates = design
    result = module.matched_rows(candidates[::-1], original)
    assert result["train_real"] == candidates[::-1]
    assert result["train_synthetic"] == original[::-1]
    assert [row.response for row in result["train_chosen"]] == [row.chosen for row in candidates[::-1]]
    assert [row.id for row in result["train_chosen"]] == [row.id for row in candidates[::-1]]


@pytest.mark.parametrize("field,value", [("chosen", "{}"), ("prompt", "changed input"), ("family_id", "other")])
def test_same_ids_cannot_hide_changed_inputs_or_chosen(design, field, value):
    module, original, candidates = design
    candidates[0] = replace(candidates[0], **{field: value})
    with pytest.raises(ValueError, match="matched input"):
        module.matched_rows(candidates, original)


def test_untraceable_synthetic_negative_is_rejected(design):
    module, original, candidates = design
    candidates[0] = replace(candidates[0], provenance={**candidates[0].provenance, "original_synthetic_rejected": "{}"})
    with pytest.raises(ValueError, match="provenance"):
        module.matched_rows(candidates, original)


def test_same_candidate_cannot_be_duplicated(design):
    module, original, candidates = design
    with pytest.raises(ValueError, match="unique IDs"):
        module.matched_rows([candidates[0], candidates[0]], original)


def test_plan_bounds_experiments_and_discloses_compute_difference(design):
    module, _, _ = design
    plan = module.make_plan("baseline", {}, {}, {}, "history")
    assert plan["budget"]["training_runs"] == 3
    assert plan["budget"]["extra_hyperparameter_trials"] == 0
    assert plan["budget"]["same_response_token_compute"] is False
    assert plan["common"]["max_steps"] == 38
    assert plan["arms"]["real_dpo"]["chosen_nll_weight"] == 0
    assert plan["arms"]["synthetic_dpo"]["chosen_nll_weight"] == 0
    assert plan["evaluation"]["final_unseen_test"] is False
