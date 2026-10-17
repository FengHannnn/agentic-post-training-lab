import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("matched_control_common")


def test_exposure_matches_ids_order_and_tokens(common):
    examples = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    plan = {"common": {"seed": 42, "learning_rate": 1e-5}}
    rows = [{"step": step+1, "pair_index": index, "pair_id": examples[index].id,
             "learning_rate": 1e-5, "chosen_response_tokens": 9+index}
            for step, index in enumerate(common.exposure_order(2, 42))]
    histories = {label: copy.deepcopy(rows) for label in common.ARMS}
    assert common.verify_exposures(histories, examples, plan)
    histories["real_dpo"][0]["pair_id"] = "wrong"
    with pytest.raises(ValueError, match="exposure"):
        common.verify_exposures(histories, examples, plan)


@pytest.mark.parametrize("field,value", [("chosen_response_tokens", 100), ("learning_rate", 1e-4)])
def test_equal_ids_do_not_hide_exposure_or_rate_difference(common, field, value):
    example = SimpleNamespace(id="a")
    rows = [{"step": 1, "pair_index": 0, "pair_id": "a", "chosen_response_tokens": 9, "learning_rate": 1e-5}]
    histories = {label: copy.deepcopy(rows) for label in common.ARMS}
    histories["real_dpo"][0][field] = value
    with pytest.raises(ValueError):
        common.verify_exposures(histories, [example], {"common": {"seed": 42, "learning_rate": 1e-5}})


def cases():
    valid = {"json_valid": True, "schema_valid": True, "route_correct": True, "hit_token_limit": False,
             "end_to_end_pass": False, "semantic_pass": False}
    rows = {"preference_v2_development": [{"id": "a", **valid}], "legacy": [{"id": "b", **valid}]}
    plan = {"evaluation": {"baseline_strict": {"preference_v2_development": 0, "legacy": 0}, "baseline_v2_semantic": 0}}
    return rows, plan


def test_matching_baseline_and_sft_is_not_dpo_added_value(common):
    rows, plan = cases()
    result = common.admission_checks(rows, rows, plan, rows)
    assert result["recovery_pass"] and not result["conditional_dpo_added_value_pass"]


def test_new_gain_without_regression_passes_conditional_check(common):
    rows, plan = cases()
    candidate = copy.deepcopy(rows)
    candidate["preference_v2_development"][0].update(end_to_end_pass=True, semantic_pass=True)
    assert common.admission_checks(rows, candidate, plan, rows)["conditional_dpo_added_value_pass"]


@pytest.mark.parametrize("field,value", [("json_valid", False), ("hit_token_limit", True)])
def test_both_fail_or_equal_totals_do_not_hide_new_failure(common, field, value):
    rows, plan = cases()
    candidate = copy.deepcopy(rows)
    candidate["legacy"][0][field] = value
    assert not common.admission_checks(rows, candidate, plan)["recovery_pass"]
