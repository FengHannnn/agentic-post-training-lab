import pytest

from agentic_post_training.evaluation.preference_diagnostics import grouped_preference_changes, summarize_preference_changes


def row(chosen, rejected, category="math"):
    return {"category": category, "delta_chosen": chosen, "delta_rejected": rejected,
            "relative_margin": 0.1 * (chosen - rejected), "loss": 0.5,
            "sft_chosen_log_prob": -5, "dpo_chosen_log_prob": -5 + chosen,
            "sft_rejected_log_prob": -5, "dpo_rejected_log_prob": -5 + rejected}


def test_suppression_can_improve_relative_preference():
    result = summarize_preference_changes([row(-1, -4), row(2, 1)])
    assert result["relative_improved"] == 2
    assert result["chosen_probability_increased"] == 1
    assert result["chosen_decreased_but_relative_improved"] == 1
    assert result["delta_chosen"] == 0.5


def test_grouping_preserves_counts():
    result = grouped_preference_changes([row(1, 0), row(1, 0), row(-1, 0, "greeting")], "category")
    assert result["math"]["pairs"] == 2
    assert result["greeting"]["relative_improved"] == 0


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize_preference_changes([row(float("nan"), 0)])
