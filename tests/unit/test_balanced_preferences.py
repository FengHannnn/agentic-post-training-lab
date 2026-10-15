from dataclasses import replace

import pytest

from agentic_post_training.data.balanced_preferences import audit_balanced_design, build_balanced_preferences
from agentic_post_training.data.preferences import validate_preferences


def test_balanced_pairs_have_valid_targets_and_disjoint_numeric_families():
    rows = build_balanced_preferences("fixed-baseline")
    design = audit_balanced_design(rows)
    assert design["train"]["pairs"] == 168
    assert design["train_small"]["pairs"] == 28
    assert design["development"]["pairs"] == 56
    for split in ("train", "train_small"):
        validate_preferences(rows[split], rows["development"], [])


def test_same_counts_do_not_hide_a_duplicate_wording_cell():
    rows = build_balanced_preferences("fixed-baseline")
    row = rows["train"][0]
    rows["train"][0] = replace(row, provenance={**row.provenance, "wording": "natural"})
    with pytest.raises(ValueError, match="factorial cells"):
        audit_balanced_design(rows)


def test_small_subset_must_match_answers_not_just_ids():
    rows = build_balanced_preferences("fixed-baseline")
    row = rows["train_small"][0]
    rows["train_small"][0] = replace(row, rejected=row.rejected + " ")
    with pytest.raises(ValueError, match="exact subset"):
        audit_balanced_design(rows)
