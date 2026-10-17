from dataclasses import replace

import pytest

from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups


def sample(index=0):
    return RolloutSample("group", "case", index, 42+index, [1, 2], [4, 0], [-0.4, -0.5],
        [-0.4, -0.5], 0, "eos", "answer", "answer<eos>")


def test_eos_as_pad_retains_real_eos_and_masks_only_padding():
    short = sample()
    long = replace(sample(1), response_token_ids=[3, 4, 0], policy_token_log_probs=[-1., -2., -3.],
        behavior_token_log_probs=[-1., -2., -3.])
    packed = pack_rollout_groups([[short, long]], 0)
    assert packed["response_token_ids"].tolist() == [[[4, 0, 0], [3, 4, 0]]]
    assert packed["response_mask"].tolist() == [[[True, True, False], [True, True, True]]]
    assert (packed["policy_token_log_probs"]*packed["response_mask"]).sum().item() == pytest.approx(-6.9)


@pytest.mark.parametrize("change", [
    {"response_token_ids": []}, {"policy_token_log_probs": [-1.]},
    {"behavior_token_log_probs": [float("nan"), -1.]}, {"policy_token_log_probs": [0.1, -1.]},
    {"finish_reason": "length"}, {"response_token_ids": [0, 4]},
])
def test_invalid_alignment_probabilities_or_eos_are_rejected(change):
    with pytest.raises(ValueError):
        replace(sample(), **change)


def test_eos_only_answer_can_be_recorded_without_filtering():
    row = replace(sample(), response_token_ids=[0], policy_token_log_probs=[-0.1],
        behavior_token_log_probs=[-0.1], text="", raw_text="<eos>")
    assert row.finish_reason == "eos" and len(row.response_token_ids) == 1


@pytest.mark.parametrize("group", [[sample()], [sample(), sample()], [sample(), replace(sample(1), case_id="other")]])
def test_missing_duplicate_or_mismatched_group_members_are_rejected(group):
    with pytest.raises(ValueError):
        pack_rollout_groups([group], 0)
