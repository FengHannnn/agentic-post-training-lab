import math

import pytest
import torch

from agentic_post_training.trainers.dpo import anchored_objective, objective


def test_zero_anchor_preserves_pure_dpo_loss_and_gradient():
    scores = torch.tensor([-3.0, -5.0], requires_grad=True)
    reference = torch.tensor([-2.0, -4.0], requires_grad=True)
    loss, _, _ = anchored_objective(scores, reference, 0.1, 3)
    expected = objective(scores, reference, 0.1).loss
    assert torch.equal(loss, expected)
    assert torch.equal(torch.autograd.grad(loss, scores, retain_graph=True)[0],
                       torch.autograd.grad(expected, scores)[0])
    assert reference.grad is None


def test_anchor_supplies_gradient_on_shared_prefix_that_cancels_in_dpo():
    shared = torch.tensor(-1.0, requires_grad=True)
    scores = torch.stack((shared - 2.0, shared - 4.0))
    reference = torch.tensor([-3.0, -5.0], requires_grad=True)
    pure = objective(scores, reference, 0.1).loss
    assert torch.autograd.grad(pure, shared, retain_graph=True)[0].item() == 0
    loss, output, nll = anchored_objective(scores, reference, 0.1, 4, 1.0)
    assert nll.item() == 0.75
    assert torch.allclose(loss, output.loss + nll)
    loss.backward()
    # Descent increases the correct shared prefix log probability.
    assert shared.grad.item() == -0.25
    assert reference.grad is None


@pytest.mark.parametrize("weight", [-1, math.nan, math.inf])
def test_invalid_anchor_is_rejected(weight):
    with pytest.raises(ValueError, match="chosen_nll_weight"):
        anchored_objective(torch.zeros(2), torch.zeros(2), 0.1, 4, weight)


def test_empty_chosen_response_is_rejected():
    with pytest.raises(ValueError, match="chosen_tokens"):
        anchored_objective(torch.zeros(2), torch.zeros(2), 0.1, 0, 1.0)
