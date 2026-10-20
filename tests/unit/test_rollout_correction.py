import math

import pytest
import torch

from agentic_post_training.algorithms.losses import grpo_clipped_loss
from agentic_post_training.algorithms.rollout_correction import token_tis_weights, corrected_grpo_policy_loss


def fixture():
    old = torch.tensor([[[-1., -2.], [-3., -4.]]], dtype=torch.float64)
    new = old.clone().requires_grad_()
    return new, old, old.clone(), torch.tensor([[1., -1.]], dtype=torch.float64), torch.ones_like(old, dtype=torch.bool)


def test_equal_sampler_reduces_to_original_grpo_policy_objective():
    new, old, q, adv, mask = fixture()
    new = (new + .1)
    loss, d = corrected_grpo_policy_loss(new, old, q, adv, mask)
    expected = grpo_clipped_loss(new, old, adv, mask).loss
    torch.testing.assert_close(loss, expected)
    assert d['min_weight'] == d['max_weight'] == 1 and not d['weights_require_grad']


def test_weighted_positive_negative_advantages_and_policy_clipping():
    new, old, q, adv, mask = fixture()
    q = old - math.log(4)  # p_old/q=4, capped to 2 (separate from PPO clip).
    new = (old + math.log(1.5)).requires_grad_()
    loss, d = corrected_grpo_policy_loss(new, old, q, adv, mask)
    # Positive A uses clipped 1.2, negative A uses unclipped 1.5.
    assert loss.item() == pytest.approx(-2 * (1.2 - 1.5) / 2)
    loss.backward()
    torch.testing.assert_close(new.grad[0, 0], torch.zeros(2, dtype=new.dtype))
    torch.testing.assert_close(new.grad[0, 1], torch.full((2,), .75, dtype=new.dtype))
    assert d['cap_fraction'] == 1


def test_conditional_one_token_importance_identity_without_cap_activation():
    # Enumerate the entire two-action distribution at ONE fixed history.
    p, q = torch.tensor([.4, .6], dtype=torch.float64), torch.tensor([.5, .5], dtype=torch.float64)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    weights, _ = token_tis_weights(p.log().reshape(1,2,1), q.log().reshape(1,2,1), mask)
    f = torch.tensor([2., -1.], dtype=torch.float64)
    torch.testing.assert_close((q * weights.flatten() * f).sum(), (p * f).sum())


def test_padding_and_excluded_samples_do_not_affect_loss_or_gradient():
    new, old, q, adv, mask = fixture()
    mask[0, 0, 1] = False; mask[0, 1] = False
    old[~mask] = float('nan'); q[~mask] = float('inf'); adv[0,1] = float('nan')
    new = old.clone().requires_grad_()
    loss, _ = corrected_grpo_policy_loss(new, old, q, adv, mask)
    assert loss.item() == -1
    loss.backward()
    assert new.grad[mask].item() == -1 and (new.grad[~mask] == 0).all()


def test_zero_advantages_give_zero_loss_and_zero_gradients():
    new, old, q, adv, mask = fixture()
    loss, _ = corrected_grpo_policy_loss(new, old, q-.02, adv*0, mask)
    loss.backward()
    assert loss.item() == 0 and (new.grad == 0).all()


def test_autograd_matches_finite_difference_with_fixed_nonunit_weights():
    new, old, q, adv, mask = fixture(); q = q-.03
    def fn(x): return corrected_grpo_policy_loss(x, old, q, adv, mask)[0]
    assert torch.autograd.gradcheck(fn, (new,), eps=1e-6, atol=1e-5)


def test_extreme_sampler_delta_is_capped_before_exp():
    old = torch.tensor([[[-.1]]]); q = torch.tensor([[[-10000.]]]); mask = torch.ones_like(old, dtype=torch.bool)
    w, d = token_tis_weights(old, q, mask)
    assert w.item() == pytest.approx(2) and d['cap_fraction'] == 1


def test_downward_policy_clip_treats_positive_and_negative_advantages_differently():
    new, old, q, adv, mask = fixture()
    new = (old + math.log(.5)).requires_grad_()
    loss, _ = corrected_grpo_policy_loss(new, old, q, adv, mask)
    assert loss.item() == pytest.approx(-(.5-.8)/2)
    loss.backward()
    torch.testing.assert_close(new.grad[0,0], torch.full((2,), -.125, dtype=new.dtype))
    torch.testing.assert_close(new.grad[0,1], torch.zeros(2, dtype=new.dtype))


@pytest.mark.parametrize('mutation', ['old_graph','q_graph','adv_graph','empty_mask','float_mask','shape','nan','positive','cap','epsilon'])
def test_invalid_inputs_rejected(mutation):
    new, old, q, adv, mask = fixture(); kw = {}
    if mutation == 'old_graph': old.requires_grad_()
    elif mutation == 'q_graph': q.requires_grad_()
    elif mutation == 'adv_graph': adv.requires_grad_()
    elif mutation == 'empty_mask': mask[:] = False
    elif mutation == 'float_mask': mask = mask.float()
    elif mutation == 'shape': q = q[:,:,:1]
    elif mutation == 'nan': q[0,0,0] = float('nan')
    elif mutation == 'positive': old[0,0,0] = 1
    elif mutation == 'cap': kw['upper_cap'] = float('inf')
    elif mutation == 'epsilon': kw['clip_epsilon'] = 0
    with pytest.raises(ValueError): corrected_grpo_policy_loss(new, old, q, adv, mask, **kw)
