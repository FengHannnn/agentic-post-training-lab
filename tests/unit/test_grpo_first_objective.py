import math

import pytest
import torch

from agentic_post_training.algorithms.rollout_correction import corrected_grpo_policy_loss
from agentic_post_training.trainers.grpo import completion_objective


def test_beta_zero_matches_existing_policy_loss():
    old=torch.tensor([-1.,-2.]);new=(old+.1).requires_grad_();q=old-.02;ref=old.clone()
    actual,metrics=completion_objective(new,old,q,ref,1.,beta=0)
    expected,_=corrected_grpo_policy_loss(new.reshape(1,1,2),old.reshape(1,1,2),q.reshape(1,1,2),
        torch.ones(1,1),torch.ones(1,1,2,dtype=torch.bool))
    torch.testing.assert_close(actual,expected)
    assert metrics['sampled_k3']>0


def test_sampled_penalty_zero_at_reference_and_gradient_toward_reference():
    old=torch.tensor([-1.,-2.],dtype=torch.float64)
    new=old.clone().requires_grad_();q=old.clone()
    loss,m=completion_objective(new,old,q,old,0.,beta=.02)
    assert loss.item()==0 and m['sampled_k3']==0
    loss.backward();assert (new.grad==0).all()
    new=old.clone().requires_grad_();ref=old+torch.tensor([-.2,.2],dtype=old.dtype)
    loss,m=completion_objective(new,old,q,ref,0.,beta=.02)
    loss.backward()
    expected=.02*(1-(ref-new.detach()).exp())/2
    torch.testing.assert_close(new.grad,expected)
    assert new.grad[0]>0 and new.grad[1]<0 and m['sampled_k3']>0


def test_combined_objective_passes_finite_difference_with_nonunit_tis():
    old=torch.tensor([-1.,-2.],dtype=torch.float64);q=old-.04;ref=old-.03
    new=(old+.05).requires_grad_()
    assert torch.autograd.gradcheck(lambda x:completion_objective(x,old,q,ref,-.5)[0],(new,),eps=1e-6,atol=1e-5)


@pytest.mark.parametrize('mutation',['reference_graph','reference_nan','reference_positive','reference_shape','negative_beta','adv_nan','old_graph'])
def test_invalid_objective_inputs_rejected(mutation):
    old=torch.tensor([-1.,-2.]);new=old.clone().requires_grad_();q=old.clone();ref=old.clone();adv=1.;beta=.02
    if mutation=='reference_graph':ref.requires_grad_()
    elif mutation=='reference_nan':ref[0]=float('nan')
    elif mutation=='reference_positive':ref[0]=1
    elif mutation=='reference_shape':ref=ref[:1]
    elif mutation=='negative_beta':beta=-.1
    elif mutation=='adv_nan':adv=float('nan')
    else:old.requires_grad_()
    with pytest.raises(ValueError):completion_objective(new,old,q,ref,adv,beta=beta)


def test_committed_schedule_is_one_pass_over_all_62_mixed_groups():
    from pathlib import Path
    import json
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/grpo_first_update.json').read_text(encoding='utf-8'))
    reward=json.loads((root/'artifacts/trajectories/phase5-signal-20261021-01/reward_audit.json').read_text(encoding='utf-8'))
    flat=[x for batch in cfg['group_batches'] for x in batch]
    assert len(flat)==len(set(flat))==62
    assert set(flat)=={r['group_id'] for r in reward['groups'] if not r['zero_variance'] and r['eligible']}
    assert len(cfg['group_batches'])==16 and max(map(len,cfg['group_batches']))==4
    assert cfg['rollout_policy_iterations']==1 and not cfg['auto_promote_baseline']
