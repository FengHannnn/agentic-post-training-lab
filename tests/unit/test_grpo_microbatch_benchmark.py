import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def runner():
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('microbatch_benchmark_test',root/'scripts/benchmark_grpo_microbatch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_gradient_comparison_checks_absolute_and_global_relative_error():
    module=runner()
    expected={'a':torch.tensor([3.,4.]),'b':torch.tensor([0.])}
    actual={'a':torch.tensor([3.,4.]),'b':torch.tensor([1.])}
    assert module.gradient_difference(actual,expected)=={'gradient_max_abs_error':1.,'gradient_relative_l2_error':.2}
    with pytest.raises(ValueError,match='names'):module.gradient_difference({},expected)
    with pytest.raises(ValueError,match='nonfinite'):module.gradient_difference({'a':torch.tensor([float('nan')])},{'a':torch.tensor([1.])})


@pytest.mark.parametrize('name',['logp_max_abs_error','loss_abs_error','gradient_max_abs_error',
    'gradient_relative_l2_error','max_adapter_abs_error','max_optimizer_state_abs_error'])
def test_every_numeric_gate_is_required_and_nonfinite_values_fail(name):
    module=runner();plan=module.read(module.PLAN)
    metrics={key:0. for key in ('logp_max_abs_error','loss_abs_error','gradient_max_abs_error',
        'gradient_relative_l2_error','max_adapter_abs_error','max_optimizer_state_abs_error')}
    assert module.numerical_gate(metrics,plan)
    metrics[name]=1.;assert not module.numerical_gate(metrics,plan)
    metrics[name]=float('nan');assert not module.numerical_gate(metrics,plan)


def test_fastest_passing_parallel_mode_or_serial_fallback():
    module=runner()
    checks={str(n):{'passed':True} for n in (1,4,8)}
    times=[{'microbatch_size':n,'seconds':t} for n,t in ((1,4.),(4,2.),(8,1.)) for _ in range(3)]
    assert module.recommend(checks,times)['recommended_microbatch_size']==8
    checks['8']['passed']=False
    assert module.recommend(checks,times)['recommended_microbatch_size']==4
    checks['4']['passed']=False
    assert module.recommend(checks,times)['recommended_microbatch_size']==1
    checks={str(n):{'passed':True} for n in (1,4,8)}
    slower=copy.deepcopy(times)
    for row in slower:
        if row['microbatch_size']!=1:row['seconds']=5.
    assert module.recommend(checks,slower)['recommended_microbatch_size']==1


def test_benchmark_contract_never_publishes_probe_updates():
    module=runner();plan=module.read(module.PLAN)
    assert plan['effective_minibatch_groups']==4 and plan['effective_completions']==32
    assert plan['microbatch_sizes']==[1,4,8] and plan['timing_repeats']==3
    assert plan['published_candidate_updates']==0
    assert plan['source_training_summary_sha256']=='1816932fd3df9a5aa6bb3e18c402cb4775378d50c78c2311c1910950194213ad'
