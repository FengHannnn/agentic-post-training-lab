import sys,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from apt_v2.reference_numerics import compare_reference_caches

def fixture():
    original={'baseline_adapter_sha256':'same','pair_ids':['p'],'scores':[{'input_id':'same','chosen':[-1.0]*100,'rejected':[-3.838167428970337]}]}
    current=copy.deepcopy(original);current['scores'][0]['rejected']=[-3.8380355834960938]
    return current,original

def test_real_reference_boundary_case_is_accepted_with_explicit_loss_bound():
    a,b=fixture();r=compare_reference_caches(a,b)
    assert r['passed'] and r['maximum_token_difference']==pytest.approx(.00013184547424316406)
    assert r['maximum_DPO_loss_difference_bound']<5e-5

def test_identity_alignment_and_nonfinite_are_rejected():
    for change in ['adapter','pair','input','length','nan','probability']:
        a,b=fixture()
        if change=='adapter':a['baseline_adapter_sha256']='other'
        elif change=='pair':a['pair_ids']=['other']
        elif change=='input':a['scores'][0]['input_id']='other'
        elif change=='length':a['scores'][0]['chosen'].append(-1.)
        elif change=='nan':a['scores'][0]['chosen']=[float('nan')]
        else:a['scores'][0]['chosen']=[.1]
        with pytest.raises(ValueError):compare_reference_caches(a,b)

def test_large_token_or_accumulated_margin_difference_is_rejected():
    a,b=fixture();a['scores'][0]['rejected']=[-3.83]
    with pytest.raises(ValueError):compare_reference_caches(a,b)
    a,b=fixture()
    for obj in [a,b]:obj['scores'][0]['chosen']=[-3.]*100;obj['scores'][0]['rejected']=[-3.]*100
    a['scores'][0]['chosen']=[-3.+1e-5]*100
    with pytest.raises(ValueError,match='sequence|margin|logit'):compare_reference_caches(a,b)

def test_original_absolute_tolerance_still_accepts_small_logp_values():
    a,b=fixture();a['scores'][0]['rejected']=[-.9537012577056885];b['scores'][0]['rejected']=[-.9536534547805786]
    assert compare_reference_caches(a,b)['passed']
