import copy
import importlib.util
from pathlib import Path
import sys

import pytest


def comparator():
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('grpo_comparison_test',root/'scripts/train_grpo_first.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.compare_evaluations


def row(identifier,passed):
    return {'case_id':identifier,'strict_pass':passed,'semantic_pass':passed,'finish_reason':'eos',
        'text':str(passed),'score':{'json_valid':passed,'schema_valid':passed,'route_correct':passed}}


def test_equal_total_score_does_not_hide_new_structure_failure():
    compare=comparator()
    before={'development':[row('a',True),row('b',False)]}
    after={'development':[row('a',False),row('b',True)]}
    result=compare(before,after)['development']
    assert result['before_strict']==result['after_strict']==1
    assert result['new_structure_failure_cases']==1 and len(result['changed_cases'])==2


def test_length_stop_is_not_counted_as_strict_completed_prediction_and_ids_must_align():
    compare=comparator();before={'development':[row('a',True)]};after=copy.deepcopy(before)
    after['development'][0]['finish_reason']='length'
    result=compare(before,after)['development']
    assert result['before_strict']==1 and result['after_strict']==0 and result['after_length_stops']==1
    after['development'][0]['case_id']='other'
    with pytest.raises(ValueError,match='evaluation identity differs'):compare(before,after)
