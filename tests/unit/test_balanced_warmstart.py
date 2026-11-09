import copy
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from balanced_warmstart_common import select_demonstrations,demonstration_schedule,FIXED
from agentic_post_training.data.preferences import load_preferences

@pytest.fixture
def data():
    train=load_preferences(ROOT/'data/preferences/dpo_v2/train.jsonl')
    plan=json.loads((ROOT/'configs/experiments/grpo_balanced_warmstart.json').read_text())
    by_id={r.id:r for r in train}
    return train,[by_id[key] for key in plan['case_ids']],plan

def test_only_precommitted_twelve_training_retention_examples_are_added(data):
    train,original,plan=data
    combined,added=select_demonstrations(train,original)
    assert [r.id for r in combined]==plan['demonstration_ids']
    assert [r.id for r in added]==plan['retention_case_ids']
    assert combined[:24]==original
    assert len(combined)==36 and len(added)==12
    assert Counter((r.family_id,r.provenance['wording']) for r in added)==Counter(
        (f,w) for f in {r.family_id for r in original} for w in ('symbolic','natural','sequence'))
    assert all(r.category=='left_grouped' and json.loads(r.chosen)['arguments']['expression'].startswith('(') for r in added)
    dev=load_preferences(ROOT/'data/preferences/dpo_v2/development.jsonl')
    assert not ({r.family_id for r in combined}&{r.family_id for r in dev})
    assert not ({(r.system,r.prompt) for r in combined}&{(r.system,r.prompt) for r in dev})

@pytest.mark.parametrize('mutation',['duplicate_train','missing_retention','incorrect_answer','missing_leading_parenthesis','outside_original'])
def test_invalid_demonstration_compositions_rejected(data,mutation):
    train,original,plan=data
    target=next(r for r in train if r.id==plan['retention_case_ids'][0])
    if mutation=='duplicate_train':train=train+[train[0]]
    elif mutation=='missing_retention':train=[r for r in train if r.id!=target.id]
    elif mutation=='outside_original':train=[r for r in train if r.id!=original[0].id]
    else:
        replacement=replace(target,chosen=target.chosen.replace(target.expected_expression,
            '(1+2)*3' if mutation=='incorrect_answer' else target.expected_expression[1:]))
        train=[replacement if r.id==target.id else r for r in train]
    with pytest.raises(ValueError):select_demonstrations(train,original)

def test_equal_number_of_steps_and_presentations_with_two_complete_epochs():
    schedule=demonstration_schedule(36,2,202610300)
    assert len(schedule)==18
    assert all(len(indices)==4 for epoch,indices in schedule)
    assert Counter(i for epoch,indices in schedule for i in indices)==Counter({i:2 for i in range(36)})
    assert [epoch for epoch,indices in schedule]==[1]*9+[2]*9
    assert schedule==demonstration_schedule(36,2,202610300)
    assert schedule!=demonstration_schedule(36,2,202610301)

@pytest.mark.parametrize('count,epochs',[(24,3),(35,2),(36,3)])
def test_uncontrolled_training_dose_is_rejected(count,epochs):
    with pytest.raises(ValueError):demonstration_schedule(count,epochs,202610300)

def test_hyperparameters_match_previous_control_except_composition_and_epochs():
    previous=json.loads((ROOT/'configs/experiments/grpo_demonstration_warmstart.json').read_text())
    current=json.loads((ROOT/'configs/experiments/grpo_balanced_warmstart.json').read_text())
    assert {key:current[key] for key in previous if key not in ('contract_version','sft_epochs')}=={
        key:value for key,value in previous.items() if key not in ('contract_version','sft_epochs')}
    assert all(current[key]==value for key,value in FIXED.items())
    assert current['sft_epochs']==2 and previous['sft_epochs']==3

@pytest.mark.parametrize('failed_stage',[0,1,2])
def test_new_driver_preserves_failure_and_never_launches_later_stages(tmp_path,monkeypatch,failed_stage):
    import zipfile
    import run_balanced_warmstart as driver
    from agentic_post_training.trainers.policy_iteration import file_hash
    stub=tmp_path/'stub';stub.write_text('data')
    manifest=tmp_path/'bundle.json'
    manifest.write_text(json.dumps({'files':{'stub':file_hash(stub)},'required_existing_files':{'stub':file_hash(stub)}}))
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'OUTPUT',tmp_path/'new-run')
    monkeypatch.setattr(driver,'ARCHIVE',tmp_path/'new-review.zip')
    monkeypatch.setattr(driver,'BUNDLE',manifest)
    monkeypatch.setattr(driver,'preflight',lambda:{'passed':True})
    calls=[]
    def spawn(*args,**kwargs):
        index=len(calls);calls.append(args[0])
        return SimpleNamespace(wait=lambda **kw:1 if index==failed_stage else 0)
    monkeypatch.setattr(driver.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError):driver.dispatch(2.18)
    assert len(calls)==failed_stage+1
    assert all('train_balanced_warmstart.py' in call[1] for call in calls)
    with zipfile.ZipFile(driver.ARCHIVE) as z:
        result=json.loads(z.read('new-run/cost_and_run.json'))
        assert result['error'] and not result['automatic_retry']
    with pytest.raises(FileExistsError):driver.dispatch(2.18)
