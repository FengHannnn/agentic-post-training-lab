import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.sft_test import holdout_tasks,holdout_summary,acceptance
from apt_v2.sft import build_dataset
from apt_v2.runtime import ScriptedPolicy,Generation,run_episode
from apt_v2.sft_diagnostics import summarize_episodes

@pytest.fixture(scope='module')
def setup_data():
    cfg=json.loads((ROOT/'configs/sft-independent-test-v1.json').read_text())
    return cfg,build_dataset()[1]

def test_holdout_selection_rejects_training_and_development_mixing(setup_data):
    cfg,examples=setup_data;tasks=holdout_tasks(cfg,examples)
    assert len(tasks)==4 and all(t.split=='test' for t in tasks)
    assert {t.family_id for t in tasks}.isdisjoint({e['family_id'] for e in examples})
    for change in ({'training_enabled':True},{'eval':{'splits':{'development':4}}}):
        bad={**cfg,**change}
        with pytest.raises(ValueError):holdout_tasks(bad,examples)
    leaked=copy.deepcopy(examples);leaked[0]['split']='test'
    with pytest.raises(ValueError,match='provenance'):holdout_tasks(cfg,leaked)

class CharTokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True):
        return [ord(c)+1 for c in ''.join(m['role']+':'+m['content']+'\n' for m in messages)+'assistant:']
    def decode(self,ids,**kwargs):return ''.join(chr(i-1) for i in ids if i)

def expert_episodes(tasks):
    tok=CharTokenizer();episodes=[]
    for task in tasks:
        expert=ScriptedPolicy(task)
        def policy(messages):
            text=expert(messages).text;ids=[ord(c)+1 for c in text]+[0]
            return Generation(text,'greedy',tok.apply_chat_template(messages),ids,[-.2]*len(ids),None,'stop')
        episodes.append(run_episode(task,policy))
    return episodes

def test_holdout_summary_requires_complete_distinct_tasks(setup_data):
    cfg,examples=setup_data;tasks=holdout_tasks(cfg,examples);episodes=expert_episodes(tasks)
    summary=holdout_summary(episodes,[t.id for t in tasks])
    assert summary['successes']==4 and summary['recovered']==2 and summary['valid_sql']==8
    assert summary['valid_checks']==summary['valid_submits']==4
    assert acceptance(summary,cfg['acceptance'])['eligible_for_local_review']
    with pytest.raises(ValueError,match='distinct'):holdout_summary(episodes[:-1]+[episodes[0]],[t.id for t in tasks])
    assert not acceptance({**summary,'successes':3},cfg['acceptance'])['eligible_for_local_review']

def test_complete_holdout_audit_replays_saved_token_contexts(tmp_path,monkeypatch,setup_data):
    import test_v2_sft as worker
    import transformers
    cfg,examples=setup_data;tasks=holdout_tasks(cfg,examples);episodes=expert_episodes(tasks)
    tok=CharTokenizer();monkeypatch.setattr(worker,'verify_inputs',lambda output:cfg)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    (tmp_path/'model_identity.json').write_text('{"files_sha256":{}}')
    from dataclasses import asdict
    (tmp_path/'test_tasks.json').write_text(json.dumps([asdict(t) for t in tasks]))
    comparison={}
    from common import file_hash
    for branch in ('parent','weighted'):
        rows=[{**e,'model_branch':branch} for e in episodes]
        file=tmp_path/(branch+'-test-episodes.jsonl')
        file.write_text(''.join(json.dumps(e)+'\n' for e in rows))
        summary={**summarize_episodes(rows),'episodes_sha256':file_hash(file)}
        (tmp_path/(branch+'-test-summary.json')).write_text(json.dumps(summary))
        comparison[branch]={'test':summary}
    (tmp_path/'comparison_summary.json').write_text(json.dumps(comparison))
    worker.audit(tmp_path)
    assert json.loads((tmp_path/'audit.json').read_text())['episodes_replayed']==8
    result=json.loads((tmp_path/'independent_test_summary.json').read_text())
    assert result['acceptance']['eligible_for_local_review'] and result['optimizer_updates']==0
    assert result['promotion_status']=='awaiting_local_review'

def test_holdout_runner_preserves_evidence_without_weights(tmp_path,monkeypatch):
    import run_v2_sft_test as runner
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    candidate=tmp_path/'artifacts/candidate';(candidate/'final/adapter').mkdir(parents=True)
    for n in ('examples.jsonl','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
        (candidate/n).write_text('fixture')
    (tmp_path/'configs/sft-independent-test-v1.json').write_text(json.dumps({'run_id':'holdout','candidate_run':'candidate','max_wall_seconds':60}))
    (tmp_path/'sft_test_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/test_v2_sft.py').write_text(
        "import sys; from pathlib import Path; p=Path(sys.argv[sys.argv.index('--output')+1]); "
        "(p/'skip.safetensors').write_bytes(b'weights'); raise RuntimeError('intentional failure')")
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    import zipfile
    with zipfile.ZipFile(tmp_path/'holdout-review.zip') as z:
        assert 'holdout/prepare.log' in z.namelist()
        assert not any(n.endswith(('.pt','.safetensors')) for n in z.namelist())
    with pytest.raises(FileExistsError):runner.run()
