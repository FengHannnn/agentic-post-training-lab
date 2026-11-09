import sys,json,zipfile,hashlib
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

def setup(tmp_path,monkeypatch):
    import run_v2_dpo_uniform_audit_repair as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    out=tmp_path/'artifacts'/runner.RUN;out.mkdir(parents=True);(out/'source').mkdir()
    for name,value in [('run_manifest.json',{'source_sha256':{}}),('training_summary.json',{'optimizer_updates':258}),('run_status.json',{'optimizer_updates':258,'stages':[{'stage':'train','exit_code':0},{'stage':'complete','exit_code':0},{'stage':'recovery','exit_code':0},{'stage':'audit','exit_code':1}]})]:
        (out/name).write_text(json.dumps(value))
    (tmp_path/(runner.RUN+'-review.zip')).write_bytes(b'original archive')
    (tmp_path/runner.MANIFEST).write_text(json.dumps({'files':{},'reviewed_artifacts_sha256':{},'original_source_sha256':{},'original_review_archive_sha256':hashlib.sha256(b'original archive').hexdigest()}))
    return runner,out

def test_success_preserves_original_and_only_invokes_saved_artifact_auditor(tmp_path,monkeypatch):
    runner,out=setup(tmp_path,monkeypatch);before=(out/'run_status.json').read_bytes();calls=[]
    def logged(command,**kwargs):
        calls.append(command);assert command[2].endswith('audit_v2_dpo_uniform_repair.py') and kwargs['timeout']==1800 and 'env' in kwargs
        (out/'audit-repair.log').write_text('audit completed')
        (out/'audit.json').write_text('{"passed":true}');(out/'reference_numerics_audit.json').write_text('{"passed":true}')
        return 0
    monkeypatch.setattr(runner,'run_logged',logged);runner.run()
    assert len(calls)==1 and (out/'run_status.json').read_bytes()==before
    status=json.loads((out/'audit_repair_status.json').read_text());assert status['error'] is None and status['optimizer_updates_this_repair']==0 and status['original_artifacts_preserved']
    with zipfile.ZipFile(tmp_path/(runner.RUN+'-audit-repaired-review.zip')) as z:assert runner.RUN+'/audit_repair_status.json' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()

def test_failure_packages_log_without_changing_original_run(tmp_path,monkeypatch):
    runner,out=setup(tmp_path,monkeypatch);before=(out/'run_status.json').read_bytes()
    def logged(*args,**kwargs):(out/'audit-repair.log').write_text('replay failed');return 1
    monkeypatch.setattr(runner,'run_logged',logged)
    with pytest.raises(RuntimeError,match='audit failed'):runner.run()
    assert (out/'run_status.json').read_bytes()==before
    status=json.loads((out/'audit_repair_status.json').read_text());assert status['error'] and status['original_artifacts_preserved']
    with zipfile.ZipFile(tmp_path/(runner.RUN+'-audit-repaired-review.zip')) as z:assert runner.RUN+'/audit-repair.log' in z.namelist()

def test_changed_reviewed_artifact_is_rejected_before_audit(tmp_path,monkeypatch):
    runner,out=setup(tmp_path,monkeypatch)
    p=out/'proof.json';p.write_text('original');release=json.loads((tmp_path/runner.MANIFEST).read_text());release['reviewed_artifacts_sha256']={'proof.json':hashlib.sha256(b'original').hexdigest()};(tmp_path/runner.MANIFEST).write_text(json.dumps(release));p.write_text('changed')
    monkeypatch.setattr(runner,'run_logged',lambda *a,**k:pytest.fail('auditor must not run'))
    with pytest.raises(ValueError):runner.run()
