import json
import sys
import zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import atomic_json,file_hash
import run_v2_grpo_c4 as driver
CFG=json.loads((ROOT/'configs/grpo-c4-confirmation-v1.json').read_text())


@pytest.mark.parametrize('mode',['complete','timeout','failure'])
def test_driver_returns_review_and_preserves_shared_budget(tmp_path,monkeypatch,mode):
    import subprocess
    root=tmp_path/'v2';root.mkdir();config=root/'configs/grpo-c4-confirmation-v1.json';atomic_json(config,CFG)
    files={'configs/grpo-c4-confirmation-v1.json':file_hash(config)};atomic_json(root/'grpo_c4_release_manifest.json',{'files':files})
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'source_files',lambda:files)
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    calls=[]
    def logged(command,**kwargs):
        operation=command[3];output=Path(command[command.index('--output')+1]);calls.append(operation)
        if operation=='prepare':atomic_json(output/'input_binding.json',{'prepared':True})
        elif operation=='evaluate':
            if mode=='timeout':raise subprocess.TimeoutExpired(command,1)
            if mode=='failure':return 1
            label=command[command.index('--label')+1];atomic_json(output/'episodes'/label/'0000.json',{'complete':True})
        else:atomic_json(output/'c4_audit.json',{'passed':True,'acceptance':{'effect_confirmed':False}})
        return 0
    monkeypatch.setattr(driver,'run_logged',logged)
    if mode=='complete':archive=driver.run(config)
    else:
        with pytest.raises(RuntimeError):driver.run(config)
        archive=root/(CFG['run_id']+'-review.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;status=json.loads(z.read(CFG['run_id']+'/run_status.json'))
    assert status['new_training_updates']==0
    assert status['stop_reason']=={'complete':'completed','timeout':'timeout','failure':'failed'}[mode]
    if mode=='complete':
        assert status['acceptance']=={'effect_confirmed':False}
        old_elapsed=status['elapsed_seconds'];archive2=driver.run(config,resume=True)
        assert archive2.name.endswith('attempt-002.zip') and archive.exists()
        second=json.loads((root/'artifacts'/CFG['run_id']/'run_status.json').read_text())
        assert second['elapsed_seconds']>=old_elapsed and calls.count('prepare')==1
