import copy
import importlib.util
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eos_apply',ROOT/'hotfixes/grpo-native-eos-v1/apply.py')
hot=importlib.util.module_from_spec(spec); spec.loader.exec_module(hot)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(data,dict): hot.atomic(path,data)
    else: path.write_bytes(data)


def installation(tmp_path):
    root=tmp_path/'v2'; bundle=tmp_path/'bundle'; output=root/'artifacts/test-formal'
    write(root/'src/apt_v2/grpo.py',b'old validator'); write(output/'source/src/apt_v2/grpo.py',b'old validator')
    cfg={'run_id':'test-formal','adapter_sha256':'initial'}
    write(root/'configs/formal.json',cfg); write(output/'config.json',cfg)
    write(output/'input_binding.json',{'files':{'config.json':hot.sha(output/'config.json')}})
    initial={'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':'initial'}
    latest={'checkpoint':'checkpoints/group-0000','version':1,'updates':1,'adapter_sha256':'updated'}
    write(output/'groups/group-0000.json',{'group':0})
    tr={'index':0,'before':initial,'after':latest,'updated':True,'config_sha256':hot.sha(output/'config.json'),'group_sha256':hot.sha(output/'groups/group-0000.json')}
    write(output/'transitions/group-0000.json',tr)
    folder=output/latest['checkpoint']; write(folder/'state.pt',b'original Adam and RNG')
    write(folder/'adapter.bin',b'updated adapter'); write(folder/'complete.json',{'files':{'adapter.bin':hot.sha(folder/'adapter.bin')},'actual_updates':1,'group_cursor':1,'state_sha256':hot.sha(folder/'state.pt')})
    write(output/'runner_state.json',{'policy':latest,'next_group':1,'elapsed_seconds':5865.})
    write(output/'run_status.json',{'elapsed_seconds':5865.,'actual_updates':1})
    write(output/'evaluations/baseline.json',{'episodes':'preserved'})
    release={'files':{n:hot.sha(root/n) for n in ['src/apt_v2/grpo.py','configs/formal.json']}}
    run={'source_sha256':{'src/apt_v2/grpo.py':hot.sha(root/'src/apt_v2/grpo.py')}}
    write(root/'grpo_release_manifest.json',release); write(output/'run_manifest.json',run)
    write(bundle/'baseline_release.json',release); write(bundle/'baseline_run_manifest.json',run)
    write(bundle/'payload/src/apt_v2/grpo.py',b'new validator')
    m={'patch_id':hot.PATCH_ID,'run_id':'test-formal','committed_groups':1,'latest_policy':latest,
       'patches':{'src/apt_v2/grpo.py':{'before':hot.sha(root/'src/apt_v2/grpo.py'),'after':hot.sha(bundle/'payload/src/apt_v2/grpo.py')}},
       'run_bindings':{n:hot.sha(output/n) for n in ['config.json','input_binding.json','run_status.json','runner_state.json']},
       'bundle_files':{n:hot.sha(bundle/n) for n in ['baseline_release.json','baseline_run_manifest.json','payload/src/apt_v2/grpo.py']}}
    write(bundle/'manifest.json',m)
    return root,bundle,output


def test_patch_preserves_optimizer_evaluation_clock_and_original_source(tmp_path):
    root,bundle,output=installation(tmp_path); before=hot.tree(output)
    r=hot.apply(root,bundle)
    assert r['cloud_optimizer_hash_verified'] and r['elapsed_seconds_preserved']==5865.
    assert hot.tree(output)==before
    assert (output/'source/src/apt_v2/grpo.py').read_bytes()==b'old validator'
    assert (output/'source_revisions'/hot.PATCH_ID/'source/src/apt_v2/grpo.py').read_bytes()==b'new validator'
    assert hot.read(output/'run_manifest.json')['source_sha256']['src/apt_v2/grpo.py']==hot.sha(root/'src/apt_v2/grpo.py')
    assert hot.apply(root,bundle)==r


@pytest.mark.parametrize('target',['source','optimizer','config'])
def test_patch_rejects_changed_source_optimizer_or_run_identity(tmp_path,target):
    root,bundle,output=installation(tmp_path)
    path={'source':root/'src/apt_v2/grpo.py','optimizer':output/'checkpoints/group-0000/state.pt','config':output/'config.json'}[target]
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError): hot.apply(root,bundle)
    assert 'source_amendments' not in hot.read(output/'run_manifest.json')


def test_patch_recovers_interrupted_manifest_commit(tmp_path,monkeypatch):
    root,bundle,output=installation(tmp_path); original=hot.atomic
    def stop(path,value):
        if path==output/'run_manifest.json' and 'source_amendments' in value:
            raise RuntimeError('simulated installation interruption')
        return original(path,value)
    monkeypatch.setattr(hot,'atomic',stop)
    with pytest.raises(RuntimeError): hot.apply(root,bundle)
    monkeypatch.setattr(hot,'atomic',original)
    r=hot.apply(root,bundle)
    assert r['installed'] and hot.sha(root/'src/apt_v2/grpo.py')==r['effective_run_manifest']['source_sha256']['src/apt_v2/grpo.py']
