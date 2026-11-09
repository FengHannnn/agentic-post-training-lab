"""Installer transaction, immutable budget and cloud Adam hash regressions."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('probe_hotfix',ROOT/'hotfixes/grpo-probe-replay-v1/apply.py')
hot=importlib.util.module_from_spec(spec);spec.loader.exec_module(hot)


def fixture(tmp_path):
    root=tmp_path/'v2';bundle=root/'hotfixes/grpo-probe-replay-v1';bundle.mkdir(parents=True)
    sources={'scripts/grpo_vllm.py':b'old collector','scripts/grpo_worker.py':b'old verifier'}
    for name,data in sources.items():p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    old_release={'files':{n:hot.sha(root/n) for n in sources}}
    hot.atomic(root/hot.RELEASE,old_release);hot.atomic(bundle/'baseline_release.json',old_release)
    run_id='test-outcome-hotfix';output=root/'artifacts'/run_id;old_run={'source_sha256':dict(old_release['files'])}
    roots={'root':output,**{b:output/'branches'/b for b in ('control','lr10x')}}
    latest={}
    for label,runroot in roots.items():
        for n,data in sources.items():p=runroot/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        hot.atomic(runroot/'run_manifest.json',old_run);hot.atomic(bundle/'baseline_runs'/f'{label}.json',old_run)
        if label=='root':continue
        cfg={'run_id':label,'adapter_sha256':'W'};hot.atomic(runroot/'config.json',cfg)
        hot.atomic(runroot/'input_binding.json',{'files':{'config.json':hot.sha(runroot/'config.json')}})
        before={'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':'W'}
        folder=runroot/'checkpoints/group-0000';adapter=folder/'adapter/adapter_model.safetensors'
        adapter.parent.mkdir(parents=True);adapter.write_bytes(('weights-'+label).encode())
        (folder/'state.pt').write_bytes(('adam-'+label).encode())
        after={'checkpoint':'checkpoints/group-0000','version':1,'updates':1,'adapter_sha256':hot.sha(adapter)}
        hot.atomic(runroot/'groups/group-0000.json',{'native':'fixed'})
        hot.atomic(runroot/'transitions/group-0000.json',{'index':0,'before':before,'after':after,'updated':True,
            'config_sha256':hot.sha(runroot/'config.json'),'group_sha256':hot.sha(runroot/'groups/group-0000.json')})
        hot.atomic(folder/'complete.json',{'files':{'adapter/adapter_model.safetensors':hot.sha(adapter)},
            'actual_updates':1,'group_cursor':1,'config_sha256':hot.sha(runroot/'config.json'),'state_sha256':hot.sha(folder/'state.pt')})
        hot.atomic(runroot/'runner_state.json',{'policy':after,'next_group':1});latest[label]=after
    hot.atomic(output/'config.json',{'max_wall_seconds':21600,'groups_per_branch':48})
    hot.atomic(output/'run_status.json',{'actual_updates':{'control':1,'lr10x':1},'committed_groups':{'control':1,'lr10x':1},'elapsed_seconds':1000.})
    hot.atomic(output/'runner_state.json',{'policies':latest,'next_groups':{'control':1,'lr10x':1},'elapsed_seconds':1000.})
    patches={}
    for name,data in sources.items():
        p=bundle/'payload'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data.replace(b'old',b'new'))
        patches[name]={'before':old_release['files'][name],'after':hot.sha(p),'source_binding':True}
    m={'patch_id':hot.PATCH_ID,'run_id':run_id,'patches':patches,'latest_policies':latest,
       'actual_updates':{'control':1,'lr10x':1},'committed_groups':{'control':1,'lr10x':1},
       'run_bindings':{n:h for n,h in hot.protected_tree(output).items() if not n.endswith('.pt')}}
    m['bundle_files']={p.relative_to(bundle).as_posix():hot.sha(p) for p in bundle.rglob('*') if p.is_file()}
    hot.atomic(bundle/'manifest.json',m)
    return root,bundle,output,old_release,old_run


def test_cloud_install_preserves_every_training_and_budget_file_and_is_idempotent(tmp_path):
    root,bundle,output,old_release,old_run=fixture(tmp_path);before=hot.protected_tree(output)
    result=hot.apply(root,bundle)
    assert result['cloud_optimizer_hashes_verified'] and result['remaining_budget_seconds']==20600
    assert hot.protected_tree(output)==before and result['config_unchanged']
    assert hot.read(output/'source_revisions'/hot.PATCH_ID/'release_before.json')==old_release
    assert (output/'source/scripts/grpo_vllm.py').read_bytes()==b'old collector'
    assert (output/'source_revisions'/hot.PATCH_ID/'source/scripts/grpo_vllm.py').read_bytes()==b'new collector'
    assert hot.apply(root,bundle)==result


def test_local_review_does_not_claim_cloud_adam_verification(tmp_path):
    root,bundle,output,_,_=fixture(tmp_path)
    for b in ('control','lr10x'):(output/'branches'/b/'checkpoints/group-0000/state.pt').unlink()
    with pytest.raises(FileNotFoundError):hot.apply(root,bundle)
    assert not (output/'source_revisions'/hot.PATCH_ID/'applied.json').exists()
    result=hot.apply(root,bundle,local_review=True)
    assert result['local_review_only'] and not result['cloud_optimizer_hashes_verified']
    with pytest.raises(ValueError,match='cloud state'):hot.apply(root,bundle)


@pytest.mark.parametrize('target',['checkpoint','budget','adam'])
def test_install_rejects_changed_checkpoint_budget_or_adam_before_source_mutation(tmp_path,target):
    root,bundle,output,old_release,_=fixture(tmp_path)
    if target=='checkpoint':(output/'branches/control/checkpoints/group-0000/adapter/adapter_model.safetensors').write_bytes(b'changed')
    elif target=='budget':hot.atomic(output/'config.json',{'max_wall_seconds':30000,'groups_per_branch':48})
    else:(output/'branches/control/checkpoints/group-0000/state.pt').write_bytes(b'wrong optimizer')
    with pytest.raises((ValueError,FileNotFoundError)):hot.apply(root,bundle)
    assert hot.read(root/hot.RELEASE)==old_release and (root/'scripts/grpo_vllm.py').read_bytes()==b'old collector'


def test_interrupted_source_install_recovers_from_journal(tmp_path):
    root,bundle,output,old_release,old_run=fixture(tmp_path)
    revision=output/'source_revisions'/hot.PATCH_ID;hot.atomic(revision/'release_before.json',old_release)
    for runroot in (output,output/'branches/control',output/'branches/lr10x'):
        hot.atomic(runroot/'source_revisions'/hot.PATCH_ID/'run_manifest_before.json',old_run)
    (root/'scripts/grpo_vllm.py').write_bytes(b'new collector')
    before=hot.protected_tree(output);result=hot.apply(root,bundle)
    assert result['installed'] and hot.protected_tree(output)==before



def test_already_installed_hotfix_rejects_changed_bundle_identity(tmp_path):
    root,bundle,output,_,_=fixture(tmp_path);hot.apply(root,bundle)
    manifest=hot.read(bundle/'manifest.json');manifest['changed_bundle_note']='another revision';hot.atomic(bundle/'manifest.json',manifest)
    with pytest.raises(ValueError,match='bundle differs'):hot.apply(root,bundle)
