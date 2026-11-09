"""Install a hash-bound probability-probe repair without changing training transactions."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

PATCH_ID='grpo-probe-same-forward-v2'
RELEASE='grpo_curriculum_release_manifest.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def under(root,name):
    p=(Path(root)/name).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('path outside hotfix root')
    return p

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.writing')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def verify(root,files):
    for name,h in files.items():
        if sha(under(root,name))!=h:raise ValueError('hash-bound file differs: '+name)

def protected_tree(output):
    result={}
    for p in sorted(output.rglob('*')):
        rel=p.relative_to(output)
        if (p.is_file() and not {'source','source_revisions'}&set(rel.parts)
                and p.name!='run_manifest.json' and '.writing' not in p.name):
            result[rel.as_posix()]=sha(p)
    return result


def validate_branch(child,expected,groups,local_review):
    cfg=read(child/'config.json');verify(child,read(child/'input_binding.json')['files'])
    state={'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':cfg['adapter_sha256']}
    if len(list((child/'transitions').glob('group-*.json')))!=groups:raise ValueError('committed group count differs')
    for i in range(groups):
        name=f'group-{i:04}.json';tr=read(child/'transitions'/name)
        if (tr['index']!=i or tr['before']!=state or tr['config_sha256']!=sha(child/'config.json')
                or tr['group_sha256']!=sha(child/'groups'/name)):
            raise ValueError('committed transition identity differs')
        if tr['updated']:
            folder=under(child,tr['after']['checkpoint']);complete=read(folder/'complete.json')
            verify(folder,complete['files'])
            if (tr['after']['updates']!=state['updates']+1 or tr['after']['version']!=state['version']+1
                    or complete['actual_updates']!=tr['after']['updates'] or complete['group_cursor']!=i+1
                    or complete['config_sha256']!=sha(child/'config.json')
                    or sha(folder/'adapter/adapter_model.safetensors')!=tr['after']['adapter_sha256']):
                raise ValueError('committed checkpoint identity differs')
        elif tr['after']!=state:raise ValueError('skipped update changed policy')
        state=tr['after']
    runner=read(child/'runner_state.json')
    if state!=expected or runner['policy']!=state or runner['next_group']!=groups:
        raise ValueError('latest branch policy/cursor differs')
    if not local_review and state['updates']:
        folder=under(child,state['checkpoint'])
        if sha(folder/'state.pt')!=read(folder/'complete.json')['state_sha256']:
            raise ValueError('cloud Adam/RNG state hash differs')
    return state


def source_root(runroot,manifest):
    amendments=manifest.get('source_amendments',[])
    return under(runroot,amendments[-1]['source_location'] if amendments else 'source')


def apply(root,bundle,local_review=False):
    root=Path(root).resolve();bundle=Path(bundle).resolve();m=read(bundle/'manifest.json')
    if m['patch_id']!=PATCH_ID:raise ValueError('hotfix identity differs')
    verify(bundle,m['bundle_files'])
    output=under(root,'artifacts/'+m['run_id']);rev=output/'source_revisions'/PATCH_ID;receipt=rev/'applied.json'
    roots={'root':output,**{b:output/'branches'/b for b in ('control','lr10x')}}
    if receipt.exists():
        result=read(receipt)
        if result['effective_release']['source_amendments'][-1]['manifest_sha256']!=sha(bundle/'manifest.json'):
            raise ValueError('already-applied hotfix bundle differs')
        if result['local_review_only'] and not local_review:raise ValueError('local-review receipt requires cloud state verification')
        verify(root,result['effective_release']['files'])
        if read(root/RELEASE)!=result['effective_release']:raise ValueError('installed release manifest differs')
        for b,runroot in roots.items():
            if read(runroot/'run_manifest.json')!=result['effective_run_manifests'][b]:raise ValueError('installed run manifest differs')
        print('Hotfix already installed and verified. Resume the existing job.',flush=True);return result
    baseline=read(bundle/'baseline_release.json')
    expected_runs={b:read(bundle/'baseline_runs'/f'{b}.json') for b in roots}
    old_release=read(rev/'release_before.json') if (rev/'release_before.json').exists() else read(root/RELEASE)
    if old_release!=baseline:raise ValueError('published baseline release differs')
    old_runs={}
    for b,runroot in roots.items():
        backup=runroot/'source_revisions'/PATCH_ID/'run_manifest_before.json'
        old_runs[b]=read(backup if backup.exists() else runroot/'run_manifest.json')
        if old_runs[b]!=expected_runs[b]:raise ValueError('published baseline run differs: '+b)
        verify(source_root(runroot,old_runs[b]),old_runs[b]['source_sha256'])
    patches=m['patches'];new_release=json.loads(json.dumps(old_release));new_runs=json.loads(json.dumps(old_runs))
    amendment={'patch_id':PATCH_ID,'manifest_sha256':sha(bundle/'manifest.json'),
        'scope':'probability verification only; same native q, training objective, config, checkpoints and cumulative budget',
        'patches':patches,'source_location':'source_revisions/'+PATCH_ID+'/source',
        'prefill_logp_diagnostic_threshold':.25,'same_forward_logp_gate':1e-5,
        'verification_method':'same_forward_sampler_v2','independent_fp64_check':'diagnostic only'}
    for name,p in patches.items():
        if old_release['files'].get(name)!=p['before'] or sha(bundle/'payload'/name)!=p['after']:
            raise ValueError('patch baseline/payload differs: '+name)
        new_release['files'][name]=p['after']
        if p['source_binding']:
            for b in roots:
                if old_runs[b]['source_sha256'].get(name)!=p['before']:
                    raise ValueError('run source baseline differs: '+b+':'+name)
                new_runs[b]['source_sha256'][name]=p['after']
    new_release['source_amendments']=[*old_release.get('source_amendments',[]),amendment]
    for b in roots:new_runs[b]['source_amendments']=[*old_runs[b].get('source_amendments',[]),amendment]
    if read(root/RELEASE) not in (old_release,new_release):raise ValueError('active release differs from journal')
    for b,runroot in roots.items():
        if read(runroot/'run_manifest.json') not in (old_runs[b],new_runs[b]):raise ValueError('active run differs from journal')
    for name,h in old_release['files'].items():
        allowed={h,patches[name]['after']} if name in patches else {h}
        if sha(root/name) not in allowed:raise ValueError('active source/input differs: '+name)
    for name,p in patches.items():
        target=under(root,name)
        if p['before'] is None and target.exists() and sha(target)!=p['after']:
            raise ValueError('new patch path already contains another file: '+name)
    verify(output,m['run_bindings'])
    cfg=read(output/'config.json');status=read(output/'run_status.json');runner=read(output/'runner_state.json')
    if (cfg['max_wall_seconds']!=21600 or cfg['groups_per_branch']!=48
            or status['actual_updates']!=m['actual_updates'] or status['committed_groups']!=m['committed_groups']
            or runner['policies']!=m['latest_policies'] or runner['next_groups']!=m['committed_groups']):
        raise ValueError('frozen budget or transaction progress differs')
    for b in ('control','lr10x'):
        validate_branch(roots[b],m['latest_policies'][b],m['committed_groups'][b],local_review)
    before=protected_tree(output)
    rev.mkdir(parents=True,exist_ok=True)
    if not (rev/'release_before.json').exists():atomic(rev/'release_before.json',old_release)
    shutil.copytree(bundle,rev/'bundle',dirs_exist_ok=True)
    atomic(rev/'protected_before.json',before)
    for b,runroot in roots.items():
        revision=runroot/'source_revisions'/PATCH_ID
        if not (revision/'run_manifest_before.json').exists():atomic(revision/'run_manifest_before.json',old_runs[b])
        shutil.copytree(source_root(runroot,old_runs[b]),revision/'source',dirs_exist_ok=True)
        for name,p in patches.items():
            if p['source_binding']:
                dest=under(revision/'source',name);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(bundle/'payload'/name,dest)
        verify(revision/'source',new_runs[b]['source_sha256'])
    for name,p in patches.items():
        dest=under(root,name);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.probe-hotfix-writing')
        shutil.copyfile(bundle/'payload'/name,tmp);os.replace(tmp,dest)
    atomic(root/RELEASE,new_release)
    for b,runroot in roots.items():atomic(runroot/'run_manifest.json',new_runs[b])
    verify(root,new_release['files'])
    if protected_tree(output)!=before:raise ValueError('protected training/budget/evaluation evidence changed')
    result={'patch_id':PATCH_ID,'installed':True,'local_review_only':local_review,
        'effective_release':new_release,'effective_run_manifests':new_runs,'protected_files_sha256':before,
        'latest_policies':m['latest_policies'],'committed_groups':m['committed_groups'],
        'elapsed_seconds_preserved':status['elapsed_seconds'],'remaining_budget_seconds':21600-status['elapsed_seconds'],
        'cloud_optimizer_hashes_verified':not local_review,'config_unchanged':True}
    atomic(receipt,result)
    print('Probability-probe hotfix installed; groups='+str(m['committed_groups'])+'; updates='+str(m['actual_updates']),flush=True)
    print(f'Remaining original budget: {result["remaining_budget_seconds"]:.0f}s. Continue with --resume.',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--local-review',action='store_true',help='returned-file audit only; omit on the L40S instance')
    a=p.parse_args();apply(a.root,Path(__file__).resolve().parent,a.local_review)
