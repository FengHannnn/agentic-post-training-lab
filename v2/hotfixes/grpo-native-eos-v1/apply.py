"""Apply one hash-bound validator correction and preserve the existing formal job."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

PATCH_ID='grpo-native-eos-v1'

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def verify(root,files):
    for name,expected in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=expected:
            raise ValueError('hash-bound file differs: '+name)
def tree(root):
    paths=[root/n for n in ('config.json','input_binding.json','runner_state.json','run_status.json')]
    paths += [p for n in ('groups','transitions','checkpoints','evaluations') for p in (root/n).rglob('*') if p.is_file() and '.writing' not in p.name]
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}

def apply(root,bundle,local_review=False):
    root=Path(root).resolve(); bundle=Path(bundle).resolve(); m=read(bundle/'manifest.json')
    if m['patch_id']!=PATCH_ID: raise ValueError('patch identity differs')
    verify(bundle,m['bundle_files'])
    output=root/'artifacts'/m['run_id']; rev=output/'source_revisions'/PATCH_ID
    receipt=rev/'applied.json'
    if receipt.exists():
        r=read(receipt); verify(root,r['effective_release']['files'])
        if r['local_review_only'] and not local_review:
            raise ValueError('local-review installation is not a cloud-state verification')
        if read(root/'grpo_release_manifest.json')!=r['effective_release'] or read(output/'run_manifest.json')!=r['effective_run_manifest']:
            raise ValueError('already-applied manifest differs')
        print('Hotfix already applied and verified; continue with --resume',flush=True)
        return r
    # Backups double as a recovery journal if installation was interrupted.
    release_before=rev/'release_before.json'; run_before=rev/'run_manifest_before.json'
    old_release=read(release_before if release_before.exists() else root/'grpo_release_manifest.json')
    old_run=read(run_before if run_before.exists() else output/'run_manifest.json')
    expected_release=read(bundle/'baseline_release.json'); expected_run=read(bundle/'baseline_run_manifest.json')
    if old_release!=expected_release or old_run!=expected_run: raise ValueError('published baseline differs')
    patches=m['patches']; new_release=json.loads(json.dumps(old_release)); new_run=json.loads(json.dumps(old_run))
    amendment={'patch_id':PATCH_ID,'manifest_sha256':sha(bundle/'manifest.json'),'patches':patches,
               'scope':'native termination validation only; same formal job/config/optimizer/budget',
               'source_location':'source_revisions/'+PATCH_ID+'/source'}
    for name,p in patches.items():
        if old_release['files'].get(name)!=p['before'] or old_run['source_sha256'].get(name)!=p['before']:
            raise ValueError('patch baseline source binding differs')
        if sha(bundle/'payload'/name)!=p['after']: raise ValueError('patch payload differs')
        new_release['files'][name]=p['after']; new_run['source_sha256'][name]=p['after']
    new_release['source_amendments']=[amendment]; new_run['source_amendments']=[amendment]
    if read(root/'grpo_release_manifest.json') not in (old_release,new_release) or read(output/'run_manifest.json') not in (old_run,new_run):
        raise ValueError('installation journal or active manifest differs')
    for name,expected in old_release['files'].items():
        permitted={expected,patches[name]['after']} if name in patches else {expected}
        if sha(root/name) not in permitted: raise ValueError('active source/input differs: '+name)
    verify(output/'source',old_run['source_sha256'])
    for name,expected in m['run_bindings'].items():
        if sha(output/name)!=expected: raise ValueError('returned run identity differs: '+name)
    verify(output,read(output/'input_binding.json')['files'])
    cfg=read(output/'config.json'); state={'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':cfg['adapter_sha256']}
    for i in range(m['committed_groups']):
        tr=read(output/'transitions'/f'group-{i:04}.json')
        if tr['index']!=i or tr['before']!=state or tr['config_sha256']!=sha(output/'config.json') or tr['group_sha256']!=sha(output/'groups'/f'group-{i:04}.json'):
            raise ValueError('committed transition differs')
        if tr['updated']:
            folder=output/tr['after']['checkpoint']; complete=read(folder/'complete.json'); verify(folder,complete['files'])
            if complete['actual_updates']!=tr['after']['updates'] or complete['group_cursor']!=i+1:
                raise ValueError('committed optimizer cursor differs')
        elif tr['after']!=state: raise ValueError('skipped transition changed state')
        state=tr['after']
    if state!=m['latest_policy'] or read(output/'runner_state.json')['policy']!=state:
        raise ValueError('latest policy differs')
    if not local_review:
        folder=output/state['checkpoint']
        if sha(folder/'state.pt')!=read(folder/'complete.json')['state_sha256']:
            raise ValueError('cloud Adam/RNG state differs')
    before=tree(output)
    print('Verified 17 committed groups, latest RL13, frozen inputs and '+('returned state evidence' if local_review else 'cloud Adam/RNG hash'),flush=True)
    rev.mkdir(parents=True,exist_ok=True)
    if not release_before.exists(): atomic(release_before,old_release)
    if not run_before.exists(): atomic(run_before,old_run)
    atomic(rev/'patch_manifest.json',m)
    shutil.copytree(bundle,rev/'bundle',dirs_exist_ok=True)
    for name,p in patches.items():
        target=root/name; tmp=target.with_name(target.name+'.eos-hotfix-writing')
        shutil.copyfile(bundle/'payload'/name,tmp); os.replace(tmp,target)
        snapshot=rev/'source'/name; snapshot.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(bundle/'payload'/name,snapshot)
    atomic(root/'grpo_release_manifest.json',new_release); atomic(output/'run_manifest.json',new_run)
    verify(root,new_release['files']); verify(root,new_run['source_sha256'])
    if tree(output)!=before: raise ValueError('protected training/evaluation/budget files changed')
    result={'patch_id':PATCH_ID,'installed':True,'local_review_only':local_review,'effective_release':new_release,
            'effective_run_manifest':new_run,'protected_files_sha256':before,'latest_policy':state,
            'committed_groups':m['committed_groups'],'elapsed_seconds_preserved':read(output/'run_status.json')['elapsed_seconds'],
            'cloud_optimizer_hash_verified':not local_review,'config_unchanged':True}
    atomic(receipt,result)
    print('EOS validator hotfix installed; source provenance saved; resume the existing job.',flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); p.add_argument('--local-review',action='store_true',help='returned-files validation only; cloud execution must omit this flag')
    a=p.parse_args(); apply(a.root,Path(__file__).resolve().parent,a.local_review)
