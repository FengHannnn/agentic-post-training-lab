"""Environment preflight and explicitly requested pinned model download."""
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,source_files,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.runtime import run_episode,ScriptedPolicy

def preflight():
    cfg=read(ROOT/'configs/baseline.json')
    if cfg['model_id']!='Qwen/Qwen2.5-1.5B-Instruct' or cfg['revision']!='989aa79':
        raise ValueError('pinned 1.5B model required')
    tasks=make_tasks()
    groups={s:{t.family_id for t in tasks if t.split==s} for s in ('train','development','test')}
    if any(groups[a]&groups[b] for a,b in [('train','development'),('train','test'),('development','test')]):
        raise ValueError('task family overlap')
    episodes=[run_episode(t,ScriptedPolicy(t)) for t in tasks]
    if not all(e['success'] for e in episodes):raise ValueError('reference environment check failed')
    if any(abs(e['process_reward_sum'])>1e-8 for e in episodes):raise ValueError('potential did not telescope')
    return {'passed':True,'model_id':cfg['model_id'],'revision':cfg['revision'],
        'tasks_by_split':{s:sum(t.split==s for t in tasks) for s in groups},
        'scripted_reference_successes':sum(e['success'] for e in episodes),'source_sha256':source_files(),
        'fault_episodes':sum(t.fault_mode!='none' for t in tasks),
        'fault_episodes_recovered':sum(e['recovered'] for e in episodes),
        'training_optimizer_steps':0,'model_loaded':False}

def download():
    cfg=read(ROOT/'configs/baseline.json')
    dest=ROOT/'.cache/qwen2.5-1.5b-instruct'
    identity=dest/'model_identity.json'
    if identity.exists():
        saved=read(identity)
        if saved['model_id']!=cfg['model_id'] or not saved['resolved_revision'].startswith(cfg['revision']):
            raise ValueError('cached model binding differs')
        verify_files(dest,saved['files_sha256']);return saved
    os.environ.pop('HF_HUB_OFFLINE',None)
    from huggingface_hub import HfApi,snapshot_download
    resolved=HfApi().model_info(cfg['model_id'],revision=cfg['revision']).sha
    if not resolved.startswith(cfg['revision']):raise ValueError('revision resolution differs')
    snapshot_download(cfg['model_id'],revision=resolved,local_dir=dest,
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    config=read(dest/'config.json')
    if config['model_type']!='qwen2' or config['num_hidden_layers']!=28:
        raise ValueError('unexpected model architecture')
    files={p.relative_to(dest).as_posix():file_hash(p) for p in sorted(dest.iterdir()) if p.is_file()}
    result={'model_id':cfg['model_id'],'requested_revision':cfg['revision'],'resolved_revision':resolved,'files_sha256':files}
    write(identity,result);return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--download-model',action='store_true')
    args=parser.parse_args()
    print(json.dumps(download() if args.download_model else preflight(),ensure_ascii=False))
