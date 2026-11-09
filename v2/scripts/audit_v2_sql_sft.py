"""CPU replay audit for parent/weighted train and development evaluations."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.sql_sft import audit_episode,data_identity,normalized_weights
from apt_v2.tasks import make_tasks
from apt_v2.sft_diagnostics import summarize_episodes

def audit(output):
    from transformers import AutoTokenizer
    cfg=read(output/'config.json');sources=read(output/'run_manifest.json')['source_sha256'];verify_files(ROOT,sources)
    identity=read(output/'model_identity.json')
    for name,digest in identity['files_sha256'].items():
        if (output/'tokenizer'/name).exists() and file_hash(output/'tokenizer'/name)!=digest:raise ValueError('tokenizer identity differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    tasks={t.id:t for t in make_tasks()};total=turns=0;comparison={}
    for branch in ('parent','weighted'):
        comparison[branch]={}
        for split,count in cfg['eval']['splits'].items():
            file=output/(branch+'-'+split+'-episodes.jsonl')
            episodes=[json.loads(line) for line in file.read_text(encoding='utf8').splitlines()]
            expected={t.id for t in tasks.values() if t.split==split}
            if len(episodes)!=count or {e['task_id'] for e in episodes}!=expected:raise ValueError('complete distinct split required')
            for e in episodes:
                if e['model_branch']!=branch:raise ValueError('model branch differs')
                turns+=audit_episode(tasks[e['task_id']],e,tok);total+=1
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(file)}
            if summary!=read(output/(branch+'-'+split+'-summary.json')):raise ValueError('evaluation summary differs')
            comparison[branch][split]=summary
    if comparison!=read(output/'comparison_summary.json'):raise ValueError('comparison summary differs')
    complete=read(output/'final/complete.json');verify_files(output/'final/adapter',complete['adapter_sha256'])
    if file_hash(output/'final/state.pt')!=complete['state_sha256']:raise ValueError('final Adam identity differs')
    training=read(output/'training_summary.json')
    if training['optimizer_updates']!=complete['steps'] or training['total_optimizer_updates']!=complete['total_steps']:
        raise ValueError('training counters differ')
    examples=[json.loads(line) for line in (output/'examples.jsonl').read_text(encoding='utf8').splitlines()]
    if data_identity(examples)!=read(output/'input_binding.json')['dataset_sha256']:raise ValueError('training dataset identity differs')
    weights=normalized_weights(examples,cfg['sql_weight'])
    if abs(sum(weights)/len(weights)-1)>1e-10:raise ValueError('weight normalization differs')
    write(output/'audit.json',{'passed':True,'episodes_replayed':total,'turns_replayed':turns,
        'source_files_verified':len(sources),'input_binding':read(output/'input_binding.json'),
        'evaluation_summary':comparison,'audit_optimizer_updates':0})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
