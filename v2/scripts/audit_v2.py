"""Replay complete agent episodes against the frozen deterministic environment."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,OUTPUT,read,write,verify_files,file_hash
from apt_v2.tasks import make_tasks
from apt_v2.environment import Environment
from apt_v2.runtime import parse_action

def audit(output=OUTPUT):
    manifest=read(output/'run_manifest.json')
    verify_files(output/'source',manifest['source_sha256']);verify_files(ROOT,manifest['source_sha256'])
    cfg=read(ROOT/'configs/baseline.json')
    if manifest['config']!=cfg:raise ValueError('baseline config differs')
    identity=read(output/'model_identity.json')
    if identity['model_id']!=cfg['model_id'] or not identity['resolved_revision'].startswith(cfg['revision']):
        raise ValueError('model revision differs')
    verify_files(output/'tokenizer',{k:v for k,v in identity['files_sha256'].items() if k.endswith(('.json','.txt','.jinja'))})
    tasks=[t for t in make_tasks() if t.split==cfg['split']][:cfg['task_count']]
    episodes=[json.loads(line) for line in (output/'episodes.jsonl').read_text(encoding='utf8').splitlines()]
    if [e['task_id'] for e in episodes]!=[t.id for t in tasks]:raise ValueError('task coverage differs')
    tokenizer=None
    if (output/'tokenizer/tokenizer_config.json').exists():
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    for task,episode in zip(tasks,episodes):
        env=Environment(task,cfg['reward_mode'])
        from apt_v2.runtime import SYSTEM
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
        try:
            for saved in episode['turns']:
                generation=saved['generation']
                try:action=parse_action(generation['text'])
                except (ValueError,TypeError):action={'invalid_action':generation['text']}
                actual=env.step(action)
                if any(saved[k]!=v for k,v in actual.items()):raise ValueError('environment replay differs')
                if generation['mode']!='greedy' or generation['behavior_token_log_probs'] is not None:
                    raise ValueError('baseline probability semantics differ')
                ids=generation['response_token_ids'];raw=generation['raw_token_log_probs']
                import math
                if not ids or len(ids)!=len(raw) or any(not math.isfinite(v) or v>1e-6 for v in raw):
                    raise ValueError('native action probabilities differ')
                if tokenizer:
                    if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=generation['text']:
                        raise ValueError('token decode differs')
                    if tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)!=generation['prompt_token_ids']:
                        raise ValueError('action prompt differs')
                messages.extend([{'role':'assistant','content':generation['text']},
                    {'role':'user','content':json.dumps({'observation':saved['observation']})}])
            if not env.done or env.success!=episode['success']:raise ValueError('terminal state differs')
            if (env.tool_errors!=episode['tool_errors'] or (env.success and env.progress_after_error)!=episode['recovered']
                or task.fault_mode!=episode['fault_mode']):raise ValueError('error recovery metrics differ')
            actual_artifact=env.records[-1]['observation'].get('artifact') if env.success else None
            if episode['artifact']!=actual_artifact:raise ValueError('delivered artifact differs')
            if abs(sum(r['process_reward'] for r in env.records))>1e-8:raise ValueError('shaping does not telescope')
            if abs(sum(r['reward'] for r in env.records)-episode['total_reward'])>1e-8:raise ValueError('return differs')
        finally:env.close()
    summary=read(output/'summary.json')
    expected={'tasks':len(episodes),'successes':sum(e['success'] for e in episodes),
        'fault_tasks':sum(e['fault_mode']!='none' for e in episodes),
        'fault_tasks_recovered':sum(e['recovered'] for e in episodes),
        'total_tool_errors':sum(e['tool_errors'] for e in episodes),
        'total_turns':sum(len(e['turns']) for e in episodes),
        'response_tokens':sum(len(r['generation']['response_token_ids']) for e in episodes for r in e['turns']),
        'episodes_sha256':file_hash(output/'episodes.jsonl')}
    if any(summary[k]!=v for k,v in expected.items()):raise ValueError('baseline metrics differ')
    result={'passed':True,**expected,'tokenizer_checks':tokenizer is not None,
        'optimizer_updates':0,'training_admitted':False}
    write(output/'audit.json',result)
    return result
if __name__=='__main__':print(json.dumps(audit()))
