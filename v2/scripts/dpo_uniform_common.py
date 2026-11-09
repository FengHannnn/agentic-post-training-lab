"""Pinned C1 inputs, paired development schedules and summary contracts."""
import sys,json
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.hard_recovery_sft import recovery_weights,verify_weighted_updates
from apt_v2.dpo_recovery_training import validate_training_pairs
from apt_v2.dpo_recovery import SCENARIOS,build_context
from apt_v2.recovery_diagnostics import messages_id,summarize
from apt_v2.sft_diagnostics import summarize_episodes

CONFIG='dpo-uniform-control-v1.json'

def load(p):return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
def jsonl(p,rows):
    with p.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def tasks():return [t for t in make_tasks() if t.split=='development']
def contexts():return [build_context(t,s,allowed_split='development') for t in tasks() for s in SCENARIOS]
def checkpoints(cfg,output):return {'parent':ROOT/'artifacts'/cfg['initial_run']/'final','weighted_sft':ROOT/'artifacts'/cfg['comparison_run']/'final','weighted_dpo':ROOT/'artifacts'/cfg['weighted_dpo_run']/'final','dpo':output/'final'}
def seed_for(cfg,task,mode,ri,context=None):
    ti=[t.id for t in tasks()].index(task.id)
    if context is not None:return cfg['recovery_eval']['seed']+ti*100000+list(SCENARIOS).index(context['scenario'])*1000+ri*100
    return (cfg['eval']['seed']+ti*10000 if mode=='greedy' else cfg['sample_eval']['seed']+ti*10000+ri*100)
def verify_preflight(output):
    cfg=read(output/'config.json');b=read(output/'preflight.json')
    verify_intervention(cfg)
    if b['probes_sha256']!=file_hash(output/'probes.jsonl') or b['intervention_sha256']!=file_hash(output/'intervention.json') or read(output/'intervention.json')!=verify_intervention(cfg):raise ValueError('frozen diagnostic inputs differ')
    if cfg!=read(ROOT/'configs'/CONFIG) or b['config_sha256']!=file_hash(output/'config.json'):raise ValueError('fixed matched DPO config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(ROOT,b['provenance'])
    for branch,p in checkpoints(cfg,output).items():
        if branch=='dpo':continue
        verify_files(p,b['checkpoints'][branch])
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',b['model_files'])
    if contexts()!=load(output/'recovery_contexts.jsonl') or b['contexts_sha256']!=file_hash(output/'recovery_contexts.jsonl'):raise ValueError('frozen development contexts differ')
    if read(output/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('development tasks differ')
    return cfg

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed config required')
    parent=ROOT/'artifacts'/cfg['initial_run'];comp=ROOT/'artifacts'/cfg['comparison_run'];data=ROOT/'artifacts'/cfg['data_run']
    cc=read(comp/'config.json');pc=read(parent/'final/complete.json')
    for key in ('initial_run','adapter_sha256','data_run','pairs_sha256','pair_count','epochs','max_steps','gradient_accumulation','learning_rate','seed','max_grad_norm','source_sft_updates','source_dpo_updates','max_length'):
        if cc[key]!=cfg[key]:raise ValueError('weighted SFT matching differs: '+key)
    if pc['algorithm']!='chosen_only_sft' or pc['total_sft_updates']!=255 or pc['total_dpo_updates']!=11:raise ValueError('parent ancestry differs')
    verify_weighted_updates(cc,read(comp/'final/complete.json'),read(comp/'training_summary.json'))
    provenance=[];bound={};identities=[]
    for branch,run,expected in [('parent',parent,cfg['adapter_sha256']),('weighted_sft',comp,cfg['comparison_adapter_sha256']),('weighted_dpo',ROOT/'artifacts'/cfg['weighted_dpo_run'],cfg['weighted_dpo_adapter_sha256'])]:
        if not read(run/'audit.json')['passed']:raise ValueError('audited comparison required')
        if file_hash(run/'final/complete.json')!=cfg['control_complete_sha256'][branch]:raise ValueError('E1-pinned control completion differs')
        complete=read(run/'final/complete.json');verify_files(run/'final/adapter',complete['adapter_sha256'])
        if file_hash(run/'final/adapter/adapter_model.safetensors')!=expected:raise ValueError('pinned adapter differs')
        bound[branch]={'complete.json':file_hash(run/'final/complete.json'),**{'adapter/'+n:h for n,h in complete['adapter_sha256'].items()}}
        identities.append(read(run/'model_identity.json'))
        provenance.extend(run/n for n in ('config.json','audit.json','training_summary.json','model_identity.json','input_binding.json','final/complete.json'))
    if any(x!=identities[0] for x in identities) or identities[0]['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('base model identity differs')
    if not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('audited train pairs differ')
    pairs=load(data/'pairs.jsonl');validate_training_pairs(pairs);weights,ws=recovery_weights(pairs,cfg['hard_weight'])
    old_weights,old_ws=recovery_weights(pairs,4)
    if read(comp/'weights.json')!={'weights':old_weights,'summary':old_ws}:raise ValueError('original scenario weights differ')
    if cfg['hard_weight']!=1 or any(w!=1 for w in weights):raise ValueError('uniform weight intervention required')
    if file_hash(parent/'pairs.jsonl')!=cfg['pairs_sha256'] or file_hash(comp/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('comparison datasets differ')
    verify_intervention(cfg)
    diagnostic=ROOT/'artifacts'/cfg['diagnosis_run']
    if file_hash(diagnostic/'config.json')!=cfg['diagnosis_config_sha256'] or file_hash(diagnostic/'audit.json')!=cfg['diagnosis_audit_sha256']:raise ValueError('reviewed E1 identity differs')
    if not read(diagnostic/'audit.json')['passed'] or read(diagnostic/'run_status.json')['error'] is not None:raise ValueError('completed E1 diagnosis required')
    from apt_v2.dpo_investigation import build_probes
    probes=build_probes()
    if probes!=load(diagnostic/'probes.jsonl'):raise ValueError('E1 fixed expert probes differ')
    jsonl(output/'probes.jsonl',probes)
    write(output/'intervention.json',verify_intervention(cfg))
    provenance.extend(diagnostic/n for n in ('config.json','audit.json','diagnosis_summary.json','run_status.json','probes.jsonl'))
    provenance.extend([ROOT/'artifacts'/cfg['weighted_dpo_run']/'reference_scores.json'])
    cs=contexts();seen={messages_id(p['messages']) for p in pairs}
    if any(messages_id(c['messages']) in seen for c in cs):raise ValueError('development context overlaps training')
    jsonl(output/'recovery_contexts.jsonl',cs);write(output/'development_tasks.json',[asdict(t) for t in tasks()])
    provenance.extend([data/'pairs.jsonl',data/'audit.json',data/'pair_summary.json',parent/'pairs.jsonl',comp/'pairs.jsonl',comp/'weights.json'])
    write(output/'preflight.json',{'config_sha256':file_hash(output/'config.json'),'checkpoints':bound,'model_files':identities[0]['files_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance},'contexts_sha256':file_hash(output/'recovery_contexts.jsonl'),
        'probes_sha256':file_hash(output/'probes.jsonl'),'intervention_sha256':file_hash(output/'intervention.json'),'pairs':len(pairs),'weight_summary':ws,'development_contexts':len(cs),'test_split_used':False})
    import shutil
    for p in provenance:
        target=output/'provenance'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    verify_preflight(output)

def result_summary(complete,recoveries,actions,cfg):
    result={}
    def paired(a,b,key):
        x={key(e):e['success'] for e in a};y={key(e):e['success'] for e in b}
        if len(x)!=len(a) or len(y)!=len(b) or x.keys()!=y.keys():raise ValueError('distinct paired evaluation keys required')
        return {'both_success':sum(x[k] and y[k] for k in x),'both_failure':sum(not x[k] and not y[k] for k in x),
            'improved':sum(not x[k] and y[k] for k in x),'regressed':sum(x[k] and not y[k] for k in x)}
    def es(es):return {'episodes':len(es),'successes':sum(e['success'] for e in es),'turns':sum(len(e['turns']) for e in es),'tool_errors':sum(e['tool_errors'] for e in es)}
    for b in cfg['evaluation_branches']:
        result[b]={}
        for mode in ('greedy','sample'):
            full=[e for e in complete if e['model_branch']==b and e['mode']==mode];rs=[e for e in recoveries if e['model_branch']==b and e['mode']==mode];aa=[r for r in actions if r['model_branch']==b and r['mode']==mode]
            result[b][mode]={'complete':{**summarize_episodes(full),'episodes':len(full)},'complete_by_task':{t.id:es([e for e in full if e['task_id']==t.id]) for t in tasks()},
                'recovery':es(rs),'recovery_by_task':{t.id:es([e for e in rs if e['task_id']==t.id]) for t in tasks()},
                'recovery_by_scenario':{s:es([e for e in rs if e['scenario']==s]) for s in SCENARIOS},'first_action':summarize(aa)}
    paired_results={}
    for baseline in ('parent','weighted_sft','weighted_dpo'):
        paired_results[baseline+'_to_dpo']={}
        for mode in ('greedy','sample'):
            pick=lambda rows,b:[e for e in rows if e['model_branch']==b and e['mode']==mode]
            paired_results[baseline+'_to_dpo'][mode]={'complete':paired(pick(complete,baseline),pick(complete,'dpo'),lambda e:(e['task_id'],e['rollout_index'])),
                'recovery':paired(pick(recoveries,baseline),pick(recoveries,'dpo'),lambda e:(e['context_id'],e['sample_index']))}
    return {'branches':result,'paired':paired_results,'test_split_used':False,'recovery_turn_budget':'original task max_turns includes injected prefix'}


TRAINING_KEYS=('algorithm','initial_run','adapter_sha256','data_run','pairs_sha256','pair_count','max_length','hard_weight','epochs','max_steps','gradient_accumulation','learning_rate','max_grad_norm','seed','checkpoint_every','checkpoint_steps','source_sft_updates','source_dpo_updates','beta')
def intervention(original,current):
    changed=[k for k in TRAINING_KEYS if original[k]!=current[k]]
    if changed!=['hard_weight'] or original['hard_weight']!=4 or current['hard_weight']!=1:raise ValueError('only hard_weight 4 to 1 may change')
    for k in ('eval','sample_eval','recovery_eval','task_counts','checkpoint_selection'):
        if original[k]!=current[k]:raise ValueError('paired protocol differs: '+k)
    if current['evaluation_branches']!=['parent','weighted_sft','weighted_dpo','dpo']:raise ValueError('four fixed branches required')
    return {'experiment':'E2_uniform_weight_control','changed_training_fields':changed,'raw_hard_weight_before':4,'raw_hard_weight_after':1,'original_run':original['run_id'],'uniform_run':current['run_id'],'fixed_training_fields':[k for k in TRAINING_KEYS if k not in changed],'test_split_used':False}
def verify_intervention(cfg):
    original=read(ROOT/'artifacts'/cfg['weighted_dpo_run']/'config.json')
    if original!=read(ROOT/'configs/hard-recovery-dpo-matched-v1.json'):raise ValueError('original weighted config differs')
    return intervention(original,cfg)
def model_specs(cfg,output):
    result={}
    for b,p in checkpoints(cfg,output).items():
        if b!='dpo' and file_hash(p/'complete.json')!=cfg['control_complete_sha256'][b]:raise ValueError('E1-pinned control completion differs')
        complete=read(p/'complete.json');verify_files(p/'adapter',complete['adapter_sha256'])
        sha=file_hash(p/'adapter/adapter_model.safetensors')
        expected={'parent':cfg['adapter_sha256'],'weighted_sft':cfg['comparison_adapter_sha256'],'weighted_dpo':cfg['weighted_dpo_adapter_sha256']}.get(b)
        if expected is not None and sha!=expected:raise ValueError('scored control adapter differs')
        if b=='dpo' and (complete['steps']!=258 or complete['total_sft_updates']!=255 or complete['total_dpo_updates']!=269):raise ValueError('uniform checkpoint ancestry differs')
        result[b]={'run':p.parent.name,'checkpoint':p.name,'adapter_sha256':sha,'complete_sha256':file_hash(p/'complete.json')}
    return result
def checkpoint_path(spec):return ROOT/'artifacts'/spec['run']/spec['checkpoint']
def verify_inputs(output):
    cfg=verify_preflight(output);binding=read(output/'input_binding.json')
    if binding['baseline_adapter_sha256']!=cfg['adapter_sha256'] or binding['config_sha256']!=file_hash(output/'config.json') or binding['weights_sha256']!=file_hash(output/'weights.json') or file_hash(output/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('training binding differs')
    verify_files(ROOT,binding['provenance'])
    weights,ws=recovery_weights(load(output/'pairs.jsonl'),1)
    if read(output/'weights.json')!={'weights':weights,'summary':ws}:raise ValueError('uniform weights differ')
    for name in (*[f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']],'final'):
        p=output/name;verify_files(p/'adapter',read(p/'complete.json')['adapter_sha256'])
        if read(p/'binding.json')!=binding:raise ValueError('checkpoint binding differs')
    return cfg

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
