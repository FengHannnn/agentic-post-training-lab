"""Dose warm starts, reusable native train probes and independent ordered-search audit."""
import argparse,copy,shutil,sys,time
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,verify_inputs as branch_inputs,runtime_identity
from apt_v2.grpo import digest
from apt_v2.grpo_dose_study import validate_master,validate_prior,validate_warm,warm_config,recipe,train_gate,selection,DOSES
from apt_v2.grpo_warmup_probe import make_plan
from apt_v2.grpo_repair_study import W_SHA,b_gate


def warm_root(output,name):
    if name not in ['r'+str(r) for r in DOSES]:raise ValueError('unknown dose model')
    return output/'warmups'/name


def probe_model_folder(output,name):
    return output/'branches/w/initial' if name=='w' else warm_root(output,name)/'warm-start/B'


def candidate_source(output,name):
    from grpo_repair_worker import sft_order
    c=read(output/'config.json');root=warm_root(output,name);r=int(name[1:]);wc=read(root/'config.json');validate_warm(wc)
    if wc!=warm_config(c,r):raise ValueError('candidate warm config differs')
    data=read(root/'warm-start/dataset.json')
    if data!=read(ROOT/c['sft_examples']):raise ValueError('candidate data differs')
    identity=digest({'config':wc['sft'],'data':data,'order':sft_order(wc)})
    folder=root/'warm-start/B';meta=read(folder/'complete.json');verify_files(folder/'adapter',meta['adapter_sha256'])
    steps=recipe(r)['steps'];last=root/'warm-start/checkpoints'/f'step-{steps:04}';commit=read(last/'complete.json')
    verify_files(last,commit['files']);sha=file_hash(folder/'adapter/adapter_model.safetensors')
    if (meta['total_sft_updates']!=513+steps or meta['total_dpo_updates']!=11 or meta['total_rl_updates']!=0
            or meta['parent_W_sha256']!=W_SHA or meta['sft_identity']!=identity or commit['identity']!=identity
            or meta['source_checkpoint']!=f'checkpoints/step-{steps:04}' or meta['source_adapter_sha256']!=sha
            or file_hash(last/'adapter/adapter_model.safetensors')!=sha or sha==W_SHA
            or read(folder/'source_config.json')!=recipe(r)
            or read(folder/'source_audit.json')!={'passed':True,'identity':identity,'train_only':True,'actual_updates':steps,'action_exposures':steps*4}):
        raise ValueError('candidate export ancestry/recipe differs')
    return {'adapter_sha256':sha,'source_sft_updates':513+steps}


def verify_probe_inputs(output,local=False):
    c=read(output/'config.json');validate_master(c);manifest=read(output/'run_manifest.json')
    verify_files(ROOT,manifest['source_sha256']);verify_files(output/'source',manifest['source_sha256'])
    if file_hash(output/'release_manifest.json')!=manifest['release_sha256']:raise ValueError('dose release snapshot differs')
    verify_files(ROOT,read(output/'release_manifest.json')['files'])
    branch_inputs(output/'branches/w',runtime=not local,local=local)
    if read(output/'branches/w/execution_environment.json')!=read(output/'execution_environment.json'):
        raise ValueError('within-dose W hardware differs')
    if not local and runtime_identity()!=read(output/'execution_environment.json'):raise ValueError('dose probe runtime differs')
    prepared=output/'prepared';m=read(prepared/'manifest.json');verify_files(prepared,m['files'])
    if m['config_sha256']!=file_hash(output/'config.json'):raise ValueError('dose prepared config differs')
    prior=read(prepared/'inputs/prior-evidence.json');validate_prior(prior,c)
    if prior!=read(ROOT/c['input_dir']/'prior-evidence.json'):raise ValueError('dose prior input differs')
    data=read(prepared/'inputs/sft-examples.json');plan=make_plan(data)
    if data!=read(ROOT/c['sft_examples']) or read(prepared/'inputs/probe-plan.json')!=plan:
        raise ValueError('dose data/probe plan differs')
    if read(prepared/'branch_config.json')!=read(output/'branches/w/config.json'):raise ValueError('probe branch config differs')
    models={'w':{'adapter_sha256':W_SHA,'source_sft_updates':513}}
    for r in DOSES:
        name='r'+str(r)
        if (warm_root(output,name)/'warm-start/B').exists():models[name]=candidate_source(output,name)
    return {**c,'models':models},plan


def prepare(output):
    c=read(output/'config.json');validate_master(c);branch_inputs(output/'branches/w')
    if (output/'prepared').exists():return verify_probe_inputs(output)
    temp=output/('.probe-preparing-'+str(time.time_ns()));shutil.copytree(ROOT/c['input_dir'],temp/'inputs')
    shutil.copytree(output/'branches/w/tokenizer',temp/'tokenizer')
    shutil.copyfile(output/'branches/w/config.json',temp/'branch_config.json')
    atomic_json(temp/'manifest.json',{'config_sha256':file_hash(output/'config.json'),
        'files':{p.relative_to(temp).as_posix():file_hash(p) for p in sorted(temp.rglob('*')) if p.is_file()}})
    temp.rename(output/'prepared');return verify_probe_inputs(output)


def warm(output,name):
    from grpo_repair_worker import warm_start
    c=read(output/'config.json');r=int(name[1:]);wc=warm_config(c,r);target=warm_root(output,name)
    if not target.exists():
        temp=output/'warmups'/('.'+name+'-'+str(time.time_ns()));temp.mkdir(parents=True)
        atomic_json(temp/'config.json',wc)
        shutil.copytree(output/'branches/w',temp/'branches/w',
                        ignore=lambda folder,names:['evaluations'] if Path(folder)==output/'branches/w' else [])
        temp.rename(target)
    elif read(target/'config.json')!=wc:raise ValueError('committed warm recipe differs')
    warm_start(target);candidate_source(output,name)


def probe_summary(output,name):
    from transformers import AutoTokenizer
    from grpo_warmup_probe_worker import verify_model
    tok=AutoTokenizer.from_pretrained(output/'prepared/tokenizer',local_files_only=True,trust_remote_code=False)
    summary=verify_model(output,name,tok,protocol=sys.modules[__name__])
    rows=[read(output/'probes'/name/'rows'/f'{i:04}.json') for i in range(12)]
    counts=[sum(bool(set(r['executions'][ri]['new_verified_metrics'])&{'current_net_cents','prior_net_cents'}) for r in rows) for ri in (0,1)]
    return {**summary,'target_certifications_by_repeat':counts}


def gate(output,name,write=True):
    from grpo_vllm import projected_checkpoint_tensors
    baseline=probe_summary(output,'w');candidate=probe_summary(output,name)
    changed=projected_checkpoint_tensors(probe_model_folder(output,'w')/'adapter')!=projected_checkpoint_tensors(probe_model_folder(output,name)/'adapter')
    result=train_gate(baseline,candidate,changed);p=output/'train-gates'/(name+'.json')
    if (not write and not p.exists()) or (p.exists() and read(p)!=result):raise ValueError('committed train gate differs')
    if write:atomic_json(p,result)
    return result


def audit(output,local=False):
    from grpo_repair_worker import audit_sft
    from run_v2_grpo_repair import evaluation
    from run_v2_grpo_dose import branch_config
    cfg,plan=verify_probe_inputs(output,local);w=evaluation(output/'branches/w','weighted_sft',local)
    probe_summary(output,'w');records=[];counts={}
    for r in tqdm(DOSES,desc='Audit target doses',unit='candidate',file=sys.stdout,disable=False):
        name='r'+str(r);record_path=output/'decisions'/(name+'.json')
        if not record_path.exists():break
        root=warm_root(output,name);sft=audit_sft(root,local);counts[name]=sft['sft_updates']
        train=gate(output,name,write=False);full=None
        if train['passed']:
            child=output/'branches'/name;branch_inputs(child,runtime=not local,local=local)
            if read(child/'execution_environment.json')!=read(output/'execution_environment.json'):
                raise ValueError('within-dose candidate hardware differs')
            if read(child/'config.json')!=branch_config(cfg,output,name):raise ValueError('candidate evaluation config differs')
            b=evaluation(child,'weighted_sft',local);full=b_gate(w,b)
            if full!=read(output/'full-gates'/(name+'.json')):raise ValueError('full B gate differs')
        expected={'dose':r,'train_gate':train,'full_gate':full,'adapter_sha256':candidate_source(output,name)['adapter_sha256']}
        if read(record_path)!=expected:raise ValueError('dose decision differs')
        records.append(expected)
        if full and full['passed']:break
    expected=selection(w,records)
    if read(output/'selection.json')!=expected:raise ValueError('dose selection differs')
    visited={str(r['dose']) for r in records}
    if any(p.is_dir() and not p.name.startswith('.') and p.name[1:] not in visited for p in (output/'warmups').glob('*')):
        raise ValueError('candidate work outside registered search order')
    ready=output/'ready-B.json'
    if expected['selected_warm_start']:
        name=expected['selected_warm_start'];identity=candidate_source(output,name)
        wanted={**identity,'model_dir':f'warmups/{name}/warm-start/B','dose':int(name[1:]),'source_sft_updates':identity['source_sft_updates'],
                'source_dpo_updates':11,'source_rl_updates':0,'parent_W_sha256':W_SHA}
        if not ready.exists() or read(ready)!=wanted:raise ValueError('ready B export differs')
    elif ready.exists():raise ValueError('unqualified B exported as ready')
    result={'passed':True,'scope':'local_returned_files' if local else 'cloud_with_state','actual_sft_updates':counts,
            'actual_rl_updates':0,'selection':expected,'doses_evaluated':[r['dose'] for r in records]}
    atomic_json(output/('local_audit.json' if local else 'audit.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','warm','probe','gate','audit']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',choices=['w','r4','r8','r16']);p.add_argument('--local',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare(a.output)
    elif a.stage=='warm':warm(a.output,a.name)
    elif a.stage=='probe':
        from grpo_warmup_probe_worker import probe_model
        probe_model(a.output,a.name,protocol=sys.modules[__name__])
    elif a.stage=='gate':gate(a.output,a.name)
    else:audit(a.output,a.local)
