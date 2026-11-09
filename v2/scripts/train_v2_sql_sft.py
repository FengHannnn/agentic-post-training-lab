"""Continue the reviewed SFT adapter with SQL-weighted mixed action loss."""
import sys,json,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.sft import encode_example,action_loss
from apt_v2.sql_sft import normalized_weights,validate_examples,data_identity,backward_batch
from apt_v2.sft_diagnostics import score_action,grouped_scores
from apt_v2.runtime import parse_action

def train(output,resume=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    cfg=read(output/'config.json');parent=ROOT/'artifacts'/cfg['parent_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    manifest=read(output/'run_manifest.json')['source_sha256'];verify_files(ROOT,manifest)
    parent_complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',parent_complete['adapter_sha256'])
    expected='95f908b1e9163907f9e72eb2c5aa5442da873193f7fb730352ed35382c7b379e'
    if file_hash(parent/'final/adapter/adapter_model.safetensors')!=expected:raise ValueError('reviewed parent adapter required')
    if file_hash(parent/'final/state.pt')!=parent_complete['state_sha256']:raise ValueError('parent Adam state differs')
    parent_config=read(parent/'config.json')
    for key in ('learning_rate','gradient_accumulation','lora','seed','max_length'):
        if cfg[key]!=parent_config[key]:raise ValueError('unchanged pilot setting required: '+key)
    if cfg['epoch_offset']!=parent_config['epochs']:raise ValueError('epoch offset differs')
    identity=read(parent/'model_identity.json');verify_files(folder,identity['files_sha256'])
    if identity['resolved_revision']!=cfg['revision']:raise ValueError('fixed model revision required')
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    examples=[json.loads(line) for line in (parent/'examples.jsonl').read_text(encoding='utf8').splitlines()]
    data_summary=read(parent/'data_summary.json')
    if file_hash(parent/'examples.jsonl')!=data_summary['examples_sha256']:raise ValueError('parent example file differs')
    tools=validate_examples(examples,data_summary);weights=normalized_weights(examples,cfg['sql_weight'])
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    samples=[encode_example(tok,e,cfg['max_length']) for e in examples]
    input_binding={'parent_adapter_sha256':expected,'parent_state_sha256':parent_complete['state_sha256'],
        'dataset_sha256':data_identity(examples),'source_sha256':manifest,'config_sha256':file_hash(output/'config.json')}
    if not resume:
        import shutil
        for name in ('examples.jsonl','demonstrations.jsonl'):shutil.copyfile(parent/name,output/name)
        write(output/'model_identity.json',identity);write(output/'input_binding.json',input_binding)
        write(output/'data_summary.json',{'examples':len(examples),'dataset_sha256':data_identity(examples),
            'parent_examples_sha256':data_summary['examples_sha256'],'tool_counts':tools,'sql_weight':cfg['sql_weight'],
            'weight_normalization':'dataset mean action weight equals one','normalized_weight_by_tool':{
                parse_action(e['response'])['tool']:w for e,w in zip(examples,weights)},
            'max_tokens':max(len(s['input_ids']) for s in samples)})
    else:
        if read(output/'input_binding.json')!=input_binding:raise ValueError('resume input differs')
        if read(resume/'binding.json')!=input_binding:raise ValueError('resume checkpoint binding differs')
        complete=read(resume/'complete.json');verify_files(resume/'adapter',complete['adapter_sha256'])
        if file_hash(resume/'state.pt')!=complete['state_sha256']:raise ValueError('resume optimizer differs')
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,(resume/'adapter') if resume else (parent/'final/adapter'),is_trainable=True).eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('reviewed LoRA structure required')
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    state=torch.load(resume/'state.pt' if resume else parent/'final/state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    parent_steps=read(parent/'training_summary.json')['optimizer_updates']
    steps=state['steps'] if resume else 0;cursor=state['cursor'] if resume else 0;history=state['history'] if resume else []
    if not resume and (state['steps']!=63 or state['cursor']!=504 or parent_steps!=63):raise ValueError('reviewed 63-step parent required')
    optimizer_steps={int(s['step']) for s in optimizer.state.values()}
    if optimizer_steps!={parent_steps+steps}:raise ValueError('Adam counter differs from logical steps')
    if any(g['lr']!=cfg['learning_rate'] or g['weight_decay']!=0 for g in optimizer.param_groups):raise ValueError('inherited Adam settings differ')
    initial=[p.detach().cpu().clone() for p in parameters]
    def evaluate_actions(name):
        rows=[];model.eval()
        for e,s in zip(examples,samples):
            score=score_action(model,s);rows.append({**{k:score[k] for k in ('tokens','nll_sum','correct_tokens','all_tokens_correct')},
                'tool':parse_action(e['response'])['tool'],'variant':e['variant']})
        write(output/name,grouped_scores(rows));model.train()
    if not resume:evaluate_actions('before_action_scores.json')
    order=[]
    for epoch in range(cfg['epoch_offset'],cfg['epoch_offset']+cfg['epochs']):
        indices=list(range(len(samples)));random.Random(cfg['seed']+epoch).shuffle(indices);order+=indices
    def save(name):
        checkpoint=output/name;checkpoint.mkdir()
        model.save_pretrained(checkpoint/'adapter',safe_serialization=True)
        write(checkpoint/'binding.json',input_binding)
        torch.save({'optimizer':optimizer.state_dict(),'cursor':cursor,'steps':steps,'total_steps':parent_steps+steps,'history':history,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint/'state.pt')
        write(checkpoint/'complete.json',{'steps':steps,'total_steps':parent_steps+steps,'cursor':cursor,
            'adapter_sha256':{p.name:file_hash(p) for p in (checkpoint/'adapter').iterdir() if p.is_file()},
            'state_sha256':file_hash(checkpoint/'state.pt')})
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();model.train()
    while cursor<len(order) and steps<cfg['max_steps']:
        batch=order[cursor:cursor+cfg['gradient_accumulation']];optimizer.zero_grad(set_to_none=True)
        value=backward_batch(model,samples,weights,batch)
        norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'],error_if_nonfinite=True)
        if norm.item()<=0:raise ValueError('nonzero adapter gradient required')
        optimizer.step();steps+=1;cursor+=len(batch)
        row={'step':steps,'total_steps':parent_steps+steps,'cursor':cursor,'weighted_loss':value,'grad_norm':norm.item(),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30};history.append(row);print(json.dumps(row),flush=True)
        if steps%cfg['checkpoint_every']==0:save(f'checkpoint-{steps:04}')
    save('final');evaluate_actions('after_action_scores.json')
    write(output/'training_summary.json',{'optimizer_updates':steps,'parent_optimizer_updates':parent_steps,
        'total_optimizer_updates':parent_steps+steps,'optimizer_start':'inherited_parent_adam','sample_exposures':cursor,
        'stop_reason':'epochs_complete' if cursor==len(order) else 'max_steps','history':history,
        'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-started})
    verify_files(ROOT,manifest)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    a=p.parse_args();train(a.output,a.resume)
