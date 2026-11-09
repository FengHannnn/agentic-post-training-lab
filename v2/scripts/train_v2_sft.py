"""Bounded FP32 LoRA SFT pilot with verified data, suffix loss and restart state."""
import sys,json,random,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.sft import build_dataset,encode_example,action_loss

def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def train(output,resume=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model,PeftModel
    cfg=read(output/'config.json');dest=ROOT/'.cache/qwen2.5-1.5b-instruct'
    identity=read(dest/'model_identity.json');verify_files(dest,identity['files_sha256'])
    if identity['resolved_revision']!=cfg['revision']:raise ValueError('model revision differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    episodes,examples=build_dataset()
    tokenizer=AutoTokenizer.from_pretrained(dest,local_files_only=True)
    samples=[encode_example(tokenizer,e,cfg['max_length']) for e in examples]
    dataset_bytes=json.dumps(examples,ensure_ascii=False,sort_keys=True).encode()
    dataset_hash=hashlib.sha256(dataset_bytes).hexdigest()
    if not resume:
        jsonl(output/'demonstrations.jsonl',episodes);jsonl(output/'examples.jsonl',examples)
        write(output/'data_summary.json',{'episodes':len(episodes),'examples':len(examples),
            'successes':sum(e['success'] for e in episodes),'recovered':sum(e['recovered'] for e in episodes),
            'train_tasks':len(set(e['task_id'] for e in episodes)),
            'variants':['expert','sql_repair','evidence_repair'],'dataset_sha256':dataset_hash,
            'max_tokens':max(len(s['input_ids']) for s in samples),
            'supervised_tokens':sum(s['response_length'] for s in samples),
            'examples_sha256':file_hash(output/'examples.jsonl')})
        write(output/'model_identity.json',identity)
    base=AutoModelForCausalLM.from_pretrained(dest,local_files_only=True,torch_dtype=torch.float32,
        attn_implementation='sdpa').cuda()
    lora=cfg['lora']
    model=PeftModel.from_pretrained(base,resume/'adapter',is_trainable=True) if resume else get_peft_model(base,
        LoraConfig(r=lora['r'],lora_alpha=lora['alpha'],lora_dropout=lora['dropout'],
            target_modules=lora['target_modules'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    parameters=[p for n,p in model.named_parameters() if p.requires_grad]
    if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('LoRA only')
    initial_adapter=[p.detach().cpu().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=0)
    order=[]
    for epoch in range(cfg['epochs']):
        indices=list(range(len(samples)));random.Random(cfg['seed']+epoch).shuffle(indices);order+=indices
    cursor=steps=0;history=[]
    binding={'dataset_sha256':dataset_hash,'config_sha256':file_hash(output/'config.json'),
        'source_sha256':read(output/'run_manifest.json')['source_sha256']}
    if resume:
        if read(resume/'binding.json')!=binding:raise ValueError('resume input binding differs')
        complete=read(resume/'complete.json')
        verify_files(resume/'adapter',complete['adapter_sha256'])
        if file_hash(resume/'state.pt')!=complete['state_sha256']:raise ValueError('resume state identity differs')
        state=torch.load(resume/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);cursor=state['cursor'];steps=state['steps'];history=state['history']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    def save(name):
        checkpoint=output/name;checkpoint.mkdir()
        model.save_pretrained(checkpoint/'adapter',safe_serialization=True)
        write(checkpoint/'binding.json',binding)
        torch.save({'optimizer':optimizer.state_dict(),'cursor':cursor,'steps':steps,'history':history,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint/'state.pt')
        write(checkpoint/'complete.json',{'steps':steps,'cursor':cursor,'adapter_sha256':{
            p.name:file_hash(p) for p in (checkpoint/'adapter').iterdir() if p.is_file()},
            'state_sha256':file_hash(checkpoint/'state.pt')})
        return checkpoint
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    while cursor<len(order) and steps<cfg['max_steps']:
        batch=order[cursor:cursor+cfg['gradient_accumulation']]
        optimizer.zero_grad(set_to_none=True);loss_value=0.
        for index in batch:
            loss=action_loss(model,samples[index])/len(batch)
            if not torch.isfinite(loss):raise ValueError('nonfinite SFT loss')
            loss.backward();loss_value+=loss.item()
        norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'],error_if_nonfinite=True)
        if norm.item()<=0:raise ValueError('nonzero adapter gradient required')
        optimizer.step();cursor+=len(batch);steps+=1
        row={'step':steps,'cursor':cursor,'loss':loss_value,'grad_norm':norm.item(),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        history.append(row);print(json.dumps(row),flush=True)
        if steps%cfg['checkpoint_every']==0:save(f'checkpoint-{steps:04}')
    final=save('final')
    write(output/'training_summary.json',{'optimizer_updates':steps,'sample_exposures':cursor,
        'stop_reason':'max_steps' if steps>=cfg['max_steps'] else 'epochs_complete',
        'trainable_parameters':sum(p.numel() for p in parameters),'history':history,
        'adapter_tensors_changed':sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial_adapter,parameters)),
        'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'elapsed_seconds':time.perf_counter()-started,'final_checkpoint':str(final),'binding':binding})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    a=p.parse_args();train(a.output,a.resume)
