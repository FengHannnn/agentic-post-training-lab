"""Observe native sampler inputs/outputs in the same forward, without changing RNG or logits."""
import copy
import functools
import hashlib
import math

METHOD='same_forward_sampler_v2'
GATE=1e-5
LIMIT=16


def sampling_contract(s):
    keys=('temperature','top_p','top_k','min_p','min_tokens','repetition_penalty','presence_penalty','frequency_penalty')
    if tuple(s[k] for k in keys)!=(1,1,-1,0,0,1,0,0) or s['logprobs_mode']!='processed_logprobs':
        raise ValueError('same-forward probe requires neutral temperature-one sampling')


def install_capture(worker,setup):
    import torch
    sampling_contract(setup['sampling'])
    if type(setup['seed']) is not int or setup['token_limit']!=LIMIT:
        raise ValueError('fixed first-request capture identity required')
    if hasattr(worker,'_grpo_sampler_capture'):
        if worker._grpo_sampler_capture['setup']!=setup:raise ValueError('capture setup already differs')
        return worker._grpo_sampler_capture['installation']
    sampler=worker.model_runner.sampler
    if sampler.logprobs_mode!='processed_logprobs' or not hasattr(sampler,'topk_topp_sampler'):
        raise ValueError('vLLM 0.11 processed-logprobs sampler interface required')
    installation={'installed':True,'method':METHOD,'token_limit':LIMIT,
                  'sampler_class':type(sampler).__module__+'.'+type(sampler).__name__}
    state={'method':METHOD,'setup':copy.deepcopy(setup),'installation':installation,
           'request':None,'rows':[],'closed':False}
    original=sampler.forward
    @functools.wraps(original)
    def observed(*args,**kwargs):
        if state['closed'] or len(state['rows'])>=LIMIT:return original(*args,**kwargs)
        logits=kwargs.get('logits',args[0] if args else None)
        metadata=kwargs.get('sampling_metadata',args[1] if len(args)>1 else None)
        runner=worker.model_runner;ids=list(runner.input_batch.req_ids)
        if len(ids)!=1 or logits is None or logits.ndim!=2 or logits.shape[0]!=1:
            raise ValueError('single native sampler request required')
        rid=ids[0]
        if state['request'] is not None and rid!=state['request']['id']:
            state['closed']=True;return original(*args,**kwargs)
        request=runner.requests[rid];params=request.sampling_params
        if state['request'] is None:
            if params.seed!=setup['seed'] or request.lora_request is None or request.lora_request.lora_int_id!=1:
                raise ValueError('first sampler request seed/LoRA differs')
            state['request']={'id':str(rid),'seed':params.seed,'lora_id':1,
                              'prompt_token_ids':list(request.prompt_token_ids)}
        temp=metadata.temperature
        neutral=(not metadata.all_greedy and metadata.all_random and temp is not None
                 and temp.numel()==1 and float(temp[0].item())==1.
                 and metadata.top_k is None and metadata.top_p is None and metadata.no_penalties
                 and metadata.allowed_token_ids_mask is None and not metadata.bad_words_token_ids
                 and metadata.max_num_logprobs==1)
        if not neutral:raise ValueError('active sampler metadata differs from untruncated T1 contract')
        raw=logits.detach().float().clone()
        processed=[]
        def observe_processed(module,hook_args,hook_kwargs):
            values=hook_kwargs.get('logits',hook_args[0] if hook_args else None)
            k=hook_kwargs.get('k',hook_args[2] if len(hook_args)>2 else None)
            p=hook_kwargs.get('p',hook_args[3] if len(hook_args)>3 else None)
            processed.append(values is not None and values.shape==raw.shape and torch.equal(values,raw) and k is None and p is None)
        handle=sampler.topk_topp_sampler.register_forward_pre_hook(observe_processed,with_kwargs=True)
        try:out=original(*args,**kwargs)
        finally:handle.remove()
        selected=out.sampled_token_ids
        lp=out.logprobs_tensors
        if selected.shape!=(1,1) or lp is None or lp.logprobs.shape[0]!=1:
            raise ValueError('native sampled-token probability layout differs')
        token=int(selected[0,0].item())
        if int(lp.logprob_token_ids[0,0].item())!=token:
            raise ValueError('sampled token must occupy native logprob column zero')
        raw_logit=float(raw[0,token].item())
        normalizer=float(torch.logsumexp(raw.double(),dim=-1)[0].item())
        fp32_logprob=float(raw.log_softmax(dim=-1,dtype=torch.float32)[0,token].item())
        native_q=float(lp.logprobs[0,0].item())
        row={'step':len(state['rows']),'token_id':token,'raw_selected_logit':raw_logit,
             'raw_logsumexp_fp64':normalizer,'raw_logprob_fp64':raw_logit-normalizer,
             'raw_logprob':fp32_logprob,'native_processed_logprob':native_q,
             'normalization_error':abs(fp32_logprob-native_q),
             'fp64_precision_difference':abs(raw_logit-normalizer-fp32_logprob),
             'full_logits_unchanged':processed==[True],'metadata_neutral':True,
             'raw_logits_sha256':hashlib.sha256(raw.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
             'vocab_size':raw.shape[1]}
        state['rows'].append(row)
        return out
    sampler.forward=observed
    worker._grpo_sampler_capture=state
    return installation


def capture_snapshot(worker):
    state=getattr(worker,'_grpo_sampler_capture',None)
    return copy.deepcopy(state) if state is not None else None


def verify_capture(capture,generation,sampling,seed,policy_sha):
    from apt_v2.grpo import digest
    if not isinstance(generation,dict):
        from dataclasses import asdict
        generation=asdict(generation)
    sampling_contract(sampling)
    count=min(LIMIT,len(generation['response_token_ids']))
    if (capture is None or capture['method']!=METHOD or capture['setup']!={
            'seed':seed,'policy_sha256':policy_sha,'token_limit':LIMIT,'sampling':sampling}
            or capture['request'] is None or capture['request']['seed']!=seed
            or capture['request']['lora_id']!=1 or capture['request']['prompt_token_ids']!=generation['prompt_token_ids']
            or not capture['installation']['installed'] or capture['installation']['method']!=METHOD
            or not count<=len(capture['rows'])<=LIMIT):
        raise ValueError('same-forward sampler capture identity differs')
    errors=[]
    for i,row in enumerate(capture['rows'][:count]):
        q=generation['behavior_token_log_probs'][i]
        values=[row[k] for k in ('raw_selected_logit','raw_logsumexp_fp64','raw_logprob_fp64','raw_logprob',
                                'native_processed_logprob','normalization_error','fp64_precision_difference')]
        if (row['step']!=i or row['token_id']!=generation['response_token_ids'][i]
                or not row['full_logits_unchanged'] or not row['metadata_neutral']
                or any(not math.isfinite(v) for v in values) or not math.isfinite(q)
                or row['native_processed_logprob']>1e-6 or row['raw_logprob']>1e-6
                or len(row['raw_logits_sha256'])!=64 or not 0<=row['token_id']<row['vocab_size']):
            raise ValueError('same-forward sampler token/metadata identity differs')
        fp64=row['raw_selected_logit']-row['raw_logsumexp_fp64'];raw=row['raw_logprob']
        if (fp64!=row['raw_logprob_fp64'] or abs(raw-row['native_processed_logprob'])!=row['normalization_error']
                or abs(fp64-raw)!=row['fp64_precision_difference']):
            raise ValueError('same-forward normalization recomputation differs')
        errors.extend((abs(raw-q),abs(row['native_processed_logprob']-q),row['normalization_error']))
    error=max(errors)
    if error>GATE:raise ValueError('same-forward native q normalization gate failed: '+str(error))
    return {'passed':True,'tokens_verified':count,'maximum_error':error,'gate':GATE,
            'generation_sha256':digest(generation),'capture_sha256':digest(capture)}
