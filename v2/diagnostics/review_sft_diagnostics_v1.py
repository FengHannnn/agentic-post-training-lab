"""Offline review of the frozen v1 diagnostic return; original ZIP stays intact."""
from pathlib import Path
import json,sys,hashlib,zipfile,shutil,importlib.util,math
ROOT=Path(__file__).resolve().parents[1]

def review(archive,output):
    output.mkdir(parents=True,exist_ok=False)
    original=output/'original'
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        for i in z.infolist():
            if not (original/i.filename).resolve().is_relative_to(original.resolve()):raise ValueError('unsafe ZIP path')
        z.extractall(original)
    raw=original/'sft-diagnostics-v1-20261101'
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(raw/'run_manifest.json')['source_sha256']
    for name,value in manifest.items():
        if digest(raw/'source'/name)!=value:raise ValueError('source differs: '+name)
    binding=read(raw/'input_binding.json')
    for name,value in binding['model_files'].items():
        if (raw/'tokenizer'/name).exists() and digest(raw/'tokenizer'/name)!=value:raise ValueError('tokenizer differs')
    pilot_archive=ROOT/'artifacts/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030-review.zip'
    with zipfile.ZipFile(pilot_archive) as z:
        for name,value in binding['pilot_files'].items():
            if hashlib.sha256(z.read('sft-pilot-v1-20261030/'+name)).hexdigest()!=value:raise ValueError('pilot binding differs: '+name)
    probes=read(raw/'encoded_probes.json');rows=read(raw/'vllm-probes.json');n=len(probes)
    if len(rows)!=2*n or n!=8:raise ValueError('legacy probe count differs')
    # Frozen collector loops base then sft and appends every probe in the saved order.
    # Preserve the exact original scores/tokens; restore only branch metadata in a separate copy.
    source=(raw/'source/diagnostics/diagnose_sft_v1.py').read_text(encoding='utf8')
    if "for mode in ('base','sft'):" not in source or "'score':score,**asdict(sample)" not in source:
        raise ValueError('unexpected legacy collector; review ordering manually')
    restored=[]
    for i,row in enumerate(rows):
        p=probes[i%n]
        if row['mode']!='greedy' or row['probe_id']!=p['probe_id'] or row['prompt_token_ids']!=p['prompt_token_ids']:
            raise ValueError('legacy probe order differs')
        restored.append({**row,'mode':'base' if i<n else 'sft','sampling_mode':row['mode'],
            'branch_recovery':'frozen_collector_base_then_sft_order'})
    recovered=output/'recovered';shutil.copytree(raw,recovered)
    (recovered/'vllm-probes.json').write_text(json.dumps(restored,ensure_ascii=False,indent=2),encoding='utf8')
    for mode in ('base','sft'):
        (recovered/('vllm-'+mode+'-probes.json')).write_text(json.dumps([r for r in restored if r['mode']==mode],ensure_ascii=False,indent=2),encoding='utf8')
    spec=importlib.util.spec_from_file_location('frozen_diagnostic_worker',raw/'source/diagnostics/diagnose_sft_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Replace cloud-only input paths after verifying the frozen sources and local pilot binding above.
    module.context=lambda path:(read(path/'config.json'),None,None)
    module.audit(recovered)
    action_rows=[json.loads(l) for l in (raw/'hf-fp32-actions.jsonl').read_text(encoding='utf8').splitlines()]
    from apt_v2.sft_diagnostics import grouped_scores
    recomputed={m:grouped_scores([r for r in action_rows if r['mode']==m]) for m in ('base','sft')}
    if recomputed!=read(raw/'hf-fp32-score-summary.json'):raise ValueError('teacher-forced summary differs')
    for file in ('hf-fp32-probes.json','hf-bf16-probes.json','vllm-probes.json'):
        for row in read(recovered/file):
            p=next(p for p in probes if p['probe_id']==row['probe_id'])
            score=row['score'];gold=p['response_token_ids']
            if len(score['token_log_probs'])!=len(gold) or len(score['argmax_token_ids'])!=len(gold):raise ValueError('gold score alignment')
            if any(not math.isfinite(v) or v>1e-6 for v in score['token_log_probs']):raise ValueError('finite score required')
    comparison=read(recovered/'comparison.json')
    result={'zip_sha256':digest(archive),'source_files_verified':len(manifest),'pilot_inputs_verified':len(binding['pilot_files']),
        'tokenizer_files_verified':sum((raw/'tokenizer'/n).exists() for n in binding['model_files']),
        'replayed_episodes':comparison['replayed_episodes'],'teacher_forced_actions':len(action_rows),
        'recovered_probe_branches':len(restored),'branch_recovery':'frozen collector loop order; preserved original ZIP',
        'optimizer_updates':0,'offline_audit_passed':True}
    (output/'local_review_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))
    for name in ('train-base-summary.json','train-sft-summary.json'):
        print(name,json.dumps(read(raw/name)))
    for pair in [('hf_fp32','hf_bf16'),('hf_bf16','vllm')]:
        for mode in ('base','sft'):
            group=[r for r in comparison['comparisons'] if (r['left'],r['right'])==pair and r['mode']==mode]
            print('COMPARISON',pair,mode,'greedy_equal',sum(r['greedy_tokens_equal'] for r in group),'of',len(group),
                'max_logp_gap',max(r['max_abs_logp_difference'] for r in group),'argmax_gaps',sum(r['argmax_disagreements'] for r in group))
    for name,group in comparison['adapter_effects'].items():print('ADAPTER_EFFECT',name,sum(r['greedy_tokens_changed'] for r in group),'of',len(group))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.archive,a.output)
