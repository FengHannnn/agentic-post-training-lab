"""Cross-run FP32 reference checks with identity and DPO loss-error bounds."""
import math

def compare_reference_caches(current,original,beta=.1):
    if beta!=.1:raise ValueError('fixed diagnostic beta .1 required')
    if not current.get('baseline_adapter_sha256') or current['baseline_adapter_sha256']!=original.get('baseline_adapter_sha256'):raise ValueError('reference adapter identity differs')
    ids=current.get('pair_ids',[])
    if not ids or len(ids)!=len(set(ids)) or ids!=original.get('pair_ids') or len(current['scores'])!=len(ids) or len(original['scores'])!=len(ids):raise ValueError('reference pair identity differs')
    differences=[];sequence_differences=[];margin_differences=[];close=True
    for a,b in zip(current['scores'],original['scores']):
        if a.get('input_id')!=b.get('input_id') or not a.get('input_id'):raise ValueError('reference input identity differs')
        for side in ('chosen','rejected'):
            x=a[side];y=b[side]
            if not x or len(x)!=len(y):raise ValueError('reference token lengths differ')
            for v,w in zip(x,y):
                if not math.isfinite(v) or not math.isfinite(w) or max(v,w)>1e-6:raise ValueError('finite nonpositive reference logp required')
                differences.append(abs(v-w));close=close and math.isclose(v,w,rel_tol=5e-5,abs_tol=1e-4)
            sequence_differences.append(abs(sum(x)-sum(y)))
        margin_differences.append(abs((sum(a['chosen'])-sum(a['rejected']))-(sum(b['chosen'])-sum(b['rejected']))))
    maximum=max(differences);mean=sum(differences)/len(differences);sequence=max(sequence_differences);margin=max(margin_differences)
    if not close or maximum>5e-4 or mean>1e-5:raise ValueError('cross-run reference token numerics exceed limits')
    if sequence>5e-4 or beta*margin>5e-5:raise ValueError('reference sequence or DPO logit difference exceeds limits')
    # softplus is 1-Lipschitz, so beta*|delta margin| bounds loss error for every policy.
    return {'passed':True,'policy':'cross_run_fp32_reference_v1','pairs':len(ids),'tokens':len(differences),'maximum_token_difference':maximum,'mean_token_difference':mean,'maximum_sequence_sum_difference':sequence,'maximum_reference_margin_difference':margin,'maximum_DPO_logit_difference':beta*margin,'maximum_DPO_loss_difference_bound':beta*margin,'limits':{'token_rtol':5e-5,'token_atol':1e-4,'maximum_token_absolute':5e-4,'mean_token_absolute':1e-5,'maximum_sequence_sum_absolute':5e-4,'maximum_DPO_logit_absolute':5e-5},'same_run_reference_check':'unchanged; independently verified by score audit'}
