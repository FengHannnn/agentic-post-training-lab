"""Allow cross-job GPU replacement; pin hardware and precision inside a paired job."""
import json
PRECISION_KEYS={'float32_matmul_precision','allow_tf32_matmul','allow_tf32_cudnn'}

def runtime_report(cfg,actual,expected=None,precision=True):
    cross_job=expected is None;expected=cfg['required_execution_environment'] if cross_job else expected
    allowed={};rejected={}
    for key in sorted(set(expected)|set(actual)):
        if key not in expected or key not in actual:
            rejected[key]={'expected':expected.get(key),'actual':actual.get(key),'reason':'runtime field missing or added'};continue
        if actual[key]==expected[key]:continue
        row={'expected':expected[key],'actual':actual[key]}
        uuid_ok=key=='gpu_uuid' and cross_job and all(isinstance(v,str) and v not in ('','None') for v in (expected[key],actual[key]))
        if uuid_ok or (not precision and key in PRECISION_KEYS):allowed[key]=row
        else:rejected[key]=row
    if actual.get('gpu_name')!=cfg['required_gpu_name']:rejected['gpu_name']={'expected':cfg['required_gpu_name'],'actual':actual.get('gpu_name')}
    return {'passed':not rejected,'scope':'cross_job_with_fresh_paired_control' if cross_job else 'within_job_exact_identity','precision_checked':precision,'allowed_differences':allowed,'rejected_differences':rejected}

def check_job_runtime(cfg,actual,expected=None,precision=True):
    report=runtime_report(cfg,actual,expected,precision)
    if not report['passed']:raise ValueError('paired runtime identity differs: '+json.dumps(report['rejected_differences'],sort_keys=True))
    return report
