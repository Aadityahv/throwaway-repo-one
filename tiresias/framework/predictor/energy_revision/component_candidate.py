"""Uncalibrated class-aware component proposal. No default/fallback coefficients."""
import math

FEATURES=('base_time','fp32_work','other_non_global','special_function','lookup_bytes','dram_bytes')


def activity(work, logical_bytes, tier):
    if work.get('all_counts_exact') is not True:
        raise ValueError('REFUSED: proposed class-aware model requires exact static counts')
    if tier not in ('L2','DRAM'):
        raise ValueError('REFUSED: no L1 calibration or verified L1 residency')
    f={k:v['lane_instructions'] for k,v in work['families'].items()}
    if any(not math.isfinite(v) or v<0 for v in f.values()):
        raise ValueError('invalid static count')
    total=work['total_lane_instructions']
    if not math.isclose(sum(f.values()),total,rel_tol=1e-12,abs_tol=1e-6):
        raise ValueError('incomplete family census')
    ordinary=sum(f.get(k,0) for k in ('fp32_add','fp32_mul','fp32_fma'))
    sfu=f.get('special_function',0)
    other=total-f.get('global_load',0)-f.get('global_store',0)-ordinary-sfu
    # FMA has two ordinary arithmetic operations, but one executed instruction.
    # This is a proxy convention, not a claim that its energy equals two adds.
    arithmetic=ordinary+f.get('fp32_fma',0)
    if not math.isfinite(logical_bytes) or logical_bytes<0 or other<0:
        raise ValueError('invalid workload activity')
    return dict(fp32_work=arithmetic,other_non_global=other,special_function=sfu,
                lookup_bytes=logical_bytes,dram_bytes=logical_bytes if tier=='DRAM' else 0)


def predict(work, logical_bytes, tier, runtime_s, profile):
    if profile is None or profile.get('status')!='calibrated_and_frozen':
        raise ValueError('REFUSED: proposal has no admitted calibration; never reuse proxy coefficients')
    if profile.get('coefficient_unit_system')!='W_and_J_per_activity':
        raise ValueError('REFUSED: calibrated coefficient units must be explicit')
    if not math.isfinite(runtime_s) or runtime_s<=0:
        raise ValueError('invalid external runtime')
    if set(profile['coefficients'])!=set(FEATURES):
        raise ValueError('incomplete calibrated basis')
    if any(not math.isfinite(v) or v<0 for v in profile['coefficients'].values()):
        raise ValueError('nonnegative finite coefficients required')
    v=activity(work,logical_bytes,tier);v['base_time']=runtime_s
    e=sum(profile['coefficients'][k]*v[k] for k in FEATURES)
    cap=profile['cap_w']
    if not math.isfinite(cap) or cap<=0:raise ValueError('missing verified cap')
    return dict(energy_j=min(e,cap*runtime_s),uncapped_j=e,capped=e>cap*runtime_s,
                components_j={k:profile['coefficients'][k]*v[k] for k in FEATURES})
