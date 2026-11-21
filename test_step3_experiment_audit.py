import numpy as np
import pandas as pd
import pytest
from crosswatch import generate_synthetic_levels,engineer_features,chronological_split,fit_crosswatch,apply_crosswatch
from crosswatch_phase3 import fit_regime_calibration,apply_regime_calibration,future_five_day_decline_proxy,run_phase3_part1
from experiment_audit import (assert_strict_time_partitions,strict_gap_check,strict_features,
    event_episode_count,evaluate_operating_points,regime_diagnostics,
    paired_moving_block_bootstrap,circular_shift_placebo,
    evaluate_rule_reliability,audit_all)

@pytest.fixture(scope='module')
def fixture_result(tmp_path_factory):
    f=tmp_path_factory.mktemp('step3')
    return run_phase3_part1(generate_synthetic_levels(n=1400,seed=2026),f,
                            source_type='SYNTHETIC_QA_ONLY')

def test_time_partitions_are_strict(fixture_result):
    s=fixture_result['scored'];d=assert_strict_time_partitions(s['train'],s['validation'],s['test'])
    assert d['train']['rows']+d['validation']['rows']+d['test']['rows']==1399

def test_time_partitions_reject_same_date(fixture_result):
    s=fixture_result['scored']
    with pytest.raises(ValueError):
        assert_strict_time_partitions(s['train'],s['train'].tail(100),s['test'])

def test_strict_gap_audit_and_exclusion():
    ix=pd.to_datetime(['2024-01-02','2024-01-03','2024-01-10','2024-01-11','2024-01-12','2024-01-15','2024-01-16'])
    d=pd.DataFrame({'SP500':[100,101,102,103,104,105,106],
         'VIXCLS':[18,18.1,18.2,18.3,18.4,18.5,18.6],
         'DGS10':[4,4.1,4.2,4.3,4.4,4.5,4.6]},index=ix)
    diag=strict_gap_check(d,5)
    assert diag['gaps_exceeding_limit']==1 and diag['gap_dates']==['2024-01-10']
    assert pd.Timestamp('2024-01-10') not in strict_features(d,5).index

def test_strict_gap_rejects_unordered():
    dates=pd.to_datetime(['2024-01-03','2024-01-02'])
    with pytest.raises(ValueError):strict_gap_check(pd.DataFrame(index=dates))

def test_episode_count_groups_overlapping_days():
    assert event_episode_count(pd.Series([0,1,1,0,1,1,1,0,1],dtype=bool))==3

def test_operating_metrics_correct_on_known_example():
    f=pd.DataFrame({'forward_proxy':pd.array([True,False,True,False,True,False],dtype='boolean'),
                    'baseline_alert':[1,1,0,0,0,0],
                    'IFOREST_ALERT':[1,0,1,0,0,0],
                    'REGIME_ALERT':[0,0,1,1,0,0]})
    a=evaluate_operating_points(f)
    assert a.iloc[0]['alerts']==2 and a.iloc[0]['proxy_hits']==1
    assert np.isclose(a.iloc[1]['proxy_hit_fraction'],1)
    assert np.isclose(a.iloc[1]['proxy_hit_lift_vs_base'],2)

def test_absent_proxy_positives_returns_nan_in_lift():
    f=pd.DataFrame({'forward_proxy':[False]*10,**{k:[True]*10 for k in ('baseline_alert','IFOREST_ALERT','REGIME_ALERT')}})
    a=evaluate_operating_points(f)
    assert a['proxy_hit_lift_vs_base'].isna().all()

def test_regime_dominance_flag(fixture_result):
    x=regime_diagnostics(fixture_result['scored'])
    assert 'elevated_fraction' in x and len(x)==3
    assert x['elevated_fraction'].between(0,1).all()

def test_block_bootstrap_is_deterministic(fixture_result):
    f=fixture_result['scored']['test']
    a=paired_moving_block_bootstrap(f,n_boot=80,seed=9)
    b=paired_moving_block_bootstrap(f,n_boot=80,seed=9)
    assert a==b and a['replicates_valid']<=80

def test_block_bootstrap_bad_small_sample():
    f=pd.DataFrame({'forward_proxy':[True]*10,'IFOREST_ALERT':[True]*10,'REGIME_ALERT':[False]*10})
    with pytest.raises(ValueError):paired_moving_block_bootstrap(f,block_length=10,n_boot=50)

def test_placebo_deterministic(fixture_result):
    f=fixture_result['scored']['test']
    a=circular_shift_placebo(f,n_repeats=40,seed=25)
    b=circular_shift_placebo(f,n_repeats=40,seed=25)
    assert a==b and 0<=a['shifted_proxy_hit_fraction_median']<=1

def test_rule_reliability_has_min_count_flag():
    x=pd.DataFrame({'EQUITY_TAIL_DOWN':[1,0,1,0], 'IFOREST_ALERT':[1,0,0,0]})
    rules=pd.DataFrame([{'antecedent':'EQUITY_TAIL_DOWN','support':.1,'confidence':.2,'lift':1.2}])
    d=evaluate_rule_reliability(x,rules,min_antecedent_count=3,min_joint_count=2)
    assert d.iloc[0]['joint_count']==1 and not d.iloc[0]['minimum_support_met']

def test_future_poison_never_changes_prior_features():
    levels=generate_synthetic_levels(n=100)
    orig=engineer_features(levels)
    changed=levels.copy()
    changed.iloc[-1,changed.columns.get_loc('SP500')]*=4
    newer=engineer_features(changed)
    pd.testing.assert_frame_equal(orig.iloc[:-1],newer.iloc[:-1],check_exact=True)

def test_audit_all_exposes_synthetic_tag(fixture_result,tmp_path):
    out=audit_all(fixture_result,tmp_path,source_label='SYNTHETIC_QA_ONLY')
    assert not out['metadata']['running_on_real_data']
    assert (tmp_path/'SYNTHETIC_QA_ONLY_audit_manifest.json').exists()

def test_audit_all_rejects_unverified_source_label(fixture_result,tmp_path):
    with pytest.raises(ValueError):audit_all(fixture_result,tmp_path,source_label='JUST_REAL_TRUST_ME')

def test_alternating_regime_fixture_activates_both_holdout_branches(tmp_path):
    from stress_fixtures import alternating_volatility_fixture
    levels=alternating_volatility_fixture(n=1600,seed=8127)
    result=run_phase3_part1(levels,tmp_path,source_type='SYNTHETIC_QA_ONLY')
    d=regime_diagnostics(result['scored'])
    for part in ('validation','test'):
        v=d[d['split']==part].iloc[0]
        assert v['ordinary_vix_days']>0 and v['elevated_vix_days']>0


def test_proprietary_official_page_excerpt_not_distributed_in_shareable_project():
    from pathlib import Path
    root=Path(__file__).resolve().parent.parent
    assert not (root/'verification'/'official_page_spotcheck_NOT_full_csv.csv').exists()
