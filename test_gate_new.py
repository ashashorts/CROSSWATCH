from pathlib import Path
import json,sys,hashlib
import pytest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Sources'))
from crosswatch import SERIES,generate_synthetic_levels,engineer_features
from data_quality_gate import audit,GateError
from approved_real_run import require_rights,require_provenance,AuthorizationError

def synth_csvs(root,n=1400):
    root.mkdir(parents=True,exist_ok=True)
    levels=generate_synthetic_levels(n=n,seed=812)
    for s in SERIES:
        pd.DataFrame({'DATE':levels.index.strftime('%Y-%m-%d'),s:levels[s].values}).to_csv(root/f'{s}.csv',index=False)
    return levels

def test_missing_one_source_refused(tmp_path):
    with pytest.raises(GateError,match='MISSING'):
        audit(tmp_path/'missing',tmp_path/'out',start='2024-01-02',end='2029-12-31')

def test_synthetic_qa_pass_does_not_claim_real(tmp_path):
    synth_csvs(tmp_path/'raw')
    rep=audit(tmp_path/'raw',tmp_path/'out','2024-01-02','2029-12-31')
    assert rep['data_quality_pass']
    assert rep['n_joined_level_days']>=1000
    assert rep['official_origin_verified'] is False
    assert rep['usage_rights_verified'] is False
    assert rep['analysis_authorized'] is False
    assert rep['split']['train']['n']>rep['split']['test']['n']
    assert 'SP500' not in (tmp_path/'out'/'raw_csv_quality_report.json').read_text().split('"raw_files"')[0]

def test_short_dataset_refused(tmp_path):
    synth_csvs(tmp_path/'raw',n=520)
    with pytest.raises(GateError,match='only.*valid rows'):
        audit(tmp_path/'raw',tmp_path/'out','2024-01-02','2029-12-31')

def test_no_fred_rights_file_refused(tmp_path):
    with pytest.raises(AuthorizationError,match='No permission record'):
        require_rights(tmp_path/'none.json',tmp_path)

def test_pending_permissions_block_all_real_modeling(tmp_path):
    p=tmp_path/'license.json'; p.write_text(json.dumps({'status':'pending_permission_review','researcher_reviewed_and_signed':False}))
    with pytest.raises(AuthorizationError,match='Permission is not confirmed'):
        require_rights(p,tmp_path)

def test_false_confirmation_without_supporting_documents_refused(tmp_path):
    p=tmp_path/'license.json'; p.write_text(json.dumps({'status':'permission_confirmed','researcher_reviewed_and_signed':True,'sources':{s:{'document':'missing.pdf','permits_student_ml_training':True} for s in SERIES}}))
    with pytest.raises(AuthorizationError,match='Missing external license'):
        require_rights(p,tmp_path)

def test_file_hash_tampering_refused(tmp_path):
    raw=tmp_path/'raw';synth_csvs(raw)
    mf={'sources':{s:{'official_source_url':'https://fred.stlouisfed.org/series/'+s,'sha256':'incorrect','origin_verified_by_researcher':True} for s in SERIES}}
    p=tmp_path/'orig.json';p.write_text(json.dumps(mf))
    with pytest.raises(AuthorizationError,match='hash mismatch'):
        require_provenance(p,tmp_path,raw)

def test_one_day_exact_feature_math():
    dates=pd.to_datetime(['2026-01-05','2026-01-06','2026-01-07'])
    frame=pd.DataFrame({'SP500':[100,98,100],'VIXCLS':[20,25,22],'DGS10':[4.50,4.47,4.52]},index=dates)
    feats=engineer_features(frame)
    assert feats.iloc[0]['sp_return_pct']==pytest.approx(-2)
    assert feats.iloc[0]['vix_delta_points']==pytest.approx(5)
    assert feats.iloc[0]['yield_delta_bp']==pytest.approx(-3)
    assert feats.iloc[1]['sp_return_pct']==pytest.approx(100*(100/98-1))

def test_parse_date_duplicate_refused(tmp_path):
    synth_csvs(tmp_path/'raw')
    p=tmp_path/'raw'/'SP500.csv'
    df=pd.read_csv(p)
    df.loc[1,'DATE']=df.loc[0,'DATE']
    df.to_csv(p,index=False)
    with pytest.raises(Exception,match='Duplicate|duplicate'):
        audit(tmp_path/'raw',tmp_path/'out','2024-01-02','2029-12-31')
