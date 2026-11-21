"""User-reported instructor-held permission route, never synthetic email fabrication."""
from pathlib import Path
import sys,json,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Sources'))
from approved_real_run import require_rights,AuthorizationError

def record():
    return {'status':'user_attested_instructor_held',
      'attestation_mode':'instructor_held_user_statement',
      'attested_by':'course_project_user','attested_date':'2026-11-20',
      'user_confirmed_approvals_on_file_with_instructor':True,
      'noncommercial_university_coursework_only':True,
      'no_public_raw_data_redistribution':True,
      'independent_document_verification':False,
      'sources':{symbol:{'data_provider':provider,
          'permits_student_ml_training':True,
          'confirmation_basis':('public_domain_treasury_user_attested' if symbol=='DGS10' else 'user_reported_provider_approval_held_by_instructor'),
          'permitted_use_description':'Offline noncommercial university machine learning, classroom report and non-public demonstration; no public raw data sharing.'}
        for symbol,provider in [('SP500','S&P Dow Jones Indices LLC'),('VIXCLS','Cboe Global Markets'),('DGS10','U.S. Treasury / Federal Reserve')]}}

def save(tmp_path,value):
    p=tmp_path/'license_review.json';p.write_text(json.dumps(value));return p

def test_instructor_held_positive_attestation_structurally_passes(tmp_path):
    r=record(); assert require_rights(save(tmp_path,r),tmp_path)==r

def test_instructor_held_missing_acknowledgement_blocks(tmp_path):
    r=record();r['user_confirmed_approvals_on_file_with_instructor']=False
    with pytest.raises(AuthorizationError,match='positively attested'):require_rights(save(tmp_path,r),tmp_path)

def test_instructor_held_rejects_fictitious_evidence_hash(tmp_path):
    r=record();r['sources']['SP500']['evidence_sha256']='fake'
    with pytest.raises(AuthorizationError,match='do not invent'):require_rights(save(tmp_path,r),tmp_path)

def test_instructor_held_forbids_public_raw_redistribution(tmp_path):
    r=record();r['no_public_raw_data_redistribution']=False
    with pytest.raises(AuthorizationError,match='redistribution'):require_rights(save(tmp_path,r),tmp_path)

def test_instructor_held_does_not_claim_document_verification(tmp_path):
    r=record();r['independent_document_verification']=True
    with pytest.raises(AuthorizationError,match='Never claim'):require_rights(save(tmp_path,r),tmp_path)


def test_run_identity_covers_rights_gate_and_plotting_code():
    from release_integrity import MODEL_CODE_FILES
    assert {'approved_real_run.py','inspect_private_inputs.py','run_experiment_stage.py'}.issubset(set(MODEL_CODE_FILES))
