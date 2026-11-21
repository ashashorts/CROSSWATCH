"""Negative controls for first remediation: no real modeling without verified records."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Sources'))
from approved_real_run import AuthorizationError, require_rights, require_provenance
from run_fred_historical import main as historical_main
from crosswatch import read_fred_csv
from prepare_origin_manifest import main as origin_main


def test_unapproved_legacy_historical_runner_fails_before_output(tmp_path):
    output = tmp_path/'generated'
    status = historical_main([
        '--raw-dir', str(tmp_path/'missing'), '--out-dir', str(output),
        '--rights-file', str(tmp_path/'missing_permission.json'),
        '--origin-file', str(tmp_path/'missing_origin.json'),
    ])
    assert status == 2
    assert not output.exists(), 'Historical artifacts must not be created when permission is missing'


def test_researcher_review_cannot_use_empty_evidence(tmp_path):
    (tmp_path/'empty.txt').write_text('x')
    raw = dict(
        status='permission_confirmed', researcher_reviewed_and_signed=True,
        reviewer_name='Test Reviewer', review_date='2026-11-20',
        sources={symbol: {
            'document':'empty.txt', 'evidence_sha256':hashlib.sha256(b'x').hexdigest(),
            'permits_student_ml_training':True, 'data_provider':'Example Provider',
            'permitted_use_description':'Student machine learning, classroom charts, recording and demonstration.'
        } for symbol in ('SP500','VIXCLS','DGS10')}
    )
    file=tmp_path/'license_review.json'; file.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError, match='empty or insufficient'):
        require_rights(file,tmp_path)


def test_evidence_path_cannot_escape_project(tmp_path):
    record = dict(
        status='permission_confirmed', researcher_reviewed_and_signed=True,
        reviewer_name='Test Reviewer', review_date='2026-11-20',
        sources={symbol: {
            'document':'../../secrets.txt', 'evidence_sha256':'x'*64,
            'permits_student_ml_training':True, 'data_provider':'Example Provider',
            'permitted_use_description':'Student machine learning, classroom charts, recording and demonstration.'
        } for symbol in ('SP500','VIXCLS','DGS10')}
    )
    file=tmp_path/'license_review.json'; file.write_text(json.dumps(record))
    with pytest.raises(AuthorizationError, match='Missing external license'):
        require_rights(file,tmp_path)


def test_integrity_hash_on_permission_document_is_enforced(tmp_path):
    evidence=tmp_path/'actual_permission.txt'
    evidence.write_text('Example, nonbinding text only; this is NOT a real permission. ' * 3)
    status=dict(
        status='permission_confirmed', researcher_reviewed_and_signed=True,
        reviewer_name='Test Reviewer', review_date='2026-11-20',
        sources={symbol: {
            'document':'actual_permission.txt', 'evidence_sha256':'not_the_actual_hash',
            'permits_student_ml_training':True, 'data_provider':'Example Provider',
            'permitted_use_description':'Student machine learning, classroom charts, recording and demonstration.'
        } for symbol in ('SP500','VIXCLS','DGS10')}
    )
    file=tmp_path/'license_review.json'; file.write_text(json.dumps(status))
    with pytest.raises(AuthorizationError, match='evidence hash'):
        require_rights(file,tmp_path)


def test_provenance_generator_never_attests(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir()
    for s in ('SP500','VIXCLS','DGS10'):
        (raw/f'{s}.csv').write_text('DATE,'+s+'\n2026-11-18,1\n')
    output=tmp_path/'origin_manifest.json'
    assert origin_main(['--raw-dir',str(raw),'--output',str(output)]) == 0
    meta=json.loads(output.read_text())
    assert meta['status'].startswith('UNVERIFIED')
    assert not any(x['origin_verified_by_researcher'] for x in meta['sources'].values())
    with pytest.raises(AuthorizationError,match='no HTTPS official source'):
        require_provenance(output,tmp_path,raw)


def test_auto_download_is_disabled(tmp_path):
    with pytest.raises(RuntimeError,match='Automatic FRED acquisition is disabled'):
        read_fred_csv('SP500',tmp_path,allow_download=True)


def test_legacy_fred_acquisition_helpers_are_disabled(tmp_path):
    from fetch_and_run_real import get_fred_series, acquire, SourceAcquisitionError
    with pytest.raises(SourceAcquisitionError, match='disabled in Fix 1'):
        get_fred_series('SP500', '2020-01-01', '2020-12-31')
    with pytest.raises(SourceAcquisitionError, match='disabled in Fix 1'):
        acquire(tmp_path, '2020-01-01', '2020-12-31')
