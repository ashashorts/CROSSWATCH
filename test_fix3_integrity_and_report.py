"""Fix 3: report number parity and immutable code/artifact-bound run identity."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Sources'))
from release_integrity import (read_manifest, MANIFEST_NAME, ReleaseIntegrityError,
                               fingerprint_run_id, model_code_hashes, sha256)
from run_integrated_synthetic_qa import main as generate
from audit_report_alignment import audit as audit_report, AlignmentError


@pytest.fixture(scope='module')
def integration(tmp_path_factory):
    p=tmp_path_factory.mktemp('fix3_run')
    results,raw=p/'out',p/'private_synthetic'
    assert generate(['--out-dir',str(results),'--raw-dir',str(raw)])==0
    return results,raw


def test_run_identity_binds_file_hashes_and_code(integration):
    results,raw=integration
    m=read_manifest(results,raw_dir=raw)
    assert m['schema_version']==3
    assert len(m['model_code_sha256'])>=10
    assert m['model_code_sha256']==model_code_hashes()
    assert m['run_id']==fingerprint_run_id(kind=m['run_kind'],label=m['source_label'],
        input_hashes=m['input_sha256'],start=m['research_window']['start'],
        end=m['research_window']['end'],variants=m['variants'],
        include_baselines=m['includes_statistical_baselines'],
        artifacts=m['artifacts'],code_hashes=m['model_code_sha256'])


def test_modified_run_identity_is_rejected(integration,tmp_path):
    results,raw=integration
    dest=tmp_path/'changed-id';shutil.copytree(results,dest)
    obj=json.loads((dest/MANIFEST_NAME).read_text());obj['run_id']='0'*20
    (dest/MANIFEST_NAME).write_text(json.dumps(obj))
    with pytest.raises(ReleaseIntegrityError,match='Run ID'):
        read_manifest(dest,raw_dir=raw)


def test_manifest_artifact_change_cannot_keep_old_identity(integration,tmp_path):
    results,raw=integration
    dest=tmp_path/'changed-reports';shutil.copytree(results,dest)
    relative='feature_comparison_budget.csv'
    file=dest/relative
    file.write_text(file.read_text()+'\n')
    obj=json.loads((dest/MANIFEST_NAME).read_text())
    obj['artifacts'][relative]={'sha256':sha256(file),'bytes':file.stat().st_size}
    (dest/MANIFEST_NAME).write_text(json.dumps(obj))
    with pytest.raises(ReleaseIntegrityError,match='Run ID'):
        read_manifest(dest,raw_dir=raw)


def test_tampered_code_hash_is_rejected(integration,tmp_path):
    results,raw=integration
    dest=tmp_path/'changed-code-inventory';shutil.copytree(results,dest)
    obj=json.loads((dest/MANIFEST_NAME).read_text())
    obj['model_code_sha256']['crosswatch.py']='0'*64
    (dest/MANIFEST_NAME).write_text(json.dumps(obj))
    with pytest.raises(ReleaseIntegrityError,match='Source-code version'):
        read_manifest(dest,raw_dir=raw)


def test_report_checker_requires_real_manifest_by_default(tmp_path):
    with pytest.raises(AlignmentError,match='No released manifest'):
        audit_report(ROOT/'Results',ROOT/'CROSSWATCH_IEEE_Style_Report.docx')


def test_report_checker_detects_wrong_research_values(integration):
    results,raw=integration
    status=audit_report(results,ROOT/'CROSSWATCH_IEEE_Style_Report.docx',raw_dir=raw)
    assert status['artifact_manifest_verified'] is True
    assert status['numerical_checks_passed'] is False
    assert len(status['failed_checks'])>=3
    assert status['final_submission_ready'] is False
    assert 'Team number and author names' in status['user_report_completion_pending']
    assert 'Verified actual video URL absent' in status['user_report_completion_pending']
