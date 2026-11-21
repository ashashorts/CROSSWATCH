"""CROSSWATCH historical run: researcher-attested permission and source provenance gates.

Passing a gate does NOT independently validate ownership, a license, or source authenticity.
An authorized researcher/institution must examine the underlying provider evidence.
No machine learning can start from this entry point without both recorded gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from crosswatch import SERIES, read_fred_csv, join_market_levels
from crosswatch_phase3 import run_phase3_part1
from data_quality_gate import audit
from inspect_private_inputs import require_full_history_window


class AuthorizationError(RuntimeError):
    pass


def _json_record(path: Path, label: str) -> dict:
    if not path.is_file():
        raise AuthorizationError(f'Missing {label}: {path}')
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise AuthorizationError(f'Unreadable {label}: {path}') from exc
    if not isinstance(obj, dict):
        raise AuthorizationError(f'{label} must be a JSON object')
    return obj


def _safe_evidence_path(base: Path, relative: str, series: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise AuthorizationError(f'Missing external license/approval evidence for {series}: {relative}')
    if Path(relative).is_absolute():
        raise AuthorizationError(f'{series}: evidence must be a relative path inside the project')
    root = base.resolve()
    evidence = (root / relative).resolve()
    if not evidence.is_relative_to(root) or not evidence.is_file():
        raise AuthorizationError(f'Missing external license/approval evidence for {series}: {relative}')
    if evidence.stat().st_size < 30:
        raise AuthorizationError(f'{series}: evidence file is empty or insufficient to examine')
    return evidence


def require_rights(path: Path, base: Path) -> dict:
    """Structural check of a human-reviewed rights record; not a legal determination."""
    if not path.is_file():
        raise AuthorizationError('No permission record: provide license_review.json and supporting rights evidence first')
    status = _json_record(path, 'permission record')
    # Separate, expressly user-attested route when genuine approval correspondence
    # is held by the course instructor rather than stored on the student's laptop.
    # This does not claim independent review or generate a fictitious email file.
    if status.get('status') == 'user_attested_instructor_held':
        if status.get('attestation_mode') != 'instructor_held_user_statement':
            raise AuthorizationError('Unsupported instructor-held authorization declaration')
        if status.get('user_confirmed_approvals_on_file_with_instructor') is not True:
            raise AuthorizationError('Instructor-held authorization must be positively attested')
        if status.get('noncommercial_university_coursework_only') is not True:
            raise AuthorizationError('Noncommercial university-only scope must be attested')
        if status.get('no_public_raw_data_redistribution') is not True:
            raise AuthorizationError('Raw-data redistribution restriction must be acknowledged')
        if status.get('independent_document_verification') is not False:
            raise AuthorizationError('Never claim independent verification of emails not inspected')
        if status.get('attested_by') != 'course_project_user':
            raise AuthorizationError('Attestation must identify the declaring project user')
        try:
            date.fromisoformat(str(status['attested_date']))
        except (KeyError, ValueError, TypeError) as exc:
            raise AuthorizationError('Attestation date must be YYYY-MM-DD') from exc
        sources = status.get('sources')
        if not isinstance(sources, dict) or set(sources) != set(SERIES):
            raise AuthorizationError('Complete series-by-series attestation required')
        for symbol in SERIES:
            source = sources[symbol]
            if not isinstance(source, dict) or source.get('permits_student_ml_training') is not True:
                raise AuthorizationError(f'{symbol}: researcher did not attest approved academic ML use')
            provider = source.get('data_provider')
            scope = source.get('permitted_use_description')
            basis = source.get('confirmation_basis')
            if not isinstance(provider, str) or len(provider.strip()) < 3:
                raise AuthorizationError(f'{symbol}: actual provider must be documented')
            if not isinstance(scope, str) or len(scope.strip()) < 35:
                raise AuthorizationError(f'{symbol}: precise academic-only scope must be documented')
            if basis not in ('user_reported_provider_approval_held_by_instructor',
                             'public_domain_treasury_user_attested'):
                raise AuthorizationError(f'{symbol}: unknown attestation basis')
            if source.get('document') or source.get('evidence_sha256'):
                raise AuthorizationError(f'{symbol}: do not invent document paths or hashes')
        return status
    if status.get('status') != 'permission_confirmed':
        raise AuthorizationError('Permission is not confirmed. Do not run ML on restricted data.')
    if status.get('researcher_reviewed_and_signed') is not True:
        raise AuthorizationError('Record must be reviewed and signed by the responsible researcher')
    # First establish that actual supporting files exist (preserves the original gate semantics).
    for s in SERIES:
        source = status.get('sources', {}).get(s, {})
        if not isinstance(source, dict):
            raise AuthorizationError(f'{s}: malformed rights record')
        _safe_evidence_path(base, source.get('document'), s)
    reviewer = status.get('reviewer_name')
    reviewed = status.get('review_date')
    if not isinstance(reviewer, str) or len(reviewer.strip()) < 3:
        raise AuthorizationError('Missing actual reviewer name in rights record')
    try:
        date.fromisoformat(str(reviewed))
    except ValueError as exc:
        raise AuthorizationError('Review date must be YYYY-MM-DD') from exc
    sources = status.get('sources')
    if not isinstance(sources, dict):
        raise AuthorizationError('Missing per-series permission decisions')
    for s in SERIES:
        source = sources.get(s, {})
        if not isinstance(source, dict):
            raise AuthorizationError(f'{s}: malformed rights record')
        evidence_name = source.get('document')
        evidence = _safe_evidence_path(base, evidence_name, s)
        if not source.get('permits_student_ml_training'):
            raise AuthorizationError(f'{s} is not expressly cleared for this project ML use')
        use = source.get('permitted_use_description', '')
        provider = source.get('data_provider', '')
        if not isinstance(use, str) or len(use.strip()) < 30:
            raise AuthorizationError(f'{s}: missing explicit scope of ML, chart, demo and sharing permissions')
        if not isinstance(provider, str) or len(provider.strip()) < 3:
            raise AuthorizationError(f'{s}: missing actual granting data provider')
        expected_hash = source.get('evidence_sha256', '')
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        if not isinstance(expected_hash, str) or digest != expected_hash.lower():
            raise AuthorizationError(f'{s}: license evidence hash missing or does not match')
    return status


def require_provenance(path: Path, base: Path, raw: Path) -> dict:
    """Verify that attested source file hashes still match; source authenticity is manual."""
    record = _json_record(path, 'provenance record')
    sources = record.get('sources', {})
    if not isinstance(sources, dict):
        raise AuthorizationError('Malformed provenance manifest')
    for s in SERIES:
        item = sources.get(s, {})
        if not isinstance(item, dict):
            raise AuthorizationError(f'{s}: missing provenance metadata')
        official_url = item.get('official_source_url', '')
        if not isinstance(official_url, str) or not official_url.startswith('https://'):
            raise AuthorizationError(f'{s}: no HTTPS official source recorded')
        if item.get('origin_verified_by_researcher') is not True:
            raise AuthorizationError(f'{s}: original file origin not attested')
        src = raw / f'{s}.csv'
        actual = hashlib.sha256(src.read_bytes()).hexdigest() if src.is_file() else None
        if actual != item.get('sha256'):
            raise AuthorizationError(f'{s}: hash mismatch; CSV changed or wrong file')
        if not isinstance(item.get('acquisition_method'), str) or not item['acquisition_method'].strip():
            raise AuthorizationError(f'{s}: original acquisition method not recorded')
    return record


def run(raw_dir: Path, out_dir: Path, rights_file: Path, origin_file: Path, start: str, end: str):
    """Backward-compatible entry point for the ONE full authorized research pipeline."""
    # Delayed import avoids a circular import at module initialization.
    # Retain the original public entry-point prechecks, including direct-call behavior.
    root = Path(__file__).resolve().parent.parent
    require_rights(Path(rights_file), root)
    require_provenance(Path(origin_file), root, Path(raw_dir))
    require_full_history_window(raw_dir, start, end)
    from run_fred_historical import run_authorized
    return run_authorized(raw_dir, out_dir, rights_file, origin_file, start, end)


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', type=Path, default=Path('private-inputs'))
    p.add_argument('--out-dir', type=Path, default=Path('private-results'))
    p.add_argument('--rights-file', type=Path, default=root/'docs'/'license_review.json')
    p.add_argument('--origin-file', type=Path, default=root/'docs'/'origin_manifest.json')
    p.add_argument('--start', default='2017-01-03')
    p.add_argument('--end', default='2026-11-18')
    a = p.parse_args(argv)
    try:
        run(a.raw_dir, a.out_dir, a.rights_file, a.origin_file, a.start, a.end)
    except Exception as exc:
        print('RESEARCHER-ATTESTED MODEL RUN BLOCKED:', type(exc).__name__, str(exc))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
