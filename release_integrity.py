"""Immutable output contract shared by the runner, case exporter, notebook and UI.

No raw licensed observations or permission-document bytes are embedded in this
manifest. Hashes validate byte identity, not legal source authenticity. Callers
must apply the independent authorization gate before processing real market data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from crosswatch import SERIES

SCHEMA_VERSION = 3
MANIFEST_NAME = 'crosswatch_run_manifest.json'
ALLOWED_KINDS = {
    'historical_research': 'RIGHTS_REVIEWED_MARKET_DATA',
    'synthetic_integration_qa': 'SYNTHETIC_INTEGRATION_ONLY',
}
SPLITS = ('train', 'validation', 'test')
VARIANTS = ('three_features', 'five_features')
MODEL_CODE_FILES = ('crosswatch.py', 'crosswatch_phase3.py', 'zscore_baseline.py',
                    'association_evaluation.py', 'episode_evaluation.py',
                    'experiment_audit.py', 'data_quality_gate.py', 'run_fred_historical.py',
                    'approved_real_run.py', 'inspect_private_inputs.py', 'run_experiment_stage.py',
                    'historical_cases.py', 'export_historical_cases.py', 'release_integrity.py')
REQUIRED_ROOT = ('raw_csv_quality_report.json', 'feature_comparison_operating.csv',
                 'feature_comparison_budget.csv', 'episode_coverage.csv',
                 'pairwise_alert_overlap.csv')
REQUIRED_VARIANT = (
    'metadata.json', 'association_rules.csv', 'association_verification.json',
    'operating_metrics.csv', 'budget_matched_metrics.csv',
    'operating_proxy_diagnostics.csv', 'training_rule_holdout_checks.csv',
    'audit_manifest.json', 'test_derived_results.csv',
    'validation_derived_results.csv', 'train_derived_results.csv',
)

class ReleaseIntegrityError(ValueError):
    """A released run cannot be safely trusted/read by its consumers."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ReleaseIntegrityError('Artifact path must be a nonempty relative path')
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ReleaseIntegrityError('Artifact path escapes the results folder')
    return candidate


def model_code_hashes() -> dict[str, str]:
    """Fingerprint the exact local implementation files used to generate scores."""
    source_dir = Path(__file__).resolve().parent
    return {name: sha256(source_dir / name) for name in MODEL_CODE_FILES}


def fingerprint_run_id(*, kind: str, label: str, input_hashes: dict,
                       start: str, end: str, variants: dict,
                       include_baselines: bool, artifacts: dict,
                       code_hashes: dict) -> str:
    """Bind a run ID to the complete, immutable result inventory and model code.

    This is not a cryptographic signature: a malicious actor able to rewrite the
    trusted manifest can recompute hashes. It detects accidental mixed outputs,
    changed execution code, and falsely reused IDs.
    """
    d = {
        'schema_version': SCHEMA_VERSION, 'kind': kind, 'label': label,
        'input_sha256': input_hashes, 'start': start, 'end': end,
        'variants': variants, 'include_baselines': include_baselines,
        'artifact_fingerprints': artifacts, 'model_code_sha256': code_hashes,
    }
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def create_manifest(folder: Path, *, kind: str, input_hashes: dict, start: str,
                    end: str, include_baselines: bool, permission_status: str) -> dict:
    """Fingerprint a complete two-variant output directory, but NOT raw CSVs."""
    folder = Path(folder)
    if kind not in ALLOWED_KINDS:
        raise ReleaseIntegrityError(f'Unsupported run kind: {kind}')
    if set(input_hashes) != set(SERIES) or any(
        not isinstance(input_hashes[k], str) or len(input_hashes[k]) != 64
        or any(c not in '0123456789abcdef' for c in input_hashes[k].lower())
        for k in SERIES
    ):
        raise ReleaseIntegrityError('Provide three full input SHA-256 values')
    label = ALLOWED_KINDS[kind]
    for f in REQUIRED_ROOT:
        if not (folder / f).is_file():
            raise ReleaseIntegrityError(f'Missing required root artifact: {f}')
    variant_details = {}
    for variant in VARIANTS:
        target = folder / variant
        for ending in REQUIRED_VARIANT:
            actual = target / (f'{label}_{ending}' if ending not in ('association_rules.csv','association_verification.json') else ending)
            if not actual.is_file():
                raise ReleaseIntegrityError(f'Missing required {variant} artifact: {actual.name}')
        meta = json.loads((target / f'{label}_metadata.json').read_text())
        if meta.get('source_type') != label:
            raise ReleaseIntegrityError(f'{variant}: source label differs from expected {label}')
        if meta.get('extra_provenance', {}).get('sha256') != input_hashes:
            raise ReleaseIntegrityError(f'{variant}: embedded input hashes differ')
        variant_details[variant] = {
            'directory': variant,
            'features': meta['features'],
            'splits': meta['splits'],
            'model_metadata': f'{variant}/{label}_metadata.json',
            'association_rules': f'{variant}/association_rules.csv',
            'association_audit': f'{variant}/association_verification.json',
            'available_baselines': 'zscore_baselines' in meta,
        }
    artifacts = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():
            raise ReleaseIntegrityError('Results must not contain symbolic links')
        if p.is_file() and p.name != MANIFEST_NAME:
            relative = p.relative_to(folder).as_posix()
            artifacts[relative] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    code_hashes = model_code_hashes()
    run_id = fingerprint_run_id(
        kind=kind, label=label, input_hashes=input_hashes, start=start, end=end,
        variants=variant_details, include_baselines=include_baselines,
        artifacts=artifacts, code_hashes=code_hashes,
    )
    return {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'model_code_sha256': code_hashes,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_kind': kind,
        'source_label': label,
        'research_window': {'start': start, 'end': end},
        'permission_status': permission_status,
        'input_sha256': input_hashes,
        'variants': variant_details,
        'available_splits': list(SPLITS),
        'includes_statistical_baselines': bool(include_baselines),
        'root_artifacts': list(REQUIRED_ROOT),
        'artifacts': artifacts,
        'data_policy': 'Raw licensed CSVs and private authorization documents remain local; never include them in public packages.',
        'interpretation_limits': 'Retrospective anomaly study, not forecasting or investment advice; five-session loss is an evaluation proxy.',
    }


def write_manifest(folder: Path, **kwargs) -> dict:
    m = create_manifest(folder, **kwargs)
    path = Path(folder) / MANIFEST_NAME
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding='utf8')
    tmp.replace(path)
    return m


def read_manifest(folder: Path, *, verify_artifacts: bool=True,
                  raw_dir: Path|None=None, require_kind: str|None=None) -> dict:
    folder = Path(folder)
    path = folder / MANIFEST_NAME
    if not path.is_file():
        raise ReleaseIntegrityError('Run manifest missing. Regenerate via the Fix 2 runner; archived Results/ are not a live dashboard run.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ReleaseIntegrityError('Unsupported run manifest schema')
    kind = manifest.get('run_kind')
    if kind not in ALLOWED_KINDS or manifest.get('source_label') != ALLOWED_KINDS[kind]:
        raise ReleaseIntegrityError('Invalid run kind / source label pairing')
    if require_kind and kind != require_kind:
        raise ReleaseIntegrityError(f'This consumer requires {require_kind}, not {kind}')
    if set(manifest.get('variants', {})) != set(VARIANTS):
        raise ReleaseIntegrityError('A complete two-feature-set run is required')
    if manifest.get('available_splits') != list(SPLITS):
        raise ReleaseIntegrityError('Invalid partition list')
    code_hashes = manifest.get('model_code_sha256')
    if set(code_hashes or {}) != set(MODEL_CODE_FILES):
        raise ReleaseIntegrityError('Model source-code fingerprint inventory missing or incomplete')
    if code_hashes != model_code_hashes():
        raise ReleaseIntegrityError('Source-code version differs from the exact run that generated these outputs')
    if raw_dir is not None:
        hashes = manifest.get('input_sha256', {})
        if set(hashes) != set(SERIES):
            raise ReleaseIntegrityError('Input-source hash manifest incomplete')
        for series in SERIES:
            src = Path(raw_dir) / f'{series}.csv'
            if not src.is_file() or sha256(src) != hashes[series]:
                raise ReleaseIntegrityError(f'{series}.csv absent or does not match this run')
    expected_id = fingerprint_run_id(
        kind=kind, label=manifest['source_label'], input_hashes=manifest['input_sha256'],
        start=manifest['research_window']['start'], end=manifest['research_window']['end'],
        variants=manifest['variants'],
        include_baselines=manifest['includes_statistical_baselines'],
        artifacts=manifest['artifacts'], code_hashes=code_hashes,
    )
    if manifest.get('run_id') != expected_id:
        raise ReleaseIntegrityError('Run ID does not match model code, inputs, parameters, and artifact inventory')
    hashes = manifest.get('input_sha256', {})
    if set(hashes) != set(SERIES):
        raise ReleaseIntegrityError('Input-source hash manifest incomplete')
    if verify_artifacts:
        artifacts = manifest.get('artifacts')
        if not isinstance(artifacts, dict) or not artifacts:
            raise ReleaseIntegrityError('Missing artifact fingerprint inventory')
        for relative, info in artifacts.items():
            candidate = _contained(folder, relative)
            if not candidate.is_file() or candidate.stat().st_size != info.get('bytes') or sha256(candidate) != info.get('sha256'):
                raise ReleaseIntegrityError(f'Artifact missing or changed: {relative}')
        actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        registered = set(artifacts) | {MANIFEST_NAME}
        if actual_files != registered:
            differences = sorted(actual_files.symmetric_difference(registered))
            raise ReleaseIntegrityError(f'Unregistered or missing release file(s): {differences[:5]}')
        for required in REQUIRED_ROOT:
            if required not in artifacts:
                raise ReleaseIntegrityError(f'Artifact not registered: {required}')
        for variant in VARIANTS:
            name = f'{variant}/{manifest["source_label"]}_metadata.json'
            if name not in artifacts:
                raise ReleaseIntegrityError(f'Missing registered model metadata: {name}')
    return manifest
