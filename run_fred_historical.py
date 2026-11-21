"""Canonical, fail-closed CROSSWATCH historical pipeline (full 3/5-feature comparison).

Official real-data execution requires private, researcher-attested evidence. The
pipeline creates results atomically, then registers every artifact and its SHA-256
for the dashboard, notebook and case-exporter. No FRED API or scraping is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from approved_real_run import require_rights, require_provenance, AuthorizationError
from association_evaluation import evaluate_associations
from crosswatch import FEATURES, PHASE1_FEATURES, SERIES, join_market_levels, read_fred_csv
from crosswatch_phase3 import run_phase3_part1
from data_quality_gate import audit
from inspect_private_inputs import require_full_history_window, DatasetCoverageError
from experiment_audit import audit_all
from episode_evaluation import evaluate_episodes
from run_experiment_stage import _plot_metrics
from release_integrity import write_manifest, read_manifest, sha256

START, END = '2017-01-03', '2026-11-18'
LABEL = 'RIGHTS_REVIEWED_MARKET_DATA'


def calculate_all(levels, equity_closes, stage: Path, *, label: str,
                  provenance: dict, start: str, end: str, kind: str, permission_basis: str | None = None):
    """The ONE underlying analysis method for authorized and explicitly synthetic runs."""
    from export_historical_cases import export_cases
    if label not in ('RIGHTS_REVIEWED_MARKET_DATA', 'SYNTHETIC_INTEGRATION_ONLY'):
        raise ValueError('Unsupported source label')
    variants = {'three_features': FEATURES, 'five_features': PHASE1_FEATURES}
    outputs = {}
    for name, features in variants.items():
        folder = stage / name
        result = run_phase3_part1(
            levels, folder, source_type=label,
            extra_provenance=provenance, equity_closes=equity_closes,
            model_features=features, comparison_features=PHASE1_FEATURES,
            include_zscore=True,
        )
        diagnostics = audit_all(result, folder, label)
        _plot_metrics(diagnostics, folder, label)
        rules, rule_audit = evaluate_associations(result['scored'])
        rules.to_csv(folder / 'association_rules.csv', index=False)
        (folder / 'association_verification.json').write_text(
            json.dumps(rule_audit, indent=2), encoding='utf-8')
        outputs[name] = result
    for split in ('train', 'validation', 'test'):
        pd.testing.assert_series_equal(
            outputs['three_features']['scored'][split]['forward_proxy'],
            outputs['five_features']['scored'][split]['forward_proxy'],
        )
    for metric in ('operating', 'budget'):
        table = pd.concat(
            [result[metric].assign(feature_set=name) for name, result in outputs.items()],
            ignore_index=True,
        )
        table.to_csv(stage / f'feature_comparison_{metric}.csv', index=False)
    for name, table in evaluate_episodes(outputs, equity_closes).items():
        table.to_csv(stage / f'{name}.csv', index=False)
    hashes = provenance['sha256']
    kwargs = dict(
        kind=kind, input_hashes=hashes, start=start, end=end,
        include_baselines=True,
        permission_status=(
            permission_basis or 'Researcher-attested academic permissions; legal sufficiency not independently certified'
            if kind == 'historical_research' else 'Synthetic QA only: no market-data license claim'
        ),
    )
    write_manifest(stage, **kwargs)
    # Figures are generated from the same verified scores as the dashboard.
    export_cases(stage, equity_closes, stage / 'Cases', variant='five_features', split='test')
    write_manifest(stage, **kwargs)
    read_manifest(stage, verify_artifacts=True)
    return outputs


def run_authorized(raw_dir: Path, out_dir: Path, rights_file: Path,
                   origin_file: Path, start: str=START, end: str=END) -> dict:
    """Validate first; stage all output privately; publish only if every step passes."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    root = Path(__file__).resolve().parent.parent
    if start != START or end != END:
        raise ValueError(f'Frozen original study requires {START} through {END}.')
    rights = require_rights(Path(rights_file), root)
    origin = require_provenance(Path(origin_file), root, raw_dir)
    require_full_history_window(raw_dir, start, end)
    if out_dir.exists():
        raise FileExistsError(f'Results directory already exists: {out_dir}. Use a new empty destination to prevent mixing runs.')
    # All preconditions have now passed; no artifact directory was created on failure.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.crosswatch-building-', dir=out_dir.parent))
    try:
        quality = audit(raw_dir, stage, start, end)
        # The data-quality gate itself never certifies licensing. Add the separate
        # documentary attestation status passed by this runner, without changing
        # the gate's independent-license/audit caveats.
        quality['researcher_rights_record_structurally_passed'] = True
        quality['origin_file_hashes_matched_private_manifest'] = True
        (stage/'raw_csv_quality_report.json').write_text(json.dumps(quality,indent=2),encoding='utf8')
        data = {s: read_fred_csv(s, raw_dir, allow_download=False) for s in SERIES}
        levels, _ = join_market_levels(data, start, end)
        origin_hashes = {s: origin['sources'][s]['sha256'] for s in SERIES}
        rights_basis = (
            'User-attested permissions held by instructor; underlying emails not independently inspected'
            if rights.get('status') == 'user_attested_instructor_held'
            else 'Researcher-attested local approval document hashes checked; legal scope not independently certified'
        )
        provenance = {
            'sha256': origin_hashes,
            'official_source_urls': {s: origin['sources'][s]['official_source_url'] for s in SERIES},
            'license_status': rights_basis,
        }
        calculate_all(levels, data['SP500'], stage, label=LABEL,
                      provenance=provenance, start=start, end=end, kind='historical_research',
                      permission_basis=rights_basis)
        # Detect replacement or modification while models ran, before promotion.
        read_manifest(stage, raw_dir=raw_dir)
        stage.rename(out_dir)
        return read_manifest(out_dir, raw_dir=raw_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--compare-features', action='store_true', help='Legacy compatibility; both models now required')
    p.add_argument('--compare-baselines', action='store_true', help='Legacy compatibility; all baselines now required')
    root = Path(__file__).resolve().parent.parent
    p.add_argument('--rights-file', type=Path, default=root / 'docs/license_review.json')
    p.add_argument('--origin-file', type=Path, default=root / 'docs/origin_manifest.json')
    p.add_argument('--start', default=START)
    p.add_argument('--end', default=END)
    args = p.parse_args(argv)
    try:
        m = run_authorized(args.raw_dir, args.out_dir, args.rights_file,
                           args.origin_file, args.start, args.end)
    except DatasetCoverageError as e:
        print('HISTORICAL COVERAGE BLOCKED:',type(e).__name__,str(e))
        return 3
    except Exception as e:
        print('HISTORICAL RUN BLOCKED:', type(e).__name__, str(e))
        return 2
    print(f'Historical run complete: {m["run_id"]}, {len(m["artifacts"])} artifacts, {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
