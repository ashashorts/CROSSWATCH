"""Create an UNVERIFIED origin manifest by hashing local files, without fetching any data.

A recorded hash is NOT evidence that the dataset came from the claimed provider.
Reviewer must inspect each file's actual origin and separately attest to it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SERIES = ('SP500', 'VIXCLS', 'DGS10')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Refusing to overwrite an existing origin record')
    records = {}
    for symbol in SERIES:
        file = args.raw_dir / f'{symbol}.csv'
        if not file.is_file():
            p.error(f'File missing: {file}; no auto-download is performed')
        records[symbol] = {
            'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
            'file_name': file.name,
            'size_bytes': file.stat().st_size,
            'official_source_url': '',
            'acquisition_method': '',
            'acquisition_date_utc': '',
            'origin_verified_by_researcher': False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'status': 'UNVERIFIED_ORIGIN_REVIEW_REQUIRED',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'notes': 'SHA-256 tracks local file integrity, not official provenance or usage rights.',
        'sources': records,
    }, indent=2) + '\n', encoding='utf-8')
    print('UNVERIFIED origin manifest saved:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
