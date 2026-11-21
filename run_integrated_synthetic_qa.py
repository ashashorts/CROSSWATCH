"""Exercise the COMPLETE Fix 2 results contract with unmistakably synthetic prices.

No restricted data, download, or permission override is involved. This command
is for engineering demonstration and is NOT an empirical financial experiment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import tempfile
import json
import numpy as np
import pandas as pd

from crosswatch import SERIES, join_market_levels
from data_quality_gate import audit
from release_integrity import read_manifest, sha256
from run_fred_historical import calculate_all, START, END


def generate_fixture(raw_dir: Path, seed: int=2026) -> dict:
    """Business-day simulation; never intended to resemble historical observations."""
    raw_dir.mkdir(parents=True, exist_ok=False)
    dates = pd.bdate_range(START, END)
    n = len(dates)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00012, .008, n)
    returns[::131] -= 0.047
    eq = 2800 * np.exp(np.cumsum(returns))
    vix = np.maximum(8., 17 + 130 * np.maximum(-returns,0) + 3 * np.sin(np.arange(n)/40))
    treasury = np.clip(2.7 + np.cumsum(rng.normal(0,.013,n)), .2, 7.)
    outputs = {'SP500':eq, 'VIXCLS':vix, 'DGS10':treasury}
    for k,values in outputs.items():
        pd.DataFrame({'DATE':dates.strftime('%Y-%m-%d'),k:values}).to_csv(raw_dir/f'{k}.csv',index=False)
    return {k:pd.Series(v,index=dates,name=k) for k,v in outputs.items()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=Path('evidence/private_synthetic_integration'))
    p.add_argument('--raw-dir',type=Path,default=Path('evidence/private_synthetic_inputs'))
    a=p.parse_args(argv)
    if a.out_dir.exists() or a.raw_dir.exists():
        raise FileExistsError('Use new paths for deterministic, immutable synthetic QA runs')
    a.out_dir.parent.mkdir(parents=True,exist_ok=True)
    a.raw_dir.parent.mkdir(parents=True,exist_ok=True)
    values = generate_fixture(a.raw_dir)
    stage = Path(tempfile.mkdtemp(prefix='.synthetic-crosswatch-',dir=a.out_dir.parent))
    try:
        audit(a.raw_dir,stage,START,END)
        hashes={k:sha256(a.raw_dir/f'{k}.csv') for k in SERIES}
        levels,_=join_market_levels(values,START,END)
        calculate_all(levels,values['SP500'],stage,label='SYNTHETIC_INTEGRATION_ONLY',
                      provenance={'sha256':hashes, 'purpose':'SYNTHETIC SOFTWARE INTEGRATION TEST ONLY'},
                      start=START,end=END,kind='synthetic_integration_qa')
        m=read_manifest(stage,raw_dir=a.raw_dir)
        stage.rename(a.out_dir)
        print('SYNTHETIC SOFTWARE TEST ONLY. Run ID:',m['run_id'], 'artifacts:',len(m['artifacts']))
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
