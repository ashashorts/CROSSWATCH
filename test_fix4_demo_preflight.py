"""Public-code demonstration preflight always uses generated synthetic data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Sources'))
from run_integrated_synthetic_qa import main as synthetic_run
from verify_demo_preflight import verify

def test_nonbrowser_demo_preflight_four_combinations(tmp_path):
    results,raw=tmp_path/'SYNTHETIC_results',tmp_path/'SYNTHETIC_input'
    assert synthetic_run(['--out-dir',str(results),'--raw-dir',str(raw)])==0
    r=verify(results,raw)
    assert r['pass'] and not r['live_browser_tested']
    assert set(r['models'])=={'three_features','five_features'}
    for v in r['models'].values():
        assert set(v)=={'validation','test'}
        assert all(s['charts_rendered_in_memory']>=1 for s in v.values())
