"""Unambiguous synthetic dry-run; NEVER use outputs as market findings."""
from pathlib import Path
import json
import tempfile
import pandas as pd
from crosswatch import generate_synthetic_levels, SERIES
from data_quality_gate import audit
from crosswatch_phase3 import run_phase3_part1

def main():
    root=Path(__file__).resolve().parent.parent
    dest=root/'evidence'/'synthetic_qa'
    dest.mkdir(parents=True,exist_ok=True)
    data=generate_synthetic_levels(n=1400,seed=812)
    # Re-date simulated points into a historical-looking window solely to test date filters.
    data.index=pd.bdate_range('2017-01-03',periods=len(data))
    with tempfile.TemporaryDirectory(prefix='CROSSWATCH_SYNTHETIC_QA_') as t:
        private=Path(t)
        for s in SERIES:
            pd.DataFrame({'DATE':data.index.strftime('%Y-%m-%d'),s:data[s].values}).to_csv(private/f'{s}.csv',index=False)
        qa=audit(private,dest,'2017-01-03','2026-11-18')
    result=run_phase3_part1(data,dest,source_type='SYNTHETIC_SOFTWARE_QA_ONLY',extra_provenance={'synthetic_test_only':True,'not_market_data':True})
    summary={'data_type':'SYNTHETIC ONLY, NOT OBSERVED MARKET DATA',
             'synthetic_rows':len(data),'audit_complete_rows':qa['n_joined_level_days'],
             'synthetic_quality_gate_pass':qa['data_quality_pass'],
             'split_sizes':{k:v['n'] for k,v in qa['split'].items()},
             'models_run':['equity_return_baseline','global_isolation_forest','VIX_regime_isolation_forest'],
             'training_rules_found':int(len(result['rules'])),
             'alert_overlap':result['overlap'].to_dict(orient='records'),
             'scientific_inference_permitted':False,
             'note':'Dry run to verify code paths only; do not compare to historical finance performance.'}
    (dest/'SYNTHETIC_QA_STATUS.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__':main()
