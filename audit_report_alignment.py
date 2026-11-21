"""Independent read-only comparison of the course report with CROSSWATCH outputs.

A numerical PASS checks report consistency, NOT data licensing, research
validity, team authorship, or demo availability. The private-QA option does not
run models, does not bypass any permission gate, and does not grant release status.

Examples:
  python Sources/audit_report_alignment.py --results-dir private-results --raw-dir private-inputs --report-docx CROSSWATCH_IEEE_Style_Report.docx --output private-review/report_alignment.json
  python Sources/audit_report_alignment.py --results-dir private-QA --allow-private-qa --report-docx CROSSWATCH_IEEE_Style_Report.docx
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import math
import pandas as pd

NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


class AlignmentError(ValueError):
    pass


def _docx_paragraphs_and_tables(path:Path):
    if not path.is_file():
        raise AlignmentError(f'Report DOCX missing: {path}')
    with zipfile.ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'))
    def text(n):
        return ''.join(node.text or '' for node in n.findall('.//w:t',NS)).strip()
    paragraphs=[text(p) for p in root.findall('.//w:p',NS)]
    tables=[]
    for table in root.findall('.//w:tbl',NS):
        rows=[]
        for row in table.findall('./w:tr',NS):
            rows.append([text(cell) for cell in row.findall('./w:tc',NS)])
        tables.append(rows)
    return '\n'.join(paragraphs), tables


def _assert_near(actual, expected, label, checks, *, decimals=None):
    if decimals is not None:
        expected=round(float(expected),decimals)
        passed=math.isclose(float(actual),expected,abs_tol=0.5*10**(-decimals)+1e-10)
    else:
        passed=int(actual)==int(expected)
    checks.append({'item':label,'pass':bool(passed),'report_value':actual,'computed_value':expected})
    return passed


def _meta(root:Path,variant:str,label:str)->dict:
    p=root/variant/f'{label}_metadata.json'
    if not p.exists():raise AlignmentError(f'Missing metadata {p}')
    return json.loads(p.read_text(encoding='utf-8'))


def audit(results_dir:Path, report_docx:Path, *, raw_dir:Path|None=None, allow_private_qa:bool=False)->dict:
    results_dir=Path(results_dir)
    if not results_dir.exists():raise AlignmentError(f'Results directory does not exist: {results_dir}')
    manifest_path=results_dir/'crosswatch_run_manifest.json'
    if manifest_path.is_file():
        from release_integrity import read_manifest
        m=read_manifest(results_dir,raw_dir=raw_dir)
        label=m['source_label'];run_id=m['run_id'];released=True;run_kind=m['run_kind']
    else:
        if not allow_private_qa:raise AlignmentError('No released manifest. Use a completed authorized run; --allow-private-qa permits read-only, explicitly unreleased checking')
        options=sorted((results_dir/'five_features').glob('*_metadata.json'))
        if len(options)!=1:raise AlignmentError('QA folder must contain exactly one five-feature metadata file')
        label=options[0].name[:-len('_metadata.json')]
        if label in ('RIGHTS_REVIEWED_MARKET_DATA','FRED_HISTORICAL'):
            raise AlignmentError('Rights-reviewed label is not accepted without the full signed manifest')
        run_id=None;released=False;run_kind='private_unreleased_numerical_qa'
    body,tables=_docx_paragraphs_and_tables(Path(report_docx))
    if len(tables)<3:raise AlignmentError('Report must have at least 3 structured tables')
    op=pd.read_csv(results_dir/'feature_comparison_operating.csv')
    budget=pd.read_csv(results_dir/'feature_comparison_budget.csv')
    meta3=_meta(results_dir,'three_features',label)
    meta5=_meta(results_dir,'five_features',label)
    checks=[]
    # Reporting and chronology contract: no split or proxy arithmetic can differ by feature set.
    if meta3['splits']!=meta5['splits']:raise AlignmentError('The two models have inconsistent chronological splits')
    for part, row in zip(('train','validation','test'),tables[0][1:4]):
        _assert_near(row[2],meta5['splits'][part]['n_rows'],f'Table I {part} rows',checks)
        dates=row[1]
        from dateutil.parser import parse as dateparse
        fields=re.split(r'\s*[–—-]\s*',dates)
        if len(fields)!=2:raise AlignmentError(f'Cannot parse reported split window {dates}')
        start=dateparse(fields[0]).date()
        end=dateparse(fields[1]).date()
        _assert_near(int(start==pd.Timestamp(meta5['splits'][part]['first']).date()),1,f'Table I {part} first date',checks)
        _assert_near(int(end==pd.Timestamp(meta5['splits'][part]['last']).date()),1,f'Table I {part} last date',checks)
    cutoff=meta5['regime_calibration']
    cutoff_expected={
        'Global severity': (meta5['global_severity_cutoff_95th'],4),
        'VIX boundary': (cutoff['vix_training_cutoff'],2),
        'Ordinary VIX severity': (cutoff['ordinary_severity_cutoff'],4),
        'Elevated VIX severity': (cutoff['elevated_severity_cutoff'],4),
    }
    for row in tables[1][1:]:
        if row[0] not in cutoff_expected:raise AlignmentError(f'Unexpected cutoff name: {row[0]}')
        v,precision=cutoff_expected[row[0]]
        _assert_near(float(row[1]),v,f'Table II {row[0]}',checks,decimals=precision)
    # Table III requires independent use of two root CSVs and both feature-set rows.
    methods={
        'Equity-tail':('baseline_alert','baseline','three_features'),
        'Training-reference z-score':('fixed_zscore_alert','fixed_zscore','three_features'),
        'Rolling z-score':('rolling_zscore_alert','rolling_zscore','three_features'),
        'Global forest, 3 features':('IFOREST_ALERT','global_iforest','three_features'),
        'Global forest, 5 features':('IFOREST_ALERT','global_iforest','five_features'),
        'VIX-context forest, 3 features':('REGIME_ALERT','regime_iforest','three_features'),
        'VIX-context forest, 5 features':('REGIME_ALERT','regime_iforest','five_features'),
    }
    for row in tables[2][1:]:
        if row[0] not in methods:raise AlignmentError(f'Unknown report Table III method {row[0]}')
        operating,budget_name,feature=methods[row[0]]
        t=op.loc[(op.method==operating)&(op.feature_set==feature)&(op.split=='test')]
        v=budget.loc[(budget.method==budget_name)&(budget.feature_set==feature)&(budget.split=='validation')]
        b=budget.loc[(budget.method==budget_name)&(budget.feature_set==feature)&(budget.split=='test')]
        if not (len(t)==len(v)==len(b)==1):raise AlignmentError(f'Missing method comparison rows for {row[0]}')
        nums=re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*',row[1])
        if not nums:raise AlignmentError(f'Bad test hits/alerts entry for {row[0]}: {row[1]}')
        _assert_near(nums.group(1),t['proxy_hits'].iloc[0],f'Table III {row[0]} test proxy hits',checks)
        _assert_near(nums.group(2),t['alerts'].iloc[0],f'Table III {row[0]} test alerts',checks)
        _assert_near(row[2],v['proxy_hits'].iloc[0],f'Table III {row[0]} validation budget hits',checks)
        _assert_near(row[3],b['proxy_hits'].iloc[0],f'Table III {row[0]} test budget hits',checks)
    # Independently check actual denominators reported in Evaluation prose.
    validation=op.loc[(op.feature_set=='five_features')&(op.split=='validation')]
    test=op.loc[(op.feature_set=='five_features')&(op.split=='test')]
    if validation.empty or test.empty:raise AlignmentError('Missing five-feature evaluation splits')
    val_evaluable=int(validation['evaluable_days'].iloc[0]);test_evaluable=int(test['evaluable_days'].iloc[0])
    val_positive=int(validation['proxy_positive_days'].iloc[0]);test_positive=int(test['proxy_positive_days'].iloc[0])
    para=re.search(r'validation and test partitions each have\s*([\d,]+)\s*evaluable dates.*?validation has\s*([\d,]+)\s*positive dates.*?test has\s*([\d,]+)',body,re.I|re.S)
    if not para:raise AlignmentError('Cannot locate report evaluation denominator paragraph')
    for name,given,actual in [('Evaluation evaluable days',int(para.group(1).replace(',','')),test_evaluable),('Evaluation validation positives',int(para.group(2).replace(',','')),val_positive),('Evaluation test positives',int(para.group(3).replace(',','')),test_positive)]:
        _assert_near(given,actual,name,checks)
    _assert_near(val_evaluable,test_evaluable,'Equal evaluation sample size',checks)
    # The known report case has a precise score and market-feature description.
    path=results_dir/'five_features'/f'{label}_test_derived_results.csv'
    data=pd.read_csv(path)
    if 'DATE' not in data.columns:raise AlignmentError('Test derived results missing DATE')
    case=data.loc[data['DATE'].eq('2026-11-05')]
    if len(case)==1:
        row=case.iloc[0]
        case_para=re.search(r'On November 5, 2026.*?severity was\s*(\d+(?:\.\d+)?).*?crossed the\s*(\d+(?:\.\d+)?)\s*global cutoff.*?below the\s*(\d+(?:\.\d+)?)\s*elevated-VIX cutoff.*?fell\s*(\d+(?:\.\d+)?)%.*?rose\s*(\d+(?:\.\d+)?)\s*points',body,re.I|re.S)
        if case_para:
            case_checks=[('Case severity',float(case_para.group(1)),row['anomaly_severity'],3),('Case global cutoff',float(case_para.group(2)),meta5['global_severity_cutoff_95th'],3),('Case elevated-VIX cutoff',float(case_para.group(3)),cutoff['elevated_severity_cutoff'],3),('Case equity decline magnitude',float(case_para.group(4)),-row['sp_return_pct'],2),('Case VIX change points',float(case_para.group(5)),row['vix_delta_points'],2)]
            for n,x,y,d in case_checks:_assert_near(x,y,n,checks,decimals=d)
        else:
            checks.append({'item':'Detailed November 5 case sentence could not be parsed','pass':False})
    else:
        checks.append({'item':'November 5 case date is absent or duplicated','pass':False})
    unresolved=[]
    for pattern,label2 in [(r'\[Team number and author names\]|Team number:\s*\[(?:REQUIRED|TBD|PENDING)\]','Team number and author names'),(r'\[Each team member:|\[Insert verified individual','Member learning outcomes'),(r'\[Enter the required peer evaluations','Group member ratings'),(r'\[.*(?:video|link|URL).*?\]','Actual video link')]:
        if re.search(pattern,body,re.I):unresolved.append(label2)
    if 'https://drive.google.com' not in body and 'https://youtu.be' not in body and 'https://www.youtube.com' not in body:
        unresolved.append('Verified actual video URL absent')
    return {
        'audit_type':'CROSSWATCH_READ_ONLY_REPORT_ALIGNMENT',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'numerical_checks_passed':all(c['pass'] for c in checks),
        'number_of_checks':len(checks),'failed_checks':[c for c in checks if not c['pass']],
        'checks':checks,'run_id':run_id,'run_kind':run_kind,'artifact_manifest_verified':released,
        'user_report_completion_pending':sorted(set(unresolved)),
        'final_submission_ready':False, # Real demo, team attestations, and permissions require human review.
        'notes':'Numerical consistency does not prove future prediction, license coverage, untouched test data, or full submission readiness. No raw source series exported.',
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',required=True,type=Path)
    p.add_argument('--raw-dir',type=Path)
    p.add_argument('--report-docx',required=True,type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--allow-private-qa',action='store_true',help='READ-ONLY consistency check of unreleased test output; NEVER classifies it as authorized')
    a=p.parse_args(argv)
    try:
        x=audit(a.results_dir,a.report_docx,raw_dir=a.raw_dir,allow_private_qa=a.allow_private_qa)
    except Exception as e:
        print('REPORT AUDIT ERROR:',type(e).__name__,str(e));return 2
    if a.output:
        if a.output.resolve().is_relative_to(a.results_dir.resolve()):
            print('Refusing to modify immutable run directory');return 2
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(x,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o))+'\n',encoding='utf-8')
    print(f'REPORT NUMERICAL CHECKS: {sum(c["pass"] for c in x["checks"])}/{len(x["checks"])} PASS; release manifest present={x["artifact_manifest_verified"]}; submission-ready={x["final_submission_ready"]}')
    for f in x['failed_checks']:print('  FAIL:',f)
    print('Human completion pending:', ', '.join(x['user_report_completion_pending']) or 'none flagged')
    return 0 if x['numerical_checks_passed'] else 3

if __name__=='__main__':raise SystemExit(main())
