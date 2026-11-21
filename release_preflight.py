"""Read-only comprehensive release-preflight for CROSSWATCH.

Does not run ML, download files, create authorization, or modify immutable
results. Outputs only aggregate statuses and non-sensitive numerical evidence.
The optional demo/video flags are HUMAN attestations, not remotely verified.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import platform
from datetime import datetime,timezone
import re
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def _record(checks,name,status,detail):
    checks.append({'check':name,'status':status,'detail':str(detail)})


def check(project_root:Path,raw_dir:Path,results_dir:Path,report_docx:Path,
          *,report_pdf:Path|None=None,video_url:str='',live_demo_attested:bool=False,
          video_access_attested:bool=False,report_pdf_reviewed:bool=False)->dict:
    project_root=Path(project_root)
    checks=[]
    from crosswatch import SERIES
    from release_integrity import read_manifest
    from approved_real_run import require_rights,require_provenance
    from inspect_private_inputs import require_full_history_window
    from data_quality_gate import audit
    from audit_report_alignment import _docx_paragraphs_and_tables,audit as report_audit

    if tuple(map(int,platform.python_version_tuple()[:2]))==(3,12):
        _record(checks,'Python version','PASS',platform.python_version())
    else:
        _record(checks,'Python version','WARNING',f'{platform.python_version()} here; verified requirements target Python 3.12')
    if importlib.util.find_spec('streamlit'):
        _record(checks,'Streamlit installed','PASS',importlib.metadata.version('streamlit'))
    else:
        _record(checks,'Streamlit installed','PENDING','Streamlit is not installed in this environment; cannot verify live UI here')
    expected=tuple(f'{s}.csv' for s in SERIES)
    if all((raw_dir/name).is_file() for name in expected):
        try:
            require_full_history_window(raw_dir,'2017-01-03','2026-11-18')
            with tempfile.TemporaryDirectory(prefix='cw-readonly-preflight-') as temporary:
                quality=audit(raw_dir,Path(temporary),'2017-01-03','2026-11-18')
            n=quality['n_joined_level_days']
            _record(checks,'CSV format, chronology and window','PASS',f'All 3 private files valid; {n} complete aligned dates')
        except Exception as exc:
            _record(checks,'CSV format, chronology and window','BLOCKED',f'{type(exc).__name__}: {exc}')
    else:
        _record(checks,'CSV format, chronology and window','PENDING','All three original private files must be present locally')
    rights_path=project_root/'docs/license_review.json'
    try:
        require_rights(rights_path,project_root)
        _record(checks,'Permission evidence gate','PASS','Local rights declaration structurally passes; instructor-held route is user-attested, not documentary or legal verification')
    except Exception as exc:
        _record(checks,'Permission evidence gate','PENDING',f'{type(exc).__name__}: {exc}')
    origin_path=project_root/'docs/origin_manifest.json'
    try:
        require_provenance(origin_path,project_root,raw_dir)
        _record(checks,'Origin/file hash gate','PASS','Source hashes match the researcher-reviewed private manifest')
    except Exception as exc:
        _record(checks,'Origin/file hash gate','PENDING',f'{type(exc).__name__}: {exc}')
    manifest=None
    if results_dir.exists():
        try:
            manifest=read_manifest(results_dir,raw_dir=raw_dir)
            if manifest['run_kind']!='historical_research':
                _record(checks,'Historical released results','BLOCKED',f'Run kind {manifest["run_kind"]} is not authentic historical release')
                manifest=None
            else:
                _record(checks,'Historical released results','PASS',f'{manifest["run_id"]}: {len(manifest["artifacts"])} artifact hashes and all 3 source hashes passed')
        except Exception as exc:
            _record(checks,'Historical released results','BLOCKED',f'{type(exc).__name__}: {exc}')
    else:
        _record(checks,'Historical released results','PENDING','Execute the rights-gated canonical runner into a fresh local private-results directory')
    if report_docx.is_file():
        try:
            body,_=_docx_paragraphs_and_tables(report_docx)
            patterns=[r'\[Team number and author names\]|Team number:\s*\[(?:REQUIRED|TBD|PENDING)\]',r'\[Each team member:|\[Insert verified individual',r'\[Enter the required peer evaluations',r'\[.*?video.*?\]']
            found=[p for p in patterns if re.search(p,body,re.I)]
            if found:
                _record(checks,'Final report author-specific fields','PENDING',f'{len(found)} unfinished author/team/video placeholder types')
            else:
                _record(checks,'Final report author-specific fields','PASS','No known author/team/video placeholders remain (truthfulness still requires manual review)')
        except Exception as exc:
            _record(checks,'Final report author-specific fields','BLOCKED',f'{type(exc).__name__}: {exc}')
        if manifest:
            try:
                alignment=report_audit(results_dir,report_docx,raw_dir=raw_dir)
                if alignment['numerical_checks_passed']:
                    _record(checks,'Report numerical alignment','PASS',f'{alignment["number_of_checks"]} numerical claims match verified release')
                else:
                    _record(checks,'Report numerical alignment','BLOCKED',f'{len(alignment["failed_checks"])} mismatched calculations')
            except Exception as exc:
                _record(checks,'Report numerical alignment','BLOCKED',f'{type(exc).__name__}: {exc}')
        else:
            _record(checks,'Report numerical alignment','PENDING','Requires the complete permission-gated released run, not archived or QA-only outputs')
    else:
        _record(checks,'Final report author-specific fields','BLOCKED','Editable final DOCX missing')
        _record(checks,'Report numerical alignment','PENDING','No report DOCX to compare')
    if report_pdf is not None and report_pdf.is_file():
        try:
            from pypdf import PdfReader
            pdf_reader=PdfReader(str(report_pdf))
            pages=len(pdf_reader.pages)
            if pages < 5 or pages > 10:
                _record(checks,'Final report PDF','BLOCKED',f'{pages} pages: course requires 5–10')
            elif not report_pdf_reviewed:
                _record(checks,'Final report PDF','PENDING',f'{pages} pages; visually inspect each page and attest to the review')
            else:
                _record(checks,'Final report PDF','PASS',f'{pages} pages; user attests visual inspection is complete')
        except ImportError:
            _record(checks,'Final report PDF','PENDING','pypdf unavailable for page count; manually verify 5–10 pages and inspect layout')
        except Exception as exc:
            _record(checks,'Final report PDF','BLOCKED',f'Unreadable PDF: {type(exc).__name__}: {exc}')
    else:
        _record(checks,'Final report PDF','PENDING','Export final 5–10 page report PDF and inspect every page')
    if video_url and re.match(r'^https://',video_url):
        _record(checks,'Recorded demo URL','PASS','HTTPS link supplied (not independently opened)')
    else:
        _record(checks,'Recorded demo URL','PENDING','Record the full narrated ~20-minute demonstration with presenter face visible and provide access URL')
    _record(checks,'Live Streamlit walkthrough','PASS' if live_demo_attested else 'PENDING',
            'User attests all tabs, models, partitions and case dates were tested locally' if live_demo_attested else 'Must execute actual browser demo; stub tests do not count')
    _record(checks,'Independent video-link access','PASS' if video_access_attested else 'PENDING',
            'User attests instructor/other account can access video' if video_access_attested else 'Check opening the uploaded video from an independent account')
    gates=['Python version','Streamlit installed','CSV format, chronology and window','Permission evidence gate','Origin/file hash gate',
           'Historical released results','Final report author-specific fields','Report numerical alignment',
           'Final report PDF','Recorded demo URL','Live Streamlit walkthrough','Independent video-link access']
    statuses={d['check']:d['status'] for d in checks}
    ready=all(statuses.get(n)=='PASS' for n in gates)
    return {
        'preflight_version':1,'checked_utc':datetime.now(timezone.utc).isoformat(),
        'pass_all_release_gates':ready,
        'recorded_human_attestations_are_not_automated_evidence':True,
        'checks':checks,
        'omitted_private_material':['CSV contents','license email contents','credential paths'],
        'summary':{s:sum(d['status']==s for d in checks) for s in ['PASS','PENDING','WARNING','BLOCKED']},
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--raw-dir',type=Path,default=Path('private-inputs'))
    p.add_argument('--results-dir',type=Path,default=Path('private-results'))
    p.add_argument('--report-docx',type=Path,default=Path('CROSSWATCH_IEEE_Style_Report.docx'))
    p.add_argument('--report-pdf',type=Path,default=Path('CROSSWATCH_IEEE_Style_Report.pdf'))
    p.add_argument('--video-url',default='')
    p.add_argument('--confirm-report-pdf-reviewed',action='store_true',help='Only after personally checking the rendered final PDF')
    p.add_argument('--confirm-live-demo-tested',action='store_true',help='Only after conducting a real browser test yourself')
    p.add_argument('--confirm-video-access-tested',action='store_true',help='Only after testing shared access to uploaded video')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    result=check(args.project_root,args.raw_dir,args.results_dir,args.report_docx,report_pdf=args.report_pdf,
                 video_url=args.video_url,live_demo_attested=args.confirm_live_demo_tested,
                 video_access_attested=args.confirm_video_access_tested,
                 report_pdf_reviewed=args.confirm_report_pdf_reviewed)
    for item in result['checks']:
        print(f"[{item['status']}] {item['check']}: {item['detail']}")
    print('READY FOR FINAL SUBMISSION:',result['pass_all_release_gates'])
    if args.output:
        target=args.output
        if target.resolve().is_relative_to(args.results_dir.resolve()):
            p.error('Cannot write preflight report inside immutable model results')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,indent=2,default=str)+'\n',encoding='utf-8')
    return 0 if result['pass_all_release_gates'] else 1

if __name__=='__main__':raise SystemExit(main())
