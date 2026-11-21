from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Sources'))
from release_preflight import check
from run_integrated_synthetic_qa import main as make_run


def test_missing_private_provenance_blocks_release(tmp_path):
    report=ROOT/'CROSSWATCH_IEEE_Style_Report.docx'
    status=check(tmp_path,tmp_path/'no-inputs',tmp_path/'no-results',report)
    assert status['pass_all_release_gates'] is False
    item={x['check']:x['status'] for x in status['checks']}
    assert item['Permission evidence gate']=='PENDING'
    assert item['Origin/file hash gate']=='PENDING'
    assert item['Historical released results']=='PENDING'
    assert item['Final report PDF']=='PENDING'


def test_synthetic_run_cannot_pass_as_authorized_historical(tmp_path):
    result,raw=tmp_path/'synthetic-models',tmp_path/'synthetic-inputs'
    assert make_run(['--out-dir',str(result),'--raw-dir',str(raw)])==0
    status=check(ROOT,raw,result,ROOT/'CROSSWATCH_IEEE_Style_Report.docx',
                 video_url='https://example.org/video',live_demo_attested=True,
                 video_access_attested=True,report_pdf_reviewed=True)
    item={x['check']:x['status'] for x in status['checks']}
    assert item['CSV format, chronology and window']=='PASS'
    assert item['Historical released results']=='BLOCKED'
    assert status['pass_all_release_gates'] is False
