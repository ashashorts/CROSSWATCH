"""Safe read-only ZIP member audit, intended to prevent private data in code packages."""
import argparse,sys,zipfile
from pathlib import PurePosixPath

REQUIRED=('Sources/','Tests/','Sources/README.md','README.md','CROSSWATCH_IEEE_Style_Report.docx')

def check(path):
    issues=[]
    with zipfile.ZipFile(path) as z:
        names=[n.replace('\\','/') for n in z.namelist() if not n.endswith('/')]
        rel=[n.partition('/')[2] if n.startswith('CROSSWATCH/') else n for n in names]
        for n in rel:
            lower=n.lower()
            parts=PurePosixPath(n).parts
            if any(segment.startswith('.crosswatch-building') for segment in parts):issues.append('unfinished temp artifacts: '+n)
            if 'private-inputs' in parts or 'private-results' in parts or 'private-review' in parts:issues.append('private run files: '+n)
            if n in ('docs/license_review.json','docs/origin_manifest.json'):issues.append('private permission/provenance: '+n)
            if lower.endswith(('.pem','.key','.env')):issues.append('possible secret file: '+n)
            if 'RIGHTS_REVIEWED_MARKET_DATA' in n or 'FRED_HISTORICAL_test_derived_results' in n:issues.append('historical model artifact: '+n)
        for r in REQUIRED:
            if not any(n==r or (r.endswith('/') and n.startswith(r)) for n in rel):issues.append('missing required path: '+r)
    return issues, len(names)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args()
    found, count=check(a.archive);print('Audited',count,'files; findings:',len(found))
    for s in found[:30]:print('BLOCK:',s)
    sys.exit(bool(found))
