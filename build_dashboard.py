"""Build an offline, one-file CROSSWATCH reviewer dashboard from QA evidence.

SECURITY/RESEARCH INTEGRITY:
- Accepts ONLY explicit synthetic QA evidence for this demonstration builder.
- Does NOT fetch, mine, reproduce, or fit models to licensed market data.
- Never labels generated observations as real market data.
- No raw market index values in embedded browser payload.
- No third-party JavaScript, CDN or network connection needed.
"""
from __future__ import annotations

import argparse
import json
import html
from pathlib import Path
import pandas as pd

PREFIX = 'SYNTHETIC_QA_ONLY'
REQUIRED = {
    'DATE','sp_return_pct','vix_delta_points','yield_delta_bp',
    'anomaly_severity','baseline_alert','IFOREST_ALERT','REGIME_ALERT',
    'cross_asset_confirmation','explanation','vix_regime'
}
PERMITTED_BROWSE_FIELDS = [
    'DATE','sp_return_pct','vix_delta_points','yield_delta_bp','anomaly_severity',
    'baseline_alert','IFOREST_ALERT','REGIME_ALERT','cross_asset_confirmation',
    'explanation','vix_regime','forward_proxy'
]

def read_and_validate(directory:Path)->dict:
    directory = Path(directory)
    def path(end:str)->Path:
        return directory/f'{PREFIX}_{end}'
    paths = {n:path(s) for n,s in {
        'rows':'test_derived_results.csv',
        'metrics':'operating_metrics.csv',
        'rules':'training_rules.csv',
        'meta':'metadata.json',
        'budget':'budget_matched_metrics.csv'
    }.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing required synthetic QA input: '+', '.join(missing))
    metadata = json.loads(paths['meta'].read_text(encoding='utf-8'))
    if metadata.get('source_type') != PREFIX:
        raise ValueError('STOP: demo builder only permits explicitly SYNTHETIC_QA_ONLY evidence.')
    rows = pd.read_csv(paths['rows'])
    absent = REQUIRED.difference(rows.columns)
    if absent:
        raise ValueError(f'Synthetic QA evidence does not contain required fields: {sorted(absent)}')
    if 'SP500' in rows.columns or 'VIXCLS' in rows.columns or 'DGS10' in rows.columns:
        raise ValueError('Raw restricted market levels are not allowed in this offline dashboard')
    if len(rows) < 40:
        raise ValueError('QA evidence too short for an informative dashboard')
    dates = pd.to_datetime(rows['DATE'], errors='coerce')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Invalid, duplicated, or unordered QA dates')
    for col in ('baseline_alert','IFOREST_ALERT','REGIME_ALERT'):
        if not rows[col].astype(str).isin(('True','False','true','false','0','1')).all():
            raise ValueError('Invalid boolean flag: '+col)
        rows[col] = rows[col].astype(str).str.lower().isin(('true','1'))
    if not rows['cross_asset_confirmation'].between(0,3).all():
        raise ValueError('Cross-asset confirmation score must be 0-3')
    rows = rows[PERMITTED_BROWSE_FIELDS]
    for col in ('sp_return_pct','vix_delta_points','yield_delta_bp','anomaly_severity'):
        rows[col] = rows[col].round(4)
    # JSON NaN is invalid in browsers: standardize missing values to null.
    rows = rows.where(pd.notnull(rows),None)
    # Conservative validation: actual output should include no price levels, nor guessed identities.
    metrics = pd.read_csv(paths['metrics'])
    budget = pd.read_csv(paths['budget'])
    rules = pd.read_csv(paths['rules'])
    if not {'method','split','alerts','precision_vs_proxy'}.issubset(metrics.columns):
        raise ValueError('Operating metrics schema mismatch')
    if not {'antecedent','support','confidence','lift'}.issubset(rules.columns):
        raise ValueError('Rule table schema mismatch')
    data = {
        'status':'SYNTHETIC SOFTWARE TEST - NOT FINANCIAL RESEARCH',
        'metadata':{
            'n_rows':int(metadata['n_rows']),
            'data_start':metadata['data_start'],
            'data_end':metadata['data_end'],
            'test':metadata.get('splits',{}).get('test',{}),
        },
        'rows':rows.to_dict(orient='records'),
        'metrics':metrics.round(4).fillna('').to_dict(orient='records'),
        'budget':budget.round(4).fillna('').to_dict(orient='records'),
        'rules':rules.round(4).fillna('').to_dict(orient='records'),
    }
    return data


def render(data:dict)->str:
    # Do not permit literal <script> delimiters from CSV fields to close the embedded script.
    payload = json.dumps(data,ensure_ascii=True,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    template = r'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CROSSWATCH | Research Software Demonstration</title>
<style>
:root{--bg:#090f1e;--card:#121e32;--line:#253650;--ink:#e8f0fa;--muted:#9bb0c7;--teal:#59dcc0;--blue:#7ca5ff;--amber:#ffc778;--red:#ff8f98}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#090f1e,#132038 75%);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1170px;margin:auto;padding:28px 20px 65px}
h1{font-size:34px;letter-spacing:-1.2px;margin:0}h2{font-size:18px;margin:0 0 13px;letter-spacing:-.25px}p{color:var(--muted)}.eyebrow{color:var(--teal);font-weight:750;letter-spacing:.16em;font-size:12px}.warning{border:1px solid #94654b;background:#30281f;padding:15px 18px;border-radius:12px;color:#ffe2b4;font-weight:700;margin:22px 0}.subtitle{margin:8px 0 18px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:24px 0}.kpi,.panel{background:var(--card);border:1px solid var(--line);border-radius:14px}.kpi{padding:18px}.kpi .k{color:var(--muted);font-size:12px;letter-spacing:.03em}.kpi .v{font-size:28px;font-weight:800;letter-spacing:-.9px;margin-top:6px}.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:15px}.panel{padding:22px;margin-top:15px}label{color:var(--muted);font-size:12px;display:block;margin-bottom:5px}select,input[type=range]{width:100%;background:#1e2e45;color:var(--ink);border:1px solid #365173;border-radius:8px;padding:11px;font-size:14px}.controlgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.flag{border-radius:8px;padding:5px 10px;font-size:12px;font-weight:700;display:inline-block;border:1px solid #38516a;margin:2px}.yes{background:#254d43;color:#9aedd1}.no{color:var(--muted)}.metric{color:var(--teal);font-weight:800}.strip{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.strip .item{border:1px solid #30445e;border-radius:10px;padding:9px 11px;min-width:110px}.strip strong{display:block;font-size:17px}.strip small{color:var(--muted)}
#chart{width:100%;height:260px;background:#0d1829;border-radius:10px}.chartkey{display:flex;gap:14px;color:var(--muted);font-size:12px;margin-bottom:13px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{color:#afc0d5;font-size:12px;text-align:left;font-weight:700}th,td{padding:10px;border-bottom:1px solid #26364a}tr:last-child td{border:0}td{font-variant-numeric:tabular-nums}.foot{margin-top:20px;font-size:13px;color:var(--muted)}.tiny{font-size:12px}.right{text-align:right}a{color:var(--teal)}.tag{font-size:11px;border:1px solid #4b6280;color:var(--muted);border-radius:15px;padding:2px 8px}
@media(max-width:820px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.controlgrid{grid-template-columns:1fr}}@media(max-width:420px){.cards{grid-template-columns:1fr}h1{font-size:29px}}
</style></head><body><main>
<div class="eyebrow">CSC 4740 · DATA MINING · PHASE 3</div><h1>CROSSWATCH</h1><p class="subtitle">Explainable cross-asset anomaly detection · Research software demonstration <span class="tag">Offline</span></p>
<div class="warning" role="alert">SYNTHETIC SOFTWARE TEST DATA — NOT REAL FINANCIAL MARKET OBSERVATIONS. Neither the dates nor results on this dashboard are evidence of historic market behavior or predictive performance.</div>
<div class="cards">
  <div class="kpi"><div class="k">Test observations</div><div class="v" id="n">--</div></div>
  <div class="kpi"><div class="k">Global IF alerts</div><div class="v" id="global">--</div></div>
  <div class="kpi"><div class="k">Regime-adjusted alerts</div><div class="v" id="regime">--</div></div>
  <div class="kpi"><div class="k">Alert disagreement</div><div class="v" id="diff">--</div></div>
</div>
<div class="grid">
 <section class="panel"><h2>Daily anomaly severity · synthetic test period</h2><div class="chartkey"><span><i class="dot" style="background:var(--blue)"></i>Isolation Forest severity</span><span><i class="dot" style="background:var(--red)"></i>Global IF alert</span><span><i class="dot" style="background:var(--teal)"></i>Regime alert</span></div><svg id="chart" viewBox="0 0 760 265" preserveAspectRatio="none" role="img" aria-label="Synthetic test scores over observed dates"></svg><p class="tiny">Scores are from one fitted Isolation Forest; the two alert variants use different training-derived thresholds. Click a plotted point to inspect its simulated day.</p></section>
 <section class="panel"><h2>Date-level explanation</h2><div class="controlgrid"><div><label for="filter">Show observations</label><select id="filter"><option value="all">All synthetic days</option><option value="global">Global IF alerts</option><option value="regime">Regime alerts</option><option value="either">Either method flags</option><option value="diff">Methods disagree</option></select></div><div><label for="when">Selected date</label><select id="when"></select></div></div><div class="strip" id="signals"></div><div style="margin-top:14px" id="flags"></div><p id="explain" style="min-height:48px"></p></section>
</div>
<div class="grid">
 <section class="panel"><h2>Operating-point comparison</h2><p class="tiny">Forward five-session decline proxy is not a verified stress label. Alert counts differ; compare budget-matched results separately.</p><div class="tablewrap"><table id="metrics"><thead><tr><th>Split</th><th>Method</th><th class="right">Alerts</th><th class="right">Proxy hits</th><th class="right">Proxy hit fraction</th></tr></thead><tbody></tbody></table></div></section>
 <section class="panel"><h2>Association rules</h2><p class="tiny">Rules were mined on synthetic training evidence. Support, confidence and lift measure co-occurrence, not causality.</p><div class="tablewrap"><table id="rules"><thead><tr><th>Antecedent → Alert</th><th class="right">Support</th><th class="right">Confidence</th><th class="right">Lift</th></tr></thead><tbody></tbody></table></div></section>
</div>
<div class="panel"><h2>Review notes and provenance</h2><p>This offline page contains <strong>generated QA data only</strong> and demonstrates the final user's workflow: see the three methods, inspect a flagged date, interpret the 0–3 cross-asset confirmation score, and compare training-discovered association rules. Raw S&amp;P 500, Cboe VIX, and 10-year Treasury prices are not embedded. A future historical evaluation requires authentic complete files and a documented permitted-use decision.</p><div class="foot">Built with the CROSSWATCH Step 4 reviewer export. Python model: scikit-learn Isolation Forest. Browser: self-contained HTML + vanilla JavaScript. Nothing is sent over the network.</div></div>
</main><script>const DATA=__PAYLOAD__;
const $=x=>document.getElementById(x);const rows=DATA.rows;
const fmt=(n,p=2)=>Number.isFinite(Number(n))?Number(n).toFixed(p):'N/A';
$('n').textContent=rows.length;$('global').textContent=rows.filter(x=>x.IFOREST_ALERT).length;$('regime').textContent=rows.filter(x=>x.REGIME_ALERT).length;$('diff').textContent=rows.filter(x=>x.IFOREST_ALERT!==x.REGIME_ALERT).length;
function allowed(r){let f=$('filter').value;return f==='all'||(f==='global'&&r.IFOREST_ALERT)||(f==='regime'&&r.REGIME_ALERT)||(f==='either'&&(r.IFOREST_ALERT||r.REGIME_ALERT))||(f==='diff'&&(r.IFOREST_ALERT!==r.REGIME_ALERT));}
function fillSelect(){let prev=$('when').value;let items=rows.filter(allowed);$('when').replaceChildren();items.forEach(r=>{let o=document.createElement('option');o.value=r.DATE;o.textContent=r.DATE;$('when').appendChild(o)});if(items.length){$('when').value=items.some(x=>x.DATE===prev)?prev:items[0].DATE;inspect();}else {$('signals').textContent='No observations match the selected filter.';$('flags').textContent='';$('explain').textContent='';}draw();}
function inspect(){let d=rows.find(r=>r.DATE===$('when').value);if(!d)return;let metrics=[['S&P 500 return',fmt(d.sp_return_pct)+'%'],['VIX change',fmt(d.vix_delta_points)+' pts'],['10Y yield change',fmt(d.yield_delta_bp)+' bp'],['IF severity',fmt(d.anomaly_severity,4)],['Cross-asset score',String(d.cross_asset_confirmation)+'/3'],['VIX regime',d.vix_regime]];$('signals').replaceChildren();metrics.forEach(([key,val])=>{let b=document.createElement('span');b.className='item';let l=document.createElement('small');l.textContent=key;let v=document.createElement('strong');v.textContent=val;b.append(l,v);$('signals').append(b)});$('flags').replaceChildren();[['Baseline',d.baseline_alert],['Global IF',d.IFOREST_ALERT],['Regime IF',d.REGIME_ALERT]].forEach(([name,on])=>{let s=document.createElement('span');s.className='flag '+(on?'yes':'no');s.textContent=name+': '+(on?'ALERT':'not flagged');$('flags').append(s)});$('explain').textContent='Same-day flags: '+d.explanation+'. Synthetic data only; these flags are not a causal explanation or future-event guarantee.';draw();}
function draw(){let svg=$('chart');svg.replaceChildren();let ns='http://www.w3.org/2000/svg';let selected=$('when').value;let w=760,h=265,m=36;let vals=rows.map(r=>r.anomaly_severity);let lo=Math.min(...vals)-0.01,hi=Math.max(...vals)+.01;let X=i=>m+i*(w-2*m)/Math.max(rows.length-1,1),Y=y=>h-m-(y-lo)/(hi-lo)*(h-2*m);function elem(tag,attrs){let e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));svg.appendChild(e);return e;}for(let q=0;q<5;q++){let yy=m+q*(h-2*m)/4;elem('line',{x1:m,x2:w-m,y1:yy,y2:yy,stroke:'#31445d','stroke-width':1});let t=elem('text',{x:5,y:yy+4,fill:'#92a7bf','font-size':11});t.textContent=fmt(hi-q*(hi-lo)/4,2)}elem('polyline',{points:rows.map((r,i)=>X(i)+','+Y(r.anomaly_severity)).join(' '),fill:'none',stroke:'#7ca5ff','stroke-width':1.8});rows.forEach((r,i)=>{if(!(r.IFOREST_ALERT||r.REGIME_ALERT||r.DATE===selected))return;let chosen=r.DATE===selected;let c=elem('circle',{cx:X(i),cy:Y(r.anomaly_severity),r:chosen?6:3.2,fill:r.IFOREST_ALERT?'#ff8f98':'#59dcc0',stroke:chosen?'white':'transparent','stroke-width':chosen?2:0,cursor:'pointer'});c.addEventListener('click',()=>{$('filter').value='all';fillSelect();$('when').value=r.DATE;inspect()});});}
function cell(tr,val){let td=document.createElement('td');td.textContent=val;tr.appendChild(td)}function buildTables(){let t=$('metrics').querySelector('tbody');DATA.metrics.filter(x=>x.split==='test'||x.split==='validation').forEach(r=>{let tr=document.createElement('tr');[r.split,r.method,r.alerts,r.proxy_hits,fmt(r.precision_vs_proxy,3)].forEach(x=>cell(tr,x));t.appendChild(tr)});let q=$('rules').querySelector('tbody');DATA.rules.slice(0,8).forEach(r=>{let tr=document.createElement('tr');[r.antecedent+' → '+r.consequent,fmt(r.support,3),fmt(r.confidence,3),fmt(r.lift,2)].forEach(x=>cell(tr,x));q.appendChild(tr)});}
$('filter').addEventListener('change',fillSelect);$('when').addEventListener('change',inspect);buildTables();fillSelect();</script></body></html>'''
    return template.replace('__PAYLOAD__', payload)


def build(evidence_dir:Path,outfile:Path)->Path:
    data=read_and_validate(evidence_dir)
    outfile=Path(outfile)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(render(data),encoding='utf-8')
    return outfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-dir',type=Path,default=Path('evidence/qa_alternating'))
    p.add_argument('--out',type=Path,default=Path('demo/CROSSWATCH_Synthetic_Research_Demo.html'))
    args=p.parse_args()
    print('Wrote:',build(args.evidence_dir,args.out))

if __name__=='__main__':main()
