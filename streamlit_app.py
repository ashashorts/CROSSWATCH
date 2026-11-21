"""Local-only CROSSWATCH Phase 3 dashboard: read one manifest-verified run."""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from crosswatch import SERIES, read_fred_csv
from historical_cases import LABELS, VARIANTS, case_figure, load_cases, matching_rules, select_examples
from release_integrity import read_manifest, ReleaseIntegrityError

st.set_page_config(page_title='CROSSWATCH | Verified Research', layout='wide')
st.title('CROSSWATCH | Explainable Market Anomaly Detection')
st.caption('Retrospective research prototype; anomaly alerts and five-session loss outcomes are not trading signals or forecasts.')

results_dir = Path(os.environ.get('CROSSWATCH_RESULTS', 'private-results'))
raw_dir = Path(os.environ.get('CROSSWATCH_RAW', 'private-inputs'))
try:
    manifest = read_manifest(results_dir, raw_dir=raw_dir)
except (OSError, KeyError, ValueError, AssertionError, TypeError) as exc:
    st.error(f'Unable to verify run and original inputs. {exc}')
    st.code('python Sources/run_fred_historical.py --raw-dir private-inputs --out-dir private-results --compare-features --compare-baselines')
    st.stop()

if manifest['run_kind'] == 'synthetic_integration_qa':
    st.warning('SYNTHETIC QA DATA ONLY — not historical market evidence.')
else:
    st.info('Locally authenticated to the saved input hashes. Licensing scope is researcher-attested, not independently certified by this app.')
st.caption(f"Run ID: {manifest['run_id']}  |  Data: {manifest['research_window']['start']}–{manifest['research_window']['end']}  |  Source: {manifest['source_label']}")
if not manifest['includes_statistical_baselines']:
    st.warning('Statistical baselines are unavailable in this run. Regenerate using the complete Fix 2 runner.')

left, right = st.columns(2)
variant = left.selectbox('Forest feature set', VARIANTS, index=1,
                         format_func=lambda value: '3 daily changes' if value=='three_features' else '5 inputs incl. trailing volatility and VIX level')
split = right.selectbox('Evaluation partition', ('test','validation'), format_func=str.title)
try:
    frame, meta, rules, closes = load_cases(results_dir, raw_dir, variant, split)
except (OSError, KeyError, ValueError, AssertionError, TypeError) as exc:
    st.error(f'Cannot load selected model and partition: {exc}')
    st.stop()

explore, markets, compare, associations, provenance = st.tabs(
    ['Case explorer', 'Market context', 'Model & baseline comparison', 'Apriori rules', 'Provenance & checks']
)

with explore:
    examples = select_examples(frame)
    st.caption('Case categories are selected from the chosen scoring partition, not optimized using future outcomes.')
    choice = st.selectbox('Example category', ('Choose any scored date', *examples))
    dates = frame.index[::-1].tolist()
    suggested = examples.get(choice, dates[0])
    date = st.selectbox('Scored date', dates, index=dates.index(suggested),
                        format_func=lambda d: d.strftime('%b %d, %Y'),
                        key=f'{manifest["run_id"]}:{variant}:{split}:{choice}')
    row = frame.loc[date]
    proxy = row['forward_proxy']
    a, b, c = st.columns(3)
    a.metric('Isolation Forest severity', f"{row['anomaly_severity']:.3f}")
    b.metric('Cross-asset confirmation', f"{int(row['cross_asset_confirmation'])}/3")
    c.metric('Future five-session loss proxy', 'Not evaluable' if pd.isna(proxy) else 'Met' if bool(proxy) else 'Not met')
    st.caption(f"VIX context: {row['vix_regime']} · Flag explanation: {row['explanation']}")
    shown = [(col, title) for col, title in LABELS.items() if col in row]
    st.dataframe(pd.DataFrame({'Method':[title for _,title in shown],
                               'Alert':['Yes' if bool(row[col]) else 'No' for col,_ in shown]}),
                 hide_index=True, use_container_width=True)
    fig = case_figure(frame, meta, closes, date)
    st.pyplot(fig)
    plt.close(fig)
    st.caption('The first four panels use aligned market observations. The last uses all equity sessions in the selected partition. This is a retrospective outcome, not a predicted return.')
    st.subheader('Training-discovered rule antecedents matching this date')
    selected = matching_rules(row, rules, split)
    if selected.empty:
        st.caption('No training-discovered antecedents matched the selected date.')
    else:
        st.dataframe(selected, hide_index=True, use_container_width=True)

with markets:
    st.caption('Source hashes for all THREE private CSVs were verified before any market chart loaded.')
    labels = {'SP500':'S&P 500 index close (index points)',
              'VIXCLS':'VIX close (index points)',
              'DGS10':'10-year constant maturity yield (%)'}
    for series in SERIES:
        values = read_fred_csv(series, raw_dir, allow_download=False).dropna()
        values = values.loc[frame.index[0]:frame.index[-1]]
        st.subheader(labels[series])
        st.line_chart(values, height=185)
    st.subheader('Alerts on selected partition')
    choices = {col:title for col,title in LABELS.items() if col in frame.columns}
    alert_method = st.selectbox('Filter by alert method', list(choices), format_func=choices.__getitem__)
    st.dataframe(frame.loc[frame[alert_method].astype(bool),
                ['anomaly_severity','cross_asset_confirmation','forward_proxy']],
                use_container_width=True)

with compare:
    st.caption('Fixed operating thresholds are set using training data. Top-5% matched budgets rank scores retrospectively and are NOT deployable thresholds.')
    root_tables = (
        ('feature_comparison_operating.csv','Feature-set comparison: operating cutoffs'),
        ('feature_comparison_budget.csv','Feature-set comparison: matched alert budget'),
        ('episode_coverage.csv','Proxy episode coverage'),
        ('pairwise_alert_overlap.csv','Pairwise overlap / Jaccard'),
    )
    for filename, heading in root_tables:
        st.subheader(heading)
        df = pd.read_csv(results_dir / filename)
        df = df.loc[df['split'].eq(split)]
        if df.empty:
            st.warning(f'No {split} rows available in {filename}')
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)
    folder = results_dir / manifest['variants'][variant]['directory']
    label = manifest['source_label']
    st.subheader(f'Operating proxy diagnostics — {variant.replace("_"," ")}')
    st.dataframe(pd.read_csv(folder / f'{label}_operating_proxy_diagnostics.csv').query('split == @split'),
                 hide_index=True, use_container_width=True)

with associations:
    st.caption('Apriori patterns are mined on training dates; support, confidence and lift describe co-occurrence, not causality or explanations of model internals.')
    st.dataframe(rules.loc[rules['split'].eq(split)],hide_index=True,use_container_width=True)
    st.subheader('Per-model rule verification')
    label=manifest['source_label']
    p=results_dir/variant/f'{label}_training_rule_holdout_checks.csv'
    audit_rules=pd.read_csv(p)
    st.dataframe(audit_rules.loc[audit_rules['split'].eq(split)],hide_index=True,use_container_width=True)

with provenance:
    st.subheader('Verified run integrity')
    st.write({'run_id':manifest['run_id'],'schema_version':manifest['schema_version'],
              'run_kind':manifest['run_kind'],'date_window':manifest['research_window'],
              'total_fingerprinted_artifacts':len(manifest['artifacts']),
              'input_sha256':manifest['input_sha256']})
    st.caption('The hashes identify local input files. They do not independently certify data licensing or original provider authenticity. Full provider approvals remain private.')
    st.subheader('Training/validation/test partitions')
    st.dataframe(pd.DataFrame([{'partition':n, **v} for n,v in meta['splits'].items()]),
                 hide_index=True,use_container_width=True)
    st.info('If a source CSV, model CSV, rule table or figure changes, refresh/rebuild the entire run; the manifest intentionally rejects stale or altered evidence.')
