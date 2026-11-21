# CROSSWATCH Fix 4 - Exact historical-run and demo procedures

## A. Project and data layout

Extract the code ZIP. All commands run from its `CROSSWATCH/` directory (containing `Sources/`, `Tests/`, and the report). Place the exact original uploaded files as `private-inputs/SP500.csv`, `private-inputs/VIXCLS.csv`, and `private-inputs/DGS10.csv`; CSV headers may use FRED `observation_date`. These are not included in shareable ZIPs. Keep the external-approval evidence with your instructor as you have described; the separate private demo bundle contains only the user-attested status, not the emails.

## B. Python environment and tests

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r Sources/requirements-reproduction.txt
python -m pytest -q
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Sources/requirements-reproduction.txt
python -m pytest -q
```

Expected: 117/117 tests. A different result requires diagnosis; do not silently adjust model or dependency versions. Python 3.13.5 was tested in this audit, but Python 3.12 and live Streamlit await local verification.

## C. Choose restore or fresh run

**Restore** (quickest for demo): Extract the separate PRIVATE historical demo assets into the same project directory, preserving `private-results/`, `private-review/` and `docs/license_review.json`, `docs/origin_manifest.json`. Copy the original three CSVs into `private-inputs/`. Do not modify the restored results. Run the checks in Part D.

**Recompute**: Keep the private authorization/provenance JSON in `docs/`, and ensure `private-results/` does not exist. Then execute:

```bash
python Sources/inspect_private_inputs.py --raw-dir private-inputs --out-file private-review/inspect_again.json
python Sources/run_fred_historical.py --raw-dir private-inputs --out-dir private-results --rights-file docs/license_review.json --origin-file docs/origin_manifest.json --compare-features --compare-baselines
```

All 3 source hashes must match their provenance records. If your FRED downloads differ, a fresh provenance review and full model rebuild are necessary. No scraping, FRED API or automated download is used.

## D. Independent verification

```bash
python Sources/verify_release.py --results-dir private-results --raw-dir private-inputs --report private-review/release_verification.json
python Sources/audit_report_alignment.py --results-dir private-results --raw-dir private-inputs --report-docx CROSSWATCH_IEEE_Style_Report.docx --output private-review/report_alignment.json
python Sources/verify_demo_preflight.py --results-dir private-results --raw-dir private-inputs --output private-review/demo_preflight.json
python Sources/release_preflight.py --project-root . --raw-dir private-inputs --results-dir private-results --report-docx CROSSWATCH_IEEE_Style_Report.docx --output private-review/release_preflight.json
```

The last command must still report `False` until authentic team/video and live-browser requirements have been met. Do not use `--confirm-live-demo-tested`, `--confirm-report-pdf-reviewed`, or `--confirm-video-access-tested` until you actually perform those tasks.

## E. Live browser and notebook

```bash
python -m streamlit run Sources/streamlit_app.py
```

Confirm initial page loads without integrity errors. Select **3 daily changes**, then **5 inputs incl. trailing volatility and VIX level**, and for each select **Validation** and **Test**. Check all five tabs: Case explorer; Market context; Model & baseline comparison; Apriori rules; Provenance & checks. Choose at least three dates across case categories, including a global-only or context-only alert if available. Explain five-session outcomes as retrospective proxy observations, not forecasts. Check that the two model variants use the same forward proxy and 370 test-scoring dates. Scroll all tables; check chart labels and plots at normal presentation zoom.

To open Jupyter:

```bash
python -m jupyter notebook CROSSWATCH_Historical_Walkthrough.ipynb
```

Run All and confirm there are no tracebacks. This audit executed all four code cells of the notebook, but in a Python 3.13.5 environment.

## F. 20-minute proposed video storyboard

| Segment | Time | What to show and explain |
| --- | --- | --- |
| Introduction and question | 0:00-2:00 | Purpose, three indicators, and why unusual joint behavior matters. Not a trading system. |
| Source quality and provenance | 2:00-4:00 | Valid date window, missing values, 2,466 aligned days, user-attested permission and private raw-data policy. |
| Feature design / training split | 4:00-6:15 | S&P daily return, VIX point change, yield bp change, trailing volatility, VIX level, chronological 70/15/15 fitting. |
| Model comparison | 6:15-9:00 | Three-feature vs five-feature forest, global vs VIX-conditioned cutoffs; explain thresholds calibrated on training. |
| Case explorer | 9:00-12:00 | Three distinct dates; five-panel plots, 0-3 confirmation, following five-session outcome. |
| Baseline and outcome table | 12:00-14:45 | Equity tail, fixed and rolling z-score, forest, fixed operating point vs retrospective matched alert budgets. |
| Apriori rules | 14:45-17:00 | Training association rules and held-out support/confidence/lift; correlation/co-occurrence not causation. |
| Integrity / source hashes | 17:00-18:30 | Run ID, 52 artifacts, private CSV hashes, why matching hashes do not prove legal permission. |
| Concluding findings | 18:30-20:00 | Limitations, no universally superior method, results, lessons learned, and team role. |

Record presenter face and commentary, not only a screen recording. Verify all five dashboard tabs and functionality before recording. If rights permit only classroom use, keep the Drive video access restricted to the instructor, not public/unlisted sharing beyond the grant scope.

## G. Last human release gates

- [ ] Team number and real member names entered in final report.
- [ ] Accurate individual contributions, lessons learned, failures and team experiences entered.
- [ ] Actual effort/participation/contribution peer evaluations out of ten entered.
- [ ] Real narrated video recorded (~20 min) with visible presenter; all project functions shown.
- [ ] Video uploaded to Drive and shared with instructor; outsider/instructor-access test completed.
- [ ] Real video HTTPS URL included in report; report exported PDF within 5-10 pages and all pages inspected.
- [ ] `release_preflight.py` reports all gates PASS after truthful local confirmations.
- [ ] Only permitted code and allowable derivatives submitted; raw proprietary CSVs/emails not placed in public ZIP/GitHub.

**Science caveats:** Data rights are user-attested; historical anomaly scores are not crisis predictions; forward loss is an imperfect evaluation proxy; later test inspection precludes describing that period as pristine untouched validation; a run ID may change on rerun due to metadata timestamps even with identical statistical results.
