# CROSSWATCH - Source code quick start (CSC 4740/6740)

**Python:** 3.12 recommended; code QA used 3.13.5. Run these commands from the folder containing `Sources` and `Tests`.

```bash
python -m pip install -r Sources/requirements-reproduction.txt
python -m pytest -q
```

**Standalone demonstration without proprietary input files:** Open `CROSSWATCH_Synthetic_Research_Demo.html` in a browser. Its values are synthetic and MUST NOT be presented as empirical results.

**Generate a fully connected synthetic test run (separate from the real results):**

```bash
python Sources/run_integrated_synthetic_qa.py --raw-dir demo-synthetic-inputs --out-dir demo-synthetic-results
CROSSWATCH_RAW=demo-synthetic-inputs CROSSWATCH_RESULTS=demo-synthetic-results python -m streamlit run Sources/streamlit_app.py
```

**Historical research:** Requires three original, authorized local CSVs (`private-inputs/SP500.csv`, `private-inputs/VIXCLS.csv`, `private-inputs/DGS10.csv`) and private authorization/origin JSON; these are intentionally NOT in this code archive. Do not fabricate permissions.

```bash
python Sources/run_fred_historical.py --raw-dir private-inputs --out-dir private-results --rights-file docs/license_review.json --origin-file docs/origin_manifest.json --compare-features --compare-baselines
python Sources/verify_release.py --results-dir private-results --raw-dir private-inputs
python Sources/audit_report_alignment.py --results-dir private-results --raw-dir private-inputs --report-docx CROSSWATCH_IEEE_Style_Report.docx
python Sources/verify_demo_preflight.py --results-dir private-results --raw-dir private-inputs --output demo_preflight.json
CROSSWATCH_RAW=private-inputs CROSSWATCH_RESULTS=private-results python -m streamlit run Sources/streamlit_app.py
```

`private-results` is immutable. Use a new empty folder for each run. `CROSSWATCH_RAW` and `CROSSWATCH_RESULTS` are environment variables; Windows PowerShell uses `$env:CROSSWATCH_RAW="private-inputs"` and `$env:CROSSWATCH_RESULTS="private-results"`.

**Safety:** Do not upload licensed raw data, private historical daily results, or authorization emails to a public GitHub repository. The standalone HTML and synthetic runner exist for graders to inspect the interface and software behavior without proprietary data.

**Final submission gates:** Real team information, actual peer contributions, video recording/link, real Streamlit browser test, and final report PDF are still required.
