# CROSSWATCH — Professor review package (not final submission)

**Project:** CROSSWATCH — Context-Aware Market Anomaly Detection  
**Course:** CSC 4740/6740 Data Mining, Georgia State University  
**Named authors in current report:** Dev Gajjar and Asha  

This ZIP contains the complete **source-code project**, the unit tests, a runnable synthetic quality-assurance workflow, the historical demonstration notebook, methodology and integration documentation, and an **editable IEEE-style report** in `CROSSWATCH_IEEE_Style_Report.docx`.

## Important current submission status

- **Report is a review draft, NOT the final submitted paper.** The team number, exact individual learning outcomes/contributions and instructor-accessible final demonstration video link remain to be entered truthfully before final submission. The author names are as supplied in the conversation. Confirm that the team roster is complete before sending.
- **Environment:** Python 3.12. The previously released requirements had a dependency conflict (`pandas==2.2.3` versus `mlxtend==0.25.0`). This handoff changes the pandas pin to `pandas==2.3.3`. The change concerns installation only; the scientific Python source files and historical reported results have not been changed here. A local environment revalidation is recommended.
- **Data use:** This ZIP excludes the proprietary original S&P 500 and Cboe VIX CSVs, historical per-day results, private approval records, and the private demo-assets bundle. The user reports academic licenses are held by the instructor. Do not put licensed raw data on GitHub or send them through a channel not authorized by the licensing terms.
- **Reproducibility:** A synthetic workflow is available without licensed historical inputs; reproducing the archived historical run requires the three authorized local CSV files and the separately held private manifest/attestation records.

## Quick start (Mac/Linux)

Open a terminal in the `CROSSWATCH` directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Sources/requirements-reproduction.txt
python -m pip check
python -m pytest -q
```

For the historical dashboard, add your **authorized local** inputs and private historical-results bundle according to `START_HERE.md` and `FIX4_RELEASE_AND_DEMO_GUIDE.md` and launch:

```bash
python -m streamlit run Sources/streamlit_app.py
```

Then open `http://localhost:8501`. The integrated release verifier is:

```bash
python Sources/verify_release.py --results-dir private-results --raw-dir private-inputs
```

**Never confuse the synthetic QA run with the historical results displayed in the paper.**
