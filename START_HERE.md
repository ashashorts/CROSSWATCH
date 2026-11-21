# CROSSWATCH Fix 4 - Start here

**Current status:** historical run and report arithmetic verified; NOT READY for final grading until live Streamlit, human authorship and recorded video gates pass.

1. Read `FIX4_RELEASE_AND_DEMO_GUIDE.md` and `FIX4_HISTORICAL_RELEASE_AUDIT.pdf`.
2. Install Python 3.12, then `python -m pip install -r Sources/requirements-reproduction.txt`.
3. Run `python -m pytest -q` (117 tests expected in audit setup).
4. Put your own UNCHANGED original full-period proprietary CSVs in `private-inputs/`, keeping them private.
5. Extract the separate PRIVATE demo asset bundle into the extracted code project. Do not put its contents in a public repository or an unapproved submission archive.
6. Run `python Sources/verify_release.py --results-dir private-results --raw-dir private-inputs`.
7. Run `python Sources/audit_report_alignment.py --results-dir private-results --raw-dir private-inputs --report-docx CROSSWATCH_IEEE_Style_Report.docx`.
8. Run `python Sources/verify_demo_preflight.py --results-dir private-results --raw-dir private-inputs --output private-review/demo_preflight.json`.
9. Start `python -m streamlit run Sources/streamlit_app.py`; test all five tabs, both models and both periods.
10. Fill human-specific report details, export final PDF, record/upload 20-minute demo, verify external access.

**Permission status:** user reports real approvals held by instructor; the local software stores that as `user_attested_instructor_held` (not independent documentary verification). Do not fabricate provider emails or assert licensing rights beyond the user's actual terms.
