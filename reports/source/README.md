# Reproducing the academic report

`report.md` is the shared content source for the editable DOCX and the independently typeset PDF. `@@` starts a planned page; figure/table/equation directives insert evidence. The generator reads the committed result CSVs, three-row real holdout snapshot and saved figures. It never trains or transforms models.

On Windows, from the repository root, use a separate report environment so the application's numerical dependencies remain unchanged:

```powershell
py -3.12 -m venv .report-venv
.\.report-venv\Scripts\python -m pip install -r reports/source/requirements-report.txt
.\.report-venv\Scripts\python scripts/build_report.py
.\.report-venv\Scripts\python scripts/validate_report.py
```

The builder uses Windows Times New Roman font files. Other operating systems need those font paths adapted to lawfully installed fonts. All committed figures suffice to rebuild the documents. To regenerate charts from local saved models' bounded evidence, run `python scripts/report_evidence.py` using the application environment, then `python scripts/report_figures.py` using the report environment. The optional ignored `artifacts/report_dependencies` path supports the original bundled-runtime installation; otherwise normal installed packages are used.

The PDF is generated directly from the same source with ReportLab, **not converted from Word**. The PDF has 34 verified pages. DOCX has explicit page boundaries, A4 geometry, Times New Roman 12-point body text, 1.5 line spacing and a Word TOC field with a cached list. Open it in Word, select all and update fields; inspect pagination before printing. Word may paginate differently. The packaged DOCX renderer was attempted but LibreOffice was unavailable, so DOCX visual pagination has not been verified.

Figure provenance: 11 charts/diagrams come from saved measurements or inspected architecture; three images are actual dashboard captures; three are prominently labelled screenshot placeholders. Six additional images typeset equations. Existing screenshots are copied only when their destination is absent, so a verified replacement can be retained on rebuild. No figure is an AI-generated dashboard reconstruction.

`validation_checks.json` records result and deliverable hashes. The small `holdout_example.json` contains three actual previously exported public taxi records, not a synthetic example or large data snapshot. Large datasets, model binaries, render scratch files and extracted 5,000-point chart inputs remain ignored.
