# Report validation

Prepared 9 October 2026 for Vighnesh Vijay Jadhav (31235) and Shreesh Jugade (31238), PICT Computer Engineering, DWM, academic year 2026–2027.

## Deliverables and checks

- `DWM_Mini_Project_Report.pdf`: **34 A4 pages**, 11 chapters, preliminary pages, references and two appendices. All 34 pages were rendered with Poppler and visually inspected. No overflowing tables, clipped text, blank overflow pages or detached captions were found after corrections.
- `DWM_Mini_Project_Report.docx`: editable source with A4 geometry, Times New Roman 12-point body, justified 1.5-spaced paragraphs, numbered headings, PAGE footer field and refreshable TOC with cached entries. Structural validation confirms 11 tables and 23 embedded images (17 figures plus six equations).
- The abstract contains **300 words**. There are **17 numbered figures, 11 numbered tables, six numbered equations and ten references**.
- `scripts/validate_report.py` checks PDF page boundaries and A4 size, abstract length, DOCX styles/fields/images/tables, paired benchmark coverage and the explicit full-KMeans skip. It passed. Input/output SHA-256 values are in `source/validation_checks.json`.
- `python -m pytest -q`: **25 passed in 49.36 s** during report preparation. No real-data models were trained. `ruff check .` and `ruff format --check .` passed after formatting the added scripts. The final Git whitespace check is recorded in the implementation status.

## Verified empirical evidence

| Evidence | Source | Verified finding |
|---|---|---|
| Data lineage and exclusions | `provenance.json`, `feature_statistics.csv` | 3,475,226 raw; 3,241,580 accepted; 233,646 rejected. Charges 127,550, distance 64,474, duration 41,488, speed 134. |
| Frozen configuration | `model_metadata.json`, `frozen_preprocessing.json` | K=3, seed 42, six standardized features; 100,000 final-training records and 5,000 UMAP references. |
| Selection | `k_selection.csv`, `k_profiles.csv`, `selection.json`, selection-review outputs | K=2–10 evaluated; K=3 passes declared share/silhouette/stability gates and has the strongest eligible normalized elbow. K=2 has higher silhouette. Sensitivity is disclosed. |
| Paired benchmarks | `benchmarks.csv/json` | Both methods completed 50K, 100K, 250K, 500K and 1M. At 1M: 6.6608 s versus 0.4426 s; sampled ARI 0.9873. |
| Full population | `benchmarks.csv`, `full_decision.json` | Ordinary and incremental MiniBatch completed 3,241,580 records. Full KMeans was skipped by memory preflight; there is no full KMeans fit result. |
| Batch study | `batch_study.csv` | Batch sizes 100, 500, 1,000, 5,000 and 10,000 on the same 100K population. |
| Profiles | `profiles.csv`, saved model metadata | Three model-specific profiles per estimator, each totaling 100,000 records. IDs are not treated as aligned across models. |
| Anomalies | ignored `artifacts/real/anomalies_*.parquet` | 1,000 of 100,000 training records flagged per model at its own 99th-percentile threshold; no fraud claim. |
| Projection | ignored `artifacts/real/sample_*.parquet` | Saved 2D/3D coordinates plotted directly; no UMAP fitting or transformation during report generation. |
| Real upload example | `inference_smoke.json`, saved holdout exports | 30 disjoint accepted real records passed both CSV/Parquet paths and both algorithms. Three actual output rows are retained in `source/holdout_example.json`. |
| Runtime | `runtime.json`, benchmark JSON | Windows 11 build 26200, 8 logical CPUs, 5.889 GiB usable RAM, Python 3.12.13, numerical threads 2. The intended Windows 10/8 GB target is distinguished from measured runtime. |

The benchmark metadata records historical HEAD `341b5da`, when K=3 changes were not yet committed. The verified application delivery snapshot is `a381277dd8dcadc13cc16c85f24fcdfc1850b2e2`. The report preserves this provenance distinction instead of altering old experiment records.

## Figure provenance

Figures 1–3 are diagrams of inspected implementation flows. Figures 4–11 are plots of saved selection, benchmark, profile, projection and anomaly evidence. Figure 10 includes both 2D and 3D coordinates. Figures 12–14 are actual Overview, Cluster Discovery and Algorithm Comparison captures. Figures 15–17 are **explicitly labelled placeholders** for Anomaly Explorer, Predict New Trips and Methodology; no fabricated screenshots are used. The comparison screenshot shows its selection-analysis tab, not a different benchmark view. Mathematical illustrations are separate equation assets.

## Remaining evidence and rendering limits

1. The packaged DOCX renderer was executed, but failed because `soffice.exe` was unavailable (with subsequent sandbox temporary-directory cleanup errors). **DOCX visual pagination is unverified.** The PDF was independently typeset from the same content source, not exported from Word. Open DOCX in Word, update fields and inspect pagination before submission; cached page numbers currently correspond to the PDF's explicit source layout.
2. Further browser capture was blocked by browser security policy when selecting the dashboard tab. Existing saved captures and labelled placeholders were used as the permitted alternative. The report includes a real exported prediction table, but not a verified screenshot of that upload result.
3. Full conventional KMeans and Google Colab benchmarks were **not completed**. Colab is an implemented fallback only. No unmeasured result is interpolated or shown as zero.
4. CPU model, installed physical RAM and remote CI execution were not independently established. Measured logical CPU count and OS-reported usable RAM are provided.
5. One final timing per setting, one month, overlapping selection/training samples and no ground-truth travel classes limit inference. Initialization stability is not independent-population stability. Global anomaly thresholds are not calibrated fraud probabilities.
6. Faculty names, signatures and approval dates remain placeholders. Certificate wording is conditional and does not assert institutional approval.

The official TLC source, original algorithm papers, scikit-learn documentation/JMLR article, UMAP paper/documentation and Streamlit documentation supply the reference list. References are paraphrased; no copyrighted article passages are reproduced. No application code, trained model, raw dataset or existing experimental result was changed by this documentation task.

## Reproduction

See `source/README.md` and `source/requirements-report.txt`. The committed content, figures and small result snapshots suffice to rebuild DOCX/PDF. Regenerating result figures additionally requires the existing ignored local model reference/anomaly tables; it does not require refitting models. Rendering scratch files and large inputs remain outside Git.
