# Faculty demonstration

1. Complete README setup. For an immediate synthetic demo, run `taxi demo`, launch Streamlit,
   and set the sidebar model directory to `artifacts/demo`. The banner labels synthetic data.
2. For empirical work, run download, clean and train commands. Use `artifacts/real`.
3. Overview shows training size; provenance separately shows full raw and cleaned counts.
4. Discovered clusters shows model-specific profiles, selectable bar charts, and 2D/3D UMAP.
   Drag the 3D chart to rotate; hover to inspect trips. IDs do not imply equivalent labels.
5. Algorithm comparison reads actual benchmark CSVs. A missing artifact gives its next command.
6. Anomalies highlights centroid-distance outliers; explain why this is not a fraud detector.
7. Predict uploaded trips: choose either saved model and upload `data/synthetic_unseen.csv`
   or `.parquet` for the synthetic demonstration. These use a different random seed from training.
   Review accepted/rejected counts, download predictions and inspect highlighted new points.
8. Add an invalid timestamp or zero distance to a copy and upload it. Inspect the rejected row
   index and reason, then download rejections. Remove a required column to demonstrate validation.
9. Download includes every accepted row, even when chart sampling leaves UMAP columns empty.
   Changing the selected algorithm uses its saved predict method; nothing is retrained.

Real model/data binaries are intentionally local. Regenerate them after a fresh clone.
No screenshots are included unless a real capture is recorded in implementation status.
