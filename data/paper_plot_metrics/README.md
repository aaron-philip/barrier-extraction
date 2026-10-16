# Compact paper plot data

These tables reproduce the interior-scored violin, empirical coverage, and
resolution figures without training HDF5s or access to the author's scratch.
The default export is `interior_inset0MeV`: evaluate within the measured energy
range, with no padding and no additional inset. Original training results remain
necessary to regenerate metrics for a different scoring interval.

Each of `gps/`, `gps_log/`, and `bnns_with_df/` contains 16 CSVs with the paired
curves used by PaperFigures. Barrier MAE/RMSE columns are **unnormalized**;
PaperFigures divides them by R0^2/hbaromega once at plotting time. Both posterior
mean and median scores, coverage levels, peak annotations, parameters, and
scoring bounds are retained. Posterior arrays are omitted.

`resolution_summary.csv` contains averages for the resolution comparison.
Its `barrier_MAE` and `barrier_RMSE` are already normalized per curve before averaging; do not
normalize them again. RMSE is the square root of the mean squared error, matching
the L2 violin metric. `resolution_per_curve.csv` retains the individual metrics,
actual stencil intervals, and energy-range audit fields. Its coverage aggregate
uses the same nominal levels as PaperFigures and measures the absolute deviation
of average empirical coverage from nominal coverage.

`manifest.json` records table checksums, source CSV checksums, the inset, and
calculation-code hashes. Tables are checked before loading; changing code does
not silently reinterpret a published export. The rendering notebooks do not
change these tables.

To explicitly regenerate the export on a machine with complete predictions:

```bash
python scripts/export_paper_plot_metrics.py --inset 0 \
  --full-h5-root /path/to/full/d3lambda1_rerun
```

Dataset selection and resolution-training reproducibility are described in
`notebooks/ResolutionExperiment.ipynb` and the repository README. The compact
exports supplement the existing experimental/example plot inputs; they do not
replace those inputs.
