# Ex5 / Ex6 nonlinear equivalence-aware metric patch

These configs enable:
- report_sq_mcc: true

Interpretation:
- full_mcc: raw source-coordinate recovery metric
- sq_mcc: equivalence-aware metric using MCC between squared sources
- posthoc_anchor_aligned_sq_mcc: the same metric after post-hoc ICA-anchor alignment

Recommended use:
1. Ex5: compare M1 vs M2 under centered and uncentered nonlinear benchmark.
2. Ex6: compare M2 vs M3 under centered and uncentered residual-shift benchmark.
