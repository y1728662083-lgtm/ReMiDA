
# Ex5 locating experiments

These configs are designed to localize why the nonlinear branch fails.

- exp5loc_710_m2_centered_nls0p2_anchorloss0_keepicainit.yaml
  Turn off anchor loss only, but keep ICA-derived W_init. This isolates whether the *loss_anchor* term is harmful while preserving the previous linear warm start.

- exp5loc_711_m2_centered_nls0p2_noanchor_identityinit.yaml
  Remove both ICA anchor loss and ICA-based initialization. This tests whether the nonlinear branch is being misled by linear-ICA artifacts.

- exp5loc_712_m2_centered_nls0p2_noanchor_identityinit_batchmoment.yaml
  Same as 711, but enable the same-U_raw cross-subject batch moment loss while setting the actual moment transform to identity (eta_mean=eta_var=0). This adds a direct alignment training signal without changing the frontend geometry at inference time.

- exp5loc_713_m2_centered_nls0p1_noanchor_identityinit_batchmoment.yaml
  Same as 712 but with weaker subject_nonlinear_strength=0.1. This checks whether there exists a working regime for M2 when the benchmark difficulty is reduced.

- exp5loc_714_m1_centered_nls0p1_reference.yaml
  M1 reference under the same weaker nonlinear setting, used to compare against 713.
