# Planned experiment configuration manifest

The following table lists all configuration files in the narrative order used by the manuscript. The `config` paths are relative to `configs/`.

| Order | Experiment | Section | Method label | X-axis variable | Config file | Description |
|---:|---|---|---|---|---|---|
| 0 | Exp1-control | main | B0 | fixedmix=0.0 | `experiments/exp1_vanilla_failure/exp1_00_fixedmix_sanity.yaml` | Sanity check: vanilla iVAE under fixed mixing (no cross-domain drift). |
| 1 | Exp1 | main | B0 | drift_strength=0.0 | `experiments/exp1_vanilla_failure/exp1_01_vanilla_linear_ds0p0.yaml` | Vanilla iVAE under linear cross-domain mixing drift. |
| 2 | Exp1 | main | B0 | drift_strength=0.05 | `experiments/exp1_vanilla_failure/exp1_02_vanilla_linear_ds0p05.yaml` | Vanilla iVAE under linear cross-domain mixing drift. |
| 3 | Exp1 | main | B0 | drift_strength=0.1 | `experiments/exp1_vanilla_failure/exp1_03_vanilla_linear_ds0p1.yaml` | Vanilla iVAE under linear cross-domain mixing drift. |
| 4 | Exp1 | main | B0 | drift_strength=0.2 | `experiments/exp1_vanilla_failure/exp1_04_vanilla_linear_ds0p2.yaml` | Vanilla iVAE under linear cross-domain mixing drift. |
| 5 | Exp1 | main | B0 | drift_strength=0.4 | `experiments/exp1_vanilla_failure/exp1_05_vanilla_linear_ds0p4.yaml` | Vanilla iVAE under linear cross-domain mixing drift. |
| 100 | Exp2 | main | B1 | drift_strength=0.0 | `experiments/exp2_linear_adapter_effectiveness/exp2_100_anchoronly_linear_ds0p0.yaml` | Anchor-only control on the linear drift benchmark. |
| 101 | Exp2 | main | M1 | drift_strength=0.0 | `experiments/exp2_linear_adapter_effectiveness/exp2_101_m1_linear_adapter_ds0p0.yaml` | First-order session/domain linear drift alignment on the linear benchmark. |
| 102 | Exp2 | main | B1 | drift_strength=0.05 | `experiments/exp2_linear_adapter_effectiveness/exp2_102_anchoronly_linear_ds0p05.yaml` | Anchor-only control on the linear drift benchmark. |
| 103 | Exp2 | main | M1 | drift_strength=0.05 | `experiments/exp2_linear_adapter_effectiveness/exp2_103_m1_linear_adapter_ds0p05.yaml` | First-order session/domain linear drift alignment on the linear benchmark. |
| 104 | Exp2 | main | B1 | drift_strength=0.1 | `experiments/exp2_linear_adapter_effectiveness/exp2_104_anchoronly_linear_ds0p1.yaml` | Anchor-only control on the linear drift benchmark. |
| 105 | Exp2 | main | M1 | drift_strength=0.1 | `experiments/exp2_linear_adapter_effectiveness/exp2_105_m1_linear_adapter_ds0p1.yaml` | First-order session/domain linear drift alignment on the linear benchmark. |
| 106 | Exp2 | main | B1 | drift_strength=0.2 | `experiments/exp2_linear_adapter_effectiveness/exp2_106_anchoronly_linear_ds0p2.yaml` | Anchor-only control on the linear drift benchmark. |
| 107 | Exp2 | main | M1 | drift_strength=0.2 | `experiments/exp2_linear_adapter_effectiveness/exp2_107_m1_linear_adapter_ds0p2.yaml` | First-order session/domain linear drift alignment on the linear benchmark. |
| 108 | Exp2 | main | B1 | drift_strength=0.4 | `experiments/exp2_linear_adapter_effectiveness/exp2_108_anchoronly_linear_ds0p4.yaml` | Anchor-only control on the linear drift benchmark. |
| 109 | Exp2 | main | M1 | drift_strength=0.4 | `experiments/exp2_linear_adapter_effectiveness/exp2_109_m1_linear_adapter_ds0p4.yaml` | First-order session/domain linear drift alignment on the linear benchmark. |
| 200 | Exp3 | main | M1-u | aux_mode=u | `experiments/exp3_aux_ablation/exp3_200_m1_aux_u_linear_ds0p2.yaml` | Aux-routing ablation under linear drift with first-order alignment enabled. |
| 201 | Exp3 | main | M1-z | aux_mode=z | `experiments/exp3_aux_ablation/exp3_201_m1_aux_z_linear_ds0p2.yaml` | Aux-routing ablation under linear drift with first-order alignment enabled. |
| 202 | Exp3 | main | M1-joint | aux_mode=joint | `experiments/exp3_aux_ablation/exp3_202_m1_aux_joint_linear_ds0p2.yaml` | Aux-routing ablation under linear drift with first-order alignment enabled. |
| 203 | Exp3 | main | M1-none | aux_mode=none | `experiments/exp3_aux_ablation/exp3_203_m1_aux_none_linear_ds0p2.yaml` | Aux-routing ablation under linear drift with first-order alignment enabled. |
| 300 | Exp4 | main | B0 | subject_nonlinear_strength=0.0 | `experiments/exp4_first_order_limit_nonlinear/exp4_300_b0_hier_nonlinear_nls0p0.yaml` | Vanilla iVAE on the hierarchical nonlinear benchmark. |
| 301 | Exp4 | main | B1 | subject_nonlinear_strength=0.0 | `experiments/exp4_first_order_limit_nonlinear/exp4_301_b1_anchoronly_hier_nonlinear_nls0p0.yaml` | Anchor-only control on the hierarchical nonlinear benchmark. |
| 302 | Exp4 | main | M1 | subject_nonlinear_strength=0.0 | `experiments/exp4_first_order_limit_nonlinear/exp4_302_m1_linearonly_hier_nonlinear_nls0p0.yaml` | First-order session alignment only on the hierarchical nonlinear benchmark. |
| 303 | Exp4 | main | B0 | subject_nonlinear_strength=0.1 | `experiments/exp4_first_order_limit_nonlinear/exp4_303_b0_hier_nonlinear_nls0p1.yaml` | Vanilla iVAE on the hierarchical nonlinear benchmark. |
| 304 | Exp4 | main | B1 | subject_nonlinear_strength=0.1 | `experiments/exp4_first_order_limit_nonlinear/exp4_304_b1_anchoronly_hier_nonlinear_nls0p1.yaml` | Anchor-only control on the hierarchical nonlinear benchmark. |
| 305 | Exp4 | main | M1 | subject_nonlinear_strength=0.1 | `experiments/exp4_first_order_limit_nonlinear/exp4_305_m1_linearonly_hier_nonlinear_nls0p1.yaml` | First-order session alignment only on the hierarchical nonlinear benchmark. |
| 306 | Exp4 | main | B0 | subject_nonlinear_strength=0.2 | `experiments/exp4_first_order_limit_nonlinear/exp4_306_b0_hier_nonlinear_nls0p2.yaml` | Vanilla iVAE on the hierarchical nonlinear benchmark. |
| 307 | Exp4 | main | B1 | subject_nonlinear_strength=0.2 | `experiments/exp4_first_order_limit_nonlinear/exp4_307_b1_anchoronly_hier_nonlinear_nls0p2.yaml` | Anchor-only control on the hierarchical nonlinear benchmark. |
| 308 | Exp4 | main | M1 | subject_nonlinear_strength=0.2 | `experiments/exp4_first_order_limit_nonlinear/exp4_308_m1_linearonly_hier_nonlinear_nls0p2.yaml` | First-order session alignment only on the hierarchical nonlinear benchmark. |
| 309 | Exp4 | main | B0 | subject_nonlinear_strength=0.3 | `experiments/exp4_first_order_limit_nonlinear/exp4_309_b0_hier_nonlinear_nls0p3.yaml` | Vanilla iVAE on the hierarchical nonlinear benchmark. |
| 310 | Exp4 | main | B1 | subject_nonlinear_strength=0.3 | `experiments/exp4_first_order_limit_nonlinear/exp4_310_b1_anchoronly_hier_nonlinear_nls0p3.yaml` | Anchor-only control on the hierarchical nonlinear benchmark. |
| 311 | Exp4 | main | M1 | subject_nonlinear_strength=0.3 | `experiments/exp4_first_order_limit_nonlinear/exp4_311_m1_linearonly_hier_nonlinear_nls0p3.yaml` | First-order session alignment only on the hierarchical nonlinear benchmark. |
| 312 | Exp4 | main | B0 | subject_nonlinear_strength=0.4 | `experiments/exp4_first_order_limit_nonlinear/exp4_312_b0_hier_nonlinear_nls0p4.yaml` | Vanilla iVAE on the hierarchical nonlinear benchmark. |
| 313 | Exp4 | main | B1 | subject_nonlinear_strength=0.4 | `experiments/exp4_first_order_limit_nonlinear/exp4_313_b1_anchoronly_hier_nonlinear_nls0p4.yaml` | Anchor-only control on the hierarchical nonlinear benchmark. |
| 314 | Exp4 | main | M1 | subject_nonlinear_strength=0.4 | `experiments/exp4_first_order_limit_nonlinear/exp4_314_m1_linearonly_hier_nonlinear_nls0p4.yaml` | First-order session alignment only on the hierarchical nonlinear benchmark. |
| 500 | Exp5 | appendix | M2 | subject_nonlinear_strength=0.0 | `experiments/exp5_high_order_drift_alignment/exp5_500_m2_highorder_hier_nonlinear_nls0p0.yaml` | High-order subject nonlinear alignment on the hierarchical nonlinear benchmark. |
| 501 | Exp5 | appendix | M2 | subject_nonlinear_strength=0.1 | `experiments/exp5_high_order_drift_alignment/exp5_501_m2_highorder_hier_nonlinear_nls0p1.yaml` | High-order subject nonlinear alignment on the hierarchical nonlinear benchmark. |
| 502 | Exp5 | appendix | M2 | subject_nonlinear_strength=0.2 | `experiments/exp5_high_order_drift_alignment/exp5_502_m2_highorder_hier_nonlinear_nls0p2.yaml` | High-order subject nonlinear alignment on the hierarchical nonlinear benchmark. |
| 503 | Exp5 | appendix | M2 | subject_nonlinear_strength=0.3 | `experiments/exp5_high_order_drift_alignment/exp5_503_m2_highorder_hier_nonlinear_nls0p3.yaml` | High-order subject nonlinear alignment on the hierarchical nonlinear benchmark. |
| 504 | Exp5 | appendix | M2 | subject_nonlinear_strength=0.4 | `experiments/exp5_high_order_drift_alignment/exp5_504_m2_highorder_hier_nonlinear_nls0p4.yaml` | High-order subject nonlinear alignment on the hierarchical nonlinear benchmark. |
| 600 | Exp6 | appendix | M2 | conditional_shift_strength=0.02 | `experiments/exp6_full_model_with_moment_alignment/exp6_600_m2_shiftbench_shift0p02.yaml` | High-order alignment without same-U_raw moment alignment under residual conditional shift. |
| 601 | Exp6 | appendix | M3 | conditional_shift_strength=0.02 | `experiments/exp6_full_model_with_moment_alignment/exp6_601_m3_fullmodel_shift0p02.yaml` | Complete model with same-U_raw moment alignment under residual conditional shift. |
| 602 | Exp6 | appendix | M2 | conditional_shift_strength=0.05 | `experiments/exp6_full_model_with_moment_alignment/exp6_602_m2_shiftbench_shift0p05.yaml` | High-order alignment without same-U_raw moment alignment under residual conditional shift. |
| 603 | Exp6 | appendix | M3 | conditional_shift_strength=0.05 | `experiments/exp6_full_model_with_moment_alignment/exp6_603_m3_fullmodel_shift0p05.yaml` | Complete model with same-U_raw moment alignment under residual conditional shift. |
| 604 | Exp6 | appendix | M2 | conditional_shift_strength=0.1 | `experiments/exp6_full_model_with_moment_alignment/exp6_604_m2_shiftbench_shift0p1.yaml` | High-order alignment without same-U_raw moment alignment under residual conditional shift. |
| 605 | Exp6 | appendix | M3 | conditional_shift_strength=0.1 | `experiments/exp6_full_model_with_moment_alignment/exp6_605_m3_fullmodel_shift0p1.yaml` | Complete model with same-U_raw moment alignment under residual conditional shift. |

## Suggested manuscript placement

- Main text: Exp1, Exp2, Exp3, and Exp4.
- Appendix: Exp5 and Exp6.
- Reuse the Exp1 B0 curve when comparing Exp2.
- Reuse the Exp4 M1 curve when comparing Exp5.
- Compare M2 and M3 under the same benchmark for Exp6.