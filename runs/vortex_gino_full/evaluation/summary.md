# VortexGINO evaluation: overall summary

checkpoint `runs/vortex_gino_full/best_model.pt` (epoch 80), integrator rk4 with 2 substeps per saved interval

Means over the cases of each group (finite values only; n = cases contributing / cases in group). TF = teacher-forced (decoded from the TRUE particles); rollout = autoregressive from the start frame.

| metric | (a) validation cases | (b) in-regime test cases (leapfrog_N160_d1p500_g1p000, leapfrog_N320_d1p250_g1p000) | (c) out-of-regime test cases (test_outside_regime*, test_temporal_sr*) |
|---|---|---|---|
| TF grid velocity relL2 | 0.05691 (n=4/4) | 0.04674 (n=2/2) | 0.3434 (n=4/4) |
| TF grid vorticity relL2 | 0.07703 (n=4/4) | 0.04445 (n=2/2) | 0.6063 (n=4/4) |
| TF velocity error share: vortex region | 0.04172 (n=4/4) | 0.03348 (n=2/2) | 0.2481 (n=4/4) |
| TF velocity error share: far field | 0.03747 (n=4/4) | 0.03228 (n=2/2) | 0.2316 (n=4/4) |
| TF particle velocity relL2 | 0.02713 (n=4/4) | 0.02312 (n=2/2) | 0.2083 (n=4/4) |
| TF particle velocity-gradient relL2 | 0.03911 (n=4/4) | 0.02694 (n=2/2) | 0.2011 (n=4/4) |
| TF particle circulation-rate relL2 | 0.05917 (n=4/4) | 0.06256 (n=2/2) | 0.5769 (n=4/4) |
| rollout diverged (fraction of cases) | 0 (n=4/4) | 0 (n=2/2) | 0 (n=4/4) |
| rollout steps completed | 180 (n=4/4) | 180 (n=2/2) | 180 (n=4/4) |
| rollout final position error / R0 | 1.33 (n=4/4) | 1.13 (n=2/2) | 2.262 (n=4/4) |
| persistence baseline / R0 | 9.084 (n=4/4) | 9.37 (n=2/2) | 4.411 (n=4/4) |
| rollout final circulation relL2 | 1.312 (n=4/4) | 1.293 (n=2/2) | 1.168 (n=4/4) |
| rollout final total-|Γ| drift vs truth | 0.1288 (n=4/4) | 0.1813 (n=2/2) | 0.05841 (n=4/4) |
| rollout final impulse drift vs truth | 1.836 (n=4/4) | 1.227 (n=2/2) | 0.6973 (n=4/4) |
| rollout final grid velocity relL2 | 0.9856 (n=4/4) | 1.056 (n=2/2) | 1.017 (n=4/4) |

## Cases per group

* (a) validation cases: leapfrog_N160_d0p750_g0p750, leapfrog_N160_d2p000_g0p500, leapfrog_N320_d0p750_g1p000, leapfrog_N640_d0p750_g1p000
* (b) in-regime test cases (leapfrog_N160_d1p500_g1p000, leapfrog_N320_d1p250_g1p000): leapfrog_N160_d1p500_g1p000, leapfrog_N320_d1p250_g1p000
* (c) out-of-regime test cases (test_outside_regime*, test_temporal_sr*): test_outside_regime__free_pair_N320_sep5p000_gm1p000, test_outside_regime__leapfrog_N320_d2p500_g0p250, test_outside_regime__toroidal_R1p500_g20p000_dx0p500, test_temporal_sr__perturbed_N320_amp0p100_mode3

Excluded from the averages above (still in metrics.json / summary.csv):
* test_spatial_sr__leapfrog_spatial_N320_d1p000_g1p000: seen IC: field super-resolution check only (same initial condition as the training case leapfrog_N320_d1p000_g1p000; only the output grid is finer)

## What the metrics mean

* **TF grid velocity relL2** (`tf_u_rel_l2`): error of the velocity field decoded from the TRUE particles, relative to the field's size (0 = perfect, 1 = as large as the field itself).
* **TF grid vorticity relL2** (`tf_w_rel_l2`): same for the vorticity (curl of the predicted velocity gradient); harder than velocity because vorticity is concentrated in thin cores.
* **TF velocity error share: vortex region** (`tf_u_err_vortex_frac_of_total`): part of the full-grid velocity relL2 that comes from the 5% of nodes with the strongest vorticity.
* **TF velocity error share: far field** (`tf_u_err_far_frac_of_total`): part of the full-grid velocity relL2 that comes from the remaining 95% of nodes (squares of the two shares add up to the square of the full relL2).
* **TF particle velocity relL2** (`tf_particle_u_rel_l2`): error of the velocity the model predicts AT the particles, which is what moves them in a rollout.
* **TF particle velocity-gradient relL2** (`tf_particle_J_rel_l2`): error of the predicted velocity gradient at the particles (includes the self-induced term).
* **TF particle circulation-rate relL2** (`tf_particle_gamma_rate_rel_l2`): error of the predicted rate of change of each particle's circulation (vortex stretching), which drives the circulation in a rollout.
* **rollout diverged (fraction of cases)** (`ro_diverged`): share of rollouts that blew up (non-finite values, Σ|Γ| off its start value by > 50% or ring spread > 3x its start value); final rollout values below are averaged over the remaining cases only.
* **rollout steps completed** (`rollout_steps_completed`): how many saved-frame intervals the model was rolled out before stopping or diverging.
* **rollout final position error / R0** (`ro_final_pos_err_over_R`): mean distance between predicted and true particles at the end of the rollout, in units of the ring radius (0.1 = one tenth of a ring radius off).
* **persistence baseline / R0** (`persistence_final_pos_err_over_R`): the same error if the particles had simply stayed where they started; the model must be well below this.
* **rollout final circulation relL2** (`ro_final_gamma_rel_l2`): error of the predicted particle circulation vectors at the end of the rollout.
* **rollout final total-|Γ| drift vs truth** (`ro_final_sum_abs_gamma_drift`): relative difference between predicted and true total circulation magnitude (conservation check).
* **rollout final impulse drift vs truth** (`ro_final_impulse_drift`): difference between predicted and true linear impulse, relative to the rings' impulse (conservation check; the impulse of an inviscid flow is constant).
* **rollout final grid velocity relL2** (`ro_final_field_u_rel_l2`): velocity-field error when the field is decoded from the model's OWN rolled-out particles.
