# Analysis plan for the confirmatory evaluation of EviAlert v2 (frozen before access to the confirmatory data)

Date: 2026-10-29 (UTC). Status at writing: the nurse stress dataset (Hosseini et al., 2022) had **not** been accessed by the analyst; no feature file, label or result from it had been seen. All design choices below were made on the development data only (InfantFER development splits, seeds 0–10; WESAD, which therefore serves as development data).

## 1. Method (EviAlert v2), frozen

* Frame evidence: temperature-scaled class probabilities (temperature fitted by NLL on calibration persons), prior-corrected log-likelihood ratio ℓ_t = logit(q_t) − logit(π_cal), **no clipping** (probabilities bounded to [1e-6, 1 − 1e-6]).
* Accumulation: two-state hidden Markov filter in log-odds. Onset and exit hazards a, b estimated from the calibration persons' target-state sequences (add-one smoothing). Evidence enters with a tempering exponent γ ∈ {1, 0.5, 0.25, 0.125}, chosen on calibration persons as the value with the highest mean recall at 0.5, 1 and 2 false alerts per hour ("decision-focused tempering"). Unusable frames receive only the prediction step.
* Abstention: Insufficient Evidence when fewer than ρ = 0.5 of the last W = 6 steps are usable; no alert decisions in that state.
* Statuses: High Alert when posterior log-odds ≥ θ; Watchlist when ≥ θ − 2; refractory R = 24 steps; threshold grid θ ∈ linspace(−6, 18, 150).
* Person-level budget certification: for each threshold, X = number of calibration persons whose false-alert rate exceeds β = 1 per hour; p = P(Binomial(n, α) ≤ X) with α = 0.2; fixed-sequence testing from the most conservative threshold at δ = 0.1. If the most conservative threshold fails, the budget is reported as not certifiable.
* Comparators: Frame, MajVote, MovAvg, SelMovAvg, MovAvg-Cal, EviAlert v1 (windowed sum, c = 3), EviAlert v1 without clipping. Threshold rules: empirical rate, Poisson bound, identity bootstrap (B = 200, δ = 0.05), certification.

## 2. Confirmatory data and pre-processing

* Data: nurse stress dataset (15 nurses, Empatica E4, hospital shifts). Features computed with `nurse/nurse_extract.py` every 5 s over the preceding 60 s (EDA, skin temperature, acceleration, heart rate). Survey times are converted from America/Chicago local time.
* Labels: confirmed stress (survey stress level 1 or 2) = target; confirmed no stress (level 0) and unlabelled time = non-target; unvalidated events (na) are excluded from false-alert counting and monitored time.
* Alert-worthy events: runs of target frames of at least 45 s (gaps ≤ 60 s merged), per recording session; grace period 5 min after each event.
* Usable frames: SD of acceleration magnitude over the last 5 s ≤ 0.25 g, with g estimated as the participant's median acceleration magnitude. Natural artefacts only (no simulated dropouts).
* Base classifiers: logistic regression, random forest (300 trees, min_samples_leaf = 5) and histogram gradient boosting, trained leave-one-nurse-out on frames labelled 0, 1 or 2, subsampled every 6th frame.
* Protocol: leave-one-nurse-out; everything except the base classifier is fitted on the 14 calibration nurses.

## 3. Hypotheses and tests

* **H1 (detection).** Per-nurse recall of stress events at ≤ 1 false alert per hour (each held-out nurse's own operating curve) is higher for EviAlert v2 than for MovAvg-Cal. Two-sided Wilcoxon signed-rank test over nurses with at least one event, for each classifier; Holm correction over the three classifiers.
* **H2 (stale alerts).** Per-nurse stale alerts per 100 h at the bootstrap-selected threshold are lower for EviAlert v2 than for MovAvg-Cal. Same test and correction.
* **H3 (budget).** At the certified threshold, the share of held-out nurses whose false-alert rate is ≤ 1 per hour is at least 1 − α = 0.8. Reported with an exact binomial 95% CI, together with the certification rate, for each classifier; compared descriptively with the empirical, Poisson and bootstrap rules.
* Secondary (descriptive): pooled and per-nurse recall at 0.5 and 2 false alerts per hour, deployed recall and false-alert rate, delay, Insufficient Evidence share, v2 versus v1.
* All results will be reported whatever their direction. Deviations from this plan will be listed in the manuscript.

## 4. Code fingerprints (SHA-256)

| File | SHA-256 |
|---|---|
| `evialert/core.py` | `35017a9d222268905e57b5e6cd40a80c9ece41c2ad7017911e32057aec44c701` |
| `evialert/experiments.py` | `3bc47036d18c14f8f8abffd98b4229b4be5bcd9426238665d16846fb9887440d` |
| `evialert/realstream.py` | `401a5734d4e239c76158a78ee5e09b8581688c376d240055249a818ff9a60c1e` |
| `evialert/simulate.py` | `b13523e4667e774809233c4eded653d486e846d76f8b7f5bca10df230cd28600` |
| `nurse/nurse_extract.py` | `71fb45da87054030d8c36af43755496acbf7a50a713a5e41b5bdfbb97240034c` |
| `nurse/run_nurse.py` | `07ec3a9bee593b5d9a34378d115ff848c2fb91a6056350ff25989cf1923733f0` |

## 5. Addendum (written before access to the confirmatory data)

Date: 2026-10-29 (UTC). The nurse data had still not been accessed by the analyst.

* **Test script.** `nurse/tests_nurse.py` implements H1–H3 exactly as stated above. It was dry-run on the WESAD leave-one-participant-out output (`results/wesad_v2_person.json`) to check that it runs; those WESAD numbers are development results and play no role in the confirmatory test. SHA-256: `12c27a26c89c371e42325c6294829fa987b8dc02386e5d0a4856f0dc8ed58868`.
* **Two clarifications of points the plan left open.** (i) H3: a held-out nurse whose fold cannot be certified enters the within-budget share at the most conservative threshold, which is what the certified rule deploys in that case; the certification rate is reported alongside. (ii) H1 and H2 use the nurses with at least one alert-worthy event (H2 refers to "the same test"); differences equal to zero are dropped (Wilcoxon's zero method), and a classifier with no non-zero difference is reported with p = 1.
* **Known stale docstring.** The module docstring of `nurse/nurse_extract.py` still describes an earlier alignment rule (choosing the time-zone offset with most events inside recordings). The code, which was frozen above, uses America/Chicago local time and only reports the other offsets as a check. The docstring was left unchanged so that the fingerprint stays valid.

## 6. Addendum 2: exploratory analyses (not part of H1–H3)

Date: 2026-10-29 (UTC). Written after the nurse feature file had been received and the frozen confirmatory run (`nurse/run_nurse.py`) had been started, but before any of its results had been inspected; only descriptive counts (hours, sessions and label counts per nurse, and the alignment check) had been printed. The following analyses are exploratory, are reported separately from H1–H3 and do not change the confirmatory analysis:

* EviAlert v2 without tempering (γ fixed at 1).
* Person-level conformal risk control with the loss 1{false-alert rate > β}: the most liberal threshold, walking from the most conservative one, with (X + 1)/(n + 1) ≤ α (marginal guarantee).
* Sensitivity of certification to α ∈ {0.1, 0.2, 0.3} and δ ∈ {0.05, 0.1, 0.2}, and to the number of calibration persons (random subsets of 5, 7, 9, 11 and all 14; other fitted quantities unchanged).

Code: `exploratory/lopo_curves.py`, `exploratory/nurse_explore.py` (the same functions were first run on WESAD with `exploratory/wesad_explore.py`, and the untempered filter on the infant test splits with `exploratory/infant_gamma1.py`).

## 7. Addendum 3: execution notes and post hoc analysis (written after the confirmatory results had been inspected)

Date: 2026-10-29 (UTC).

* **Execution.** `nurse/run_nurse.py` and `nurse/tests_nurse.py` were run unchanged (fingerprints re-checked immediately before the run): `python3 nurse/run_nurse.py nurse_features results/nurse_results.json` and `python3 nurse/tests_nurse.py results/nurse_results.json results/nurse_tests.json`. A second process started to compute the gradient-boosting part in parallel was stopped by the container's memory limit and its output was not used; all reported results come from the single planned run.
* **Post hoc sensitivity analysis.** After inspecting the results we noticed that the probability grid of the moving-average baselines (0.30–0.995, fixed on the infant data where the target class made up half of the images) did not reach low enough for a target that covers about 6% of the nurses' frames. With the weakly separating logistic regression, the calibrated moving average therefore could not approach the false-alert budgets from below, which favours EviAlert in H1. We added a calibrated moving average whose grid is extended downwards (log-spaced from 0.005; a superset of the original grid) and repeated the H1 comparison with it. This analysis is post hoc and is reported as such; the confirmatory results are reported unchanged.
