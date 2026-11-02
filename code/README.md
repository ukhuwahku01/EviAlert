# EviAlert — evidence-aware, abstaining alerting with person-level false-alert guarantees

Code and results for the manuscript *Evidence-aware alerting with abstention and person-level false-alert guarantees for uncertain affective monitoring streams* (target: Information Processing & Management).

The final method (called `EviAlert-F` in the code and **EviAlert** in the manuscript) is a tempered two-state evidence filter with abstention and person-level certification of the false-alert budget. The first, windowed version (`EviAlert-W`, clipped evidence summed over 30 s) is kept as a comparator.

## Contents

| Path | What it is |
|---|---|
| `evialert/core.py` | The method: temperature scaling, prior-corrected LLR evidence, windowed accumulation (first version), two-state filter in log-odds (`fit_transitions`, `filter_logodds`), Insufficient Evidence gate, tiered status with refractory period, Poisson bound, fixed-sequence selection, person-level certification (`certify_threshold`) |
| `evialert/simulate.py` | Semi-synthetic session generator (real out-of-fold outputs, synthetic timing) |
| `evialert/experiments.py` | Baselines, ablations, the final filter with decision-focused tempering, identity-disjoint split protocol, empirical / Poisson / identity-bootstrap rules, metrics |
| `evialert/realstream.py` | Leave-one-person-out evaluation on real per-person streams, incl. person-level certification and per-person metrics |
| `run_main.py`, `run_robust.py` | First version and all comparators: 5 backbones × 20 test splits (seeds 101–120); robustness (seeds 201–210) |
| `run_main_v2.py`, `run_robust_v2.py` | Final version on the same splits → `results/main_v2_<model>.json`, `results/robust_v2_<which>.json` |
| `stats_tables.py`, `make_tables.py` | Statistics and every number of the first version (`results/numbers.txt`) |
| `figures_v2.py` | Figures 2–5 of the manuscript; Fig. 1 is `figs/fig1.dot` (Graphviz) |
| `dev/` | Development diagnostics (seeds 0–10) used to fix design choices |
| `wesad/` | WESAD (development data): feature extraction, 200-split analysis, LOPO analysis, tables and Fig. 6 |
| `nurse/` | Confirmatory test on the nurse stress dataset: `nurse_extract.py`, `run_nurse.py` (frozen), `tests_nurse.py` (H1–H3) |
| `PREREGISTRATION.md` | Analysis plan of the confirmatory test, frozen before data access, with SHA-256 fingerprints and a pre-data addendum |
| `WESAD_ekstraksi_fitur_v2.ipynb`, `Nurse_ekstraksi_fitur.ipynb` | Google Colab notebooks that download the datasets and run the feature extraction |
| `exploratory/` | Exploratory analyses (not part of H1–H3): untempered filter, person-level conformal risk control, (α, δ) grid, calibration-population size, extended baseline grid (post hoc); a faster policy implementation checked against the frozen one (`test_fast_policy.py`); Fig. 4 and Supplementary Tables S4–S8 |
| `results/` | All result files used in the manuscript (`nurse_results.json`, `nurse_tests.json`: confirmatory test; `explore_*.json`: exploratory analyses) |

## Input data

The infant scripts read the out-of-fold outputs of the InfantFER benchmark:
`<INFANTFER>/results/runs/grouped_<model>_fold{0-4}.json` and `<INFANTFER>/results/clean_index.csv`
(models: `resnet50`, `vit_b16`, `convnext_t`, `swin_t`, `effnet_b0`). By default the scripts expect this folder to sit inside the InfantFER_Experiment folder; otherwise set `export INFANTFER_RESULTS=/path/to/InfantFER_Experiment/results`.

WESAD (Schmidt et al., ICMI 2018) is distributed by its authors for scientific, non-commercial use, and the nurse stress dataset (Hosseini et al., Scientific Data 2022; Dryad doi:10.5061/dryad.5hqbzkh6f) is public. Raw recordings are not redistributed here; the notebooks regenerate the feature files.

## Reproduce

```bash
pip install -r requirements.txt
# infant sessions: comparators and first version, then the final version
for m in resnet50 vit_b16 convnext_t swin_t effnet_b0; do python run_main.py $m; python run_main_v2.py $m; done
for w in missing persist budget; do python run_robust.py $w; done
for w in missing persist; do python run_robust_v2.py $w; done
python stats_tables.py; python make_tables.py; python figures_v2.py
dot -Tpng figs/fig1.dot -o figs/fig1_framework.png

# WESAD (development data)
python3 wesad/extract_features.py /path/to/WESAD wesad_features              # folder with S2/S2.pkl ... S17/S17.pkl
python3 wesad/run_wesad.py wesad_features results/wesad_results.json 200      # 200 splits, first version + comparators
python3 wesad/run_wesad_addF.py wesad_features results/wesad_addF.json        # adds the final version to the 200 splits
python3 wesad/run_wesad_v2.py wesad_features results/wesad_v2_person.json     # leave-one-participant-out, certification
python3 wesad/make_wesad_tables.py results/wesad_results.json figs

# Nurse stress dataset (confirmatory, pre-registered)
python3 nurse/nurse_extract.py <extracted_root> <SurveyResults.xlsx> nurse_features
python3 nurse/run_nurse.py nurse_features results/nurse_results.json
python3 nurse/tests_nurse.py results/nurse_results.json results/nurse_tests.json

# Exploratory analyses (Section 6.10, Fig. 4, Supplementary Tables S4-S8)
python3 exploratory/test_fast_policy.py
for m in resnet50 vit_b16 convnext_t swin_t effnet_b0; do python3 exploratory/infant_gamma1.py $m; done
python3 exploratory/wesad_explore.py wesad_features results/explore_wesad.json
python3 exploratory/nurse_explore.py nurse_features results/explore_nurse.json
python3 exploratory/fig_tradeoff.py results/explore_wesad.json results/explore_nurse.json figs/fig4_tradeoff.png
python3 exploratory/make_supp_tables.py results/explore_wesad.json results/explore_nurse.json results/nurse_results.json results/nurse_tests.json
```

The confirmatory run takes about 3–4 h on two CPU cores and needs about 4 GB of memory; run the classifiers one after another (two parallel runs exceeded 8 GB). The exploratory scripts reproduce the frozen results exactly (checked) and are about five times faster.

Check the frozen files before the confirmatory run: `sha256sum evialert/*.py nurse/*.py` must match the fingerprints in `PREREGISTRATION.md`.

## Default settings

Window W = 6 steps (30 s at 5 s per step), sufficiency ρ = 0.5, refractory R = 24 steps (2 min), Watchlist at θ − 2 (filter) or h/2 (first version), clip c = 3 (first version only), candidate tempering γ ∈ {1, 0.5, 0.25, 0.125}, filter threshold grid linspace(−6, 18, 150), budget β = 1 false alert per hour, δ = 0.05 for pooled bounds, certification at α = 0.2, δ = 0.1 (needs ≥ 11 calibration persons). Infant sessions: 300 calibration sessions, 20 identity-bootstrap pools × 100 sessions, 600 evaluation sessions (150 h) per split.

## Notes on reproducibility

Runs are deterministic given the seeds (re-running split 101 for ConvNeXt-T reproduces the stored results exactly). The random forest uses all CPU cores, so its probabilities can differ in the last floating-point digits between machines; re-running the WESAD LOPO analysis changed one value that is not reported in the manuscript (false-alert rate of the first version at its uncertifiable threshold, random forest, natural artefacts: 0.49 vs 0.45 per hour). `results/wesad_v2.json` and `results/wesad_v2_person.json` are otherwise identical; the latter adds per-person metrics. The module docstring of `nurse/nurse_extract.py` describes an earlier time-zone rule; the frozen code uses America/Chicago (see the addendum in `PREREGISTRATION.md`).

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, scikit-learn 1.9, matplotlib 3.11.
