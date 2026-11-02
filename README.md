# EviAlert: evidence-aware alerting with abstention and person-level false-alert guarantees

Code, frozen analysis plan, model outputs and results for the manuscript *Evidence-aware alerting with abstention and person-level false-alert guarantees for uncertain affective monitoring streams* (submitted to Information Processing & Management).

| Folder | Contents |
|---|---|
| `code/` | The EviAlert implementation, experiment scripts, analysis plan (`PREREGISTRATION.md`), exploratory analyses, figures and all result files. See `code/README.md` for instructions. |
| `infantfer_outputs/` | Out-of-fold probability outputs of the five image backbones (`runs/grouped_<model>_fold<k>.json`), the cleaned image index with identity-proxy groups (`clean_index.csv`), the grouped folds and the duplicate audit. |

To run the infant experiments with these outputs:

```bash
cd code
export INFANTFER_RESULTS=../infantfer_outputs
pip install -r requirements.txt
python run_main_v2.py resnet50
```

## Data sources and licences

* Infant images: "Cry, Laugh, or Angry?" (Hasan et al., 2025), Mendeley Data, doi:10.17632/hy969mrx9p.1, CC BY 4.0. The images themselves are not included; `clean_index.csv` refers to their file names.
* WESAD (Schmidt et al., 2018): scientific, non-commercial use only. Raw recordings and features are not included; `code/wesad/extract_features.py` regenerates the features.
* Nurse stress dataset (Hosseini et al., 2022), Dryad, doi:10.5061/dryad.5hqbzkh6f. Raw recordings and features are not included; `code/nurse/nurse_extract.py` regenerates the features.

Code is released under the MIT licence (`LICENSE`); model outputs and result files under CC BY 4.0.
