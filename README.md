# Conformal prediction and explanation fidelity for fuzzy concept-cognitive learning

Code and result files for the manuscript submitted to Scientific Reports
(T Maherswari). Every number in the tables and figures of the paper comes from
the csv files in `results/`.

## Files
| File | What it does |
|---|---|
| `ccl_helpers.py` | All functions: fuzzy context, class-mean pseudo-concepts, hybrid distance, membership, split conformal sets (one threshold for all classes and one per class), fidelity, trust gates, networks, attribute-level check |
| `01_prepare_data.py` | Downloads the public UCI datasets into `data/` |
| `02_run_part_a.py` | Conformal answer sets (alpha 0.05, 0.10, 0.20) and trust gates, 20 splits |
| `03_run_part_b.py` | Forest and three networks, pooled and class-by-class sets, attribute-level fidelity, 10 splits |
| `04_check_coverage_gap.py` | Coverage gap in standard errors, and 40 splits on Wilt |
| `results/` | Result csv files: `final_A_cp_*`, `final_A_gate_*`, `final_B1_*`, `final_B2_*` |

## Data
Six public UCI datasets. Shill Bidding has to be downloaded by hand, the others by script.
Shill Bidding 10.24432/C5Z611, Wilt 10.24432/C5KS4M, HTRU2 10.24432/C5DK6R,
MAGIC 10.24432/C52C8B, Page Blocks 10.24432/C5J590, Dry Bean 10.24432/C50S4B.

## How to run
1. `pip install -r requirements.txt`
2. `python 01_prepare_data.py` and put `Shill Bidding Dataset.csv` in `data/`
3. `python 02_run_part_a.py`, then `python 03_run_part_b.py`, then `python 04_check_coverage_gap.py`

The scripts skip a dataset if its result file already exists in `results/`, so the
included files let you reproduce the tables without re-running the experiments
(delete a file to re-run it). Seeds are 0-19 (part A), 0-9 (part B) and 10-39
(extra Wilt splits). Use the environment variables `CCL_DATA_DIR` and
`CCL_RESULTS_DIR` to change the folders.

## Licence
MIT
