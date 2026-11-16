# Conformal answer sets and explanation fidelity for fuzzy concept-cognitive learning

Code and result files for the manuscript "Conformal answer sets and explanation fidelity for
fuzzy concept-cognitive learning" (Uma Maherswari Thathappagaari and Kaspar S), submitted to
Scientific Reports. Every number in the tables and figures of the paper comes from the csv files
in this repository.

Archived version: Zenodo DOI  **10.5281/zenodo.24279487** (version 2.0.1; the DOI of all versions together is on the Zenodo page)

All files are in the main folder of the repository.

## Files
| File | What it does |
|---|---|
| `ccl_helpers.py` | Fuzzy context, class-mean pseudo-concepts, hybrid distance, membership, split conformal sets (one threshold for all classes and one per class), fidelity, singleton gate, networks, attribute-level check |
| `ua_fccl.py` | UA-FCCL: level-adaptive class-conditional calibration, clustered and grouped baselines, and the first version of the explanation gate (thresholds from the calibration data) |
| `ua_gate.py` | The explanation gate used in the paper: fixed grid of thresholds from the training data, Clopper-Pearson bound with a Bonferroni correction |
| `sim_gate.py` | Simulation check of the gate guarantee |
| `sens.py` | Sensitivity to the distance weight and the fuzziness parameter |
| `extra.py` | 100-split check and the concept-lattice test |
| `cxt_export.py` | Writes the formal contexts (.cxt files) that ConExp opens |
| `lattice_count.py` | Counts the formal concepts of the thresholded contexts (same counts as ConExp, bottom concept included) |
| `01_prepare_data.py` | Downloads the public UCI datasets into `data/` |
| `02_run_part_a.py` | Conformal answer sets (alpha 0.05, 0.10, 0.20) and singleton gates, 20 splits |
| `03_run_part_b.py` | Forest and three networks, pooled and class-by-class sets, attribute-level fidelity, 10 splits |
| `04_check_coverage_gap.py` | Coverage gap in standard errors, and 40 splits on Wilt |
| `05_run_ua_fccl.py` | Rare-class tables, adaptive and clustered sets, and the first version of the gate (20 splits, 50 for Page Blocks) |
| `UA_FCCL_final.ipynb`, `UA_FCCL_extra_runs.ipynb` | Colab notebooks for step 5 on all nine datasets, and steps 2 and 3 for Shuttle, Yeast and Ecoli |
| `UA_FCCL_exact_gate.ipynb`, `UA_FCCL_sensitivity.ipynb`, `UA_FCCL_extra_checks.ipynb`, `UA_FCCL_conexp.ipynb` | Colab notebooks for the gate in the paper and the added checks (see below) |
| `final_*.csv` | Result files of steps 2 to 5 (`final_A_*`, `final_B*`, `final_splits_*`, `final_classes_*`, `final_gate_*`, `final_tableA_rare_classes.csv`) |
| `gate2_*.csv`, `sim_gate_results.csv`, `sens_*.csv`, `rare100_*.csv`, `lattice_*.csv`, `concept_counts.csv`, `concept_counts_all.csv`, `*.cxt` | Result files of the added checks (see below) |
| `Fig1_*`, `Fig2_*`, `Fig3_*` | Figures 1 to 3 (png, pdf, eps, tiff) |

## Data
Nine public UCI datasets. Shill Bidding has to be downloaded by hand, the others by script.
Shill Bidding 10.24432/C5Z611, Wilt 10.24432/C5KS4M, HTRU2 10.24432/C5DK6R,
MAGIC 10.24432/C52C8B, Page Blocks 10.24432/C5J590, Dry Bean 10.24432/C50S4B,
Statlog (Shuttle) 10.24432/C5WS31, Yeast 10.24432/C5KG68, Ecoli 10.24432/C5388M.
For Shuttle, Yeast and Ecoli, classes with fewer than 8 cases are dropped (done by `01_prepare_data.py`).

## How to run
1. `pip install -r requirements.txt`
2. `python 01_prepare_data.py` and put `Shill Bidding Dataset.csv` in `data/`
3. `python 02_run_part_a.py`, then `python 03_run_part_b.py`, then `python 04_check_coverage_gap.py`
4. For step 5, clone the clustered-conformal code of Ding et al. (2023):
   `git clone https://github.com/tiffanyding/class-conditional-conformal`, set
   `CCL_DING_DIR` to that folder, then run `python 05_run_ua_fccl.py`.

The scripts skip a dataset if its result file already exists, so the included files
let you reproduce the tables without re-running the experiments (delete a file to re-run it).
Seeds are 0-19 (part A), 0-9 (part B) and 10-39 (extra Wilt splits); Page Blocks uses 0-49 in
step 5. Use the environment variables `CCL_DATA_DIR` and `CCL_RESULTS_DIR` to change the folders.

## Explanation gate (fixed grid)

The explanation gate in the paper uses a fixed grid of thresholds. The grid is set from the training data before the calibration cases are seen. Given the thresholds, the number of unfaithful cases among the calibration cases at each level is binomial. A one-sided Clopper-Pearson upper bound with a Bonferroni correction over the 20 levels picks the largest level whose bound is at most delta. With probability at least 1 - epsilon, the share of unfaithful explanations among the shown cases is at most delta. This needs the training, calibration and test data to be exchangeable. The loss is disagreement with the random forest, not correctness. The guarantee is for the share among the shown cases and not for each case.

The gate results in the paper come from `ua_gate.py`, run through `UA_FCCL_exact_gate.ipynb`. They do not come from `ua_fccl.py` or `05_run_ua_fccl.py`, which hold the first version of the gate (thresholds from the calibration data). The paper does not use that version.

### Notebooks for the added checks (run each cell in order in Google Colab)

| Notebook | What it does | Output files |
|---|---|---|
| `UA_FCCL_exact_gate.ipynb` | Simulation check of the guarantee and the fixed-grid gate on the nine datasets | `sim_gate_results.csv`, `gate2_<dataset>.csv`, `gate2_summary.csv` |
| `UA_FCCL_sensitivity.ipynb` | Changes lambda (0.25, 0.5, 0.75) and p (1.5, 2, 3) | `sens_<dataset>.csv`, `sens_summary.csv` |
| `UA_FCCL_extra_checks.ipynb` | 100 splits for Shuttle, Yeast and Ecoli; concept-lattice test | `rare100_*.csv`, `lattice_*.csv` |
| `UA_FCCL_conexp.ipynb` | Writes the contexts that ConExp opens | `*.cxt`, `concept_counts.csv` |
| `UA_FCCL_lattice_counts.ipynb` | Writes the contexts for all nine datasets at θ = 0.3, 0.5 and 0.7 and counts their concepts | `*.cxt`, `concept_counts_all.csv` |

### Which results file feeds which table

- Table 3 and Supplementary Table S7 (gate): `gate2_<dataset>.csv` for the controlled gate, and the singleton-gate rows of `final_gate_<dataset>.csv`. The other rows of `final_gate_<dataset>.csv` come from the first version of the gate and are not used.
- Supplementary Table S9 (lambda and p): `sens_<dataset>.csv`, 20 splits for every dataset, including Page Blocks.
- Supplementary Table S11 (lattice): agreement values from `lattice_summary.csv`; concept counts from `concept_counts_all.csv`. The counts are from split 0 and include the bottom concept, as ConExp does.
- The concept counts of the two ConExp figures (Supplementary Figs. S1 and S2): `concept_counts.csv` and `concept_counts_all.csv`.

## Licence
MIT
