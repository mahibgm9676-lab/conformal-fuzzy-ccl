# Conformal answer sets and explanation fidelity for fuzzy concept-cognitive learning

Code and result files for the manuscript "Conformal answer sets and explanation fidelity for
fuzzy concept-cognitive learning" (Uma Maherswari Thathappagaari and Kaspar S), submitted to
Scientific Reports. Every number in the tables and figures of the paper comes from the csv files
in `results/`.

Archived version: Zenodo DOI **10.5281/zenodo.23791950** 

## Files
| File | What it does |
|---|---|
| `ccl_helpers.py` | Fuzzy context, class-mean pseudo-concepts, hybrid distance, membership, split conformal sets (one threshold for all classes and one per class), fidelity, singleton gate, networks, attribute-level check |
| `ua_fccl.py` | UA-FCCL: level-adaptive class-conditional calibration, clustered and grouped baselines, and the controlled explanation gate (Clopper-Pearson bound) |
| `01_prepare_data.py` | Downloads the public UCI datasets into `data/` |
| `02_run_part_a.py` | Conformal answer sets (alpha 0.05, 0.10, 0.20) and singleton gates, 20 splits |
| `03_run_part_b.py` | Forest and three networks, pooled and class-by-class sets, attribute-level fidelity, 10 splits |
| `04_check_coverage_gap.py` | Coverage gap in standard errors, and 40 splits on Wilt |
| `05_run_ua_fccl.py` | Rare-class tables, adaptive and clustered sets, and the controlled gate (20 splits, 50 for Page Blocks) |
| `notebooks/` | Colab notebooks: `UA_FCCL_final.ipynb` (step 5 on all nine datasets) and `UA_FCCL_extra_runs.ipynb` (steps 2 and 3 for Shuttle, Yeast and Ecoli) |
| `results/` | Result csv files: `final_A_*`, `final_B*`, `final_splits_*`, `final_classes_*`, `final_gate_*` and the two summary tables |
| `figures/` | Figures 1 to 3 (png, pdf, eps, tiff) |

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

The scripts skip a dataset if its result file already exists in `results/`, so the included files
let you reproduce the tables without re-running the experiments (delete a file to re-run it).
Seeds are 0-19 (part A), 0-9 (part B) and 10-39 (extra Wilt splits); Page Blocks uses 0-49 in
step 5. Use the environment variables `CCL_DATA_DIR` and `CCL_RESULTS_DIR` to change the folders.
## Explanation gate (fixed grid)

The explanation gate uses a fixed grid of thresholds. The grid is set from the training data before the calibration cases are seen. Given the thresholds, the number of unfaithful cases among the calibration cases at each level is binomial. A one-sided Clopper–Pearson upper bound with a Bonferroni correction over the 20 levels picks the largest level whose bound is at most δ. With probability at least 1 − ε, the share of unfaithful explanations among the shown cases is at most δ. This needs the training, calibration and test data to be exchangeable. The loss is disagreement with the random forest, not correctness. The guarantee is for the share among the shown cases and not for each case.

The gate results in the paper come from `ua_gate.py`, run through the notebook below. They do not come from `ua_fccl.run_all`.

### Notebooks (run each cell in order in Google Colab)

| Notebook | What it does | Output files |
|---|---|---|
| `notebooks/UA_FCCL_exact_gate.ipynb` | Simulation check of the guarantee and the fixed-grid gate on the nine datasets | `results/sim_gate_results.csv`, `results/gate2/gate2_<dataset>.csv`, `gate2_summary.csv` |
| `notebooks/UA_FCCL_sensitivity.ipynb` | Changes λ (0.25, 0.5, 0.75) and p (1.5, 2, 3) | `results/sens_<dataset>.csv`, `results/sens_summary.csv` |
| `notebooks/UA_FCCL_extra_checks.ipynb` | 100 splits for Shuttle, Yeast and Ecoli; concept-lattice test | `results/rare100_*.csv`, `results/lattice_*.csv` |
| `notebooks/UA_FCCL_conexp.ipynb` | Writes the contexts (.cxt) that ConExp opens | `results/cxt/*.cxt`, `results/cxt/concept_counts.csv` |

### Which results file feeds which table

- Table 3 and Supplementary Table S7 (gate): `results/gate2/gate2_<dataset>.csv` for the controlled gate, and the singleton-gate rows of `results/singleton_original_runs/final_gate_<dataset>.csv`.
- Supplementary Table S9 (λ and p): `results/sens_<dataset>.csv`.
- Supplementary Table S10 (100 splits): `results/rare100_<dataset>.csv`.
- Supplementary Table S11 (lattice): `results/lattice_summary.csv`.
- The concept counts in the ConExp figures: `results/cxt/concept_counts.csv`. ConExp counts the bottom concept, so its counts are one higher than `n_concepts` in `lattice_summary.csv`.

The old approximate gate (thresholds from the calibration data) is in the first release. The paper does not use it.


## Licence
MIT
