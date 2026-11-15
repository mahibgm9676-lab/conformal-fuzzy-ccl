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
