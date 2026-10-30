"""Step 5: rare-class tables and gate comparison with ua_fccl.py (20 splits, 50 for Page Blocks).
Run: python 05_run_ua_fccl.py
Needs the authors' clustered-conformal code (Ding et al. 2023):
  git clone https://github.com/tiffanyding/class-conditional-conformal
  export CCL_DING_DIR=/path/to/class-conditional-conformal
Writes final_splits_*, final_classes_*, final_gate_* and the two summary tables to results/.
"""
import os
import pandas as pd
import ccl_helpers as H
import ua_fccl as U

data = {n: H.load(p) for n, p in H.FILES.items()}
seeds_for = {"pageblocks": range(50)}
for n, (X, y) in data.items():
    f = H.RESULTS_DIR / f"final_splits_{n}.csv"
    if f.exists():
        print(n, "already done, skipping"); continue
    s, c, g, fails = U.run_all(n, X, y, seeds_for.get(n, range(20)))
    s.to_csv(f, index=False)
    c.to_csv(H.RESULTS_DIR / f"final_classes_{n}.csv", index=False)
    g.to_csv(H.RESULTS_DIR / f"final_gate_{n}.csv", index=False)
    print(n, "done | methods that failed to run:", fails or "none")
names = list(data)
rd = lambda k: pd.concat([pd.read_csv(H.RESULTS_DIR / f"final_{k}_{n}.csv") for n in names])
U.summarise_module1(rd("splits"), rd("classes")).to_csv(H.RESULTS_DIR / "final_tableA_rare_classes.csv")
U.summarise_gate(rd("gate")).to_csv(H.RESULTS_DIR / "final_tableB_gate.csv")
