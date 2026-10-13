"""Step 3: predictors, class-by-class sets, attribute-level fidelity (10 splits). Run: python 03_run_part_b.py"""
import sys, os, time
import importlib
import warnings
import pandas as pd
import ccl_helpers as H
importlib.reload(H)
warnings.filterwarnings("ignore")

t0 = time.time()
b1_all, b2_all = [], []
for name, path in H.FILES.items():
    f1 = str(H.RESULTS_DIR / f"final_B1_{name}.csv")
    f2 = str(H.RESULTS_DIR / f"final_B2_{name}.csv")
    if os.path.exists(f1) and os.path.exists(f2):
        b1, b2 = pd.read_csv(f1), pd.read_csv(f2)
    else:
        a, b = H.part_b(name, path)
        b1, b2 = pd.DataFrame(a, columns=H.B1_COLS), pd.DataFrame(b, columns=H.B2_COLS)
        b1.to_csv(f1, index=False)
        b2.to_csv(f2, index=False)
    b1_all.append(b1)
    b2_all.append(b2)
    print(name, "done after", round(time.time() - t0), "seconds", flush=True)

b1, b2 = pd.concat(b1_all), pd.concat(b2_all)
cols = ["accuracy", "bal_acc", "coverage", "min_class_cov", "answered", "acc_on_answered", "avg_size", "fidelity"]
print("\n=== B1: pooled thresholds, mean over 10 splits ===")
print(b1[b1.variant == "pooled"].groupby(["dataset", "model"])[cols].mean().round(4).to_string())
print("\n=== B1b: class-by-class thresholds ===")
print(b1[b1.variant == "classcond"].groupby(["dataset", "model"])[["coverage", "min_class_cov", "answered", "avg_size"]].mean().round(4).to_string())
print("\n=== B2: attribute-level fidelity ===")
print(b2.groupby(["dataset", "model"])[["share_defined", "top1_agree", "random_top1", "spearman"]].mean().round(4).to_string())