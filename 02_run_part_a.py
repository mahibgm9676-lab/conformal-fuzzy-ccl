"""Step 2: conformal answer sets and trust gates (20 splits). Run: python 02_run_part_a.py"""
import sys, os, time
import importlib
import pandas as pd
import ccl_helpers as H
importlib.reload(H)

t0 = time.time()
cp_all, gate_all = [], []
for name, path in H.FILES.items():
    f_cp = str(H.RESULTS_DIR / f"final_A_cp_{name}.csv")
    f_g = str(H.RESULTS_DIR / f"final_A_gate_{name}.csv")
    if os.path.exists(f_cp) and os.path.exists(f_g):
        cp, g = pd.read_csv(f_cp), pd.read_csv(f_g)
    else:
        a, b = H.part_a(name, path)
        cp, g = pd.DataFrame(a, columns=H.CP_COLS), pd.DataFrame(b, columns=H.GATE_COLS)
        cp.to_csv(f_cp, index=False)
        g.to_csv(f_g, index=False)
    cp_all.append(cp)
    gate_all.append(g)
    print(name, "done after", round(time.time() - t0), "seconds", flush=True)

cp, gt = pd.concat(cp_all), pd.concat(gate_all)
print("\n=== A1: alpha = 0.1, mean over 20 splits ===")
print(cp[cp.alpha == 0.10].groupby(["dataset", "score", "variant"])[H.METRIC_COLS].mean().round(4).to_string())
print("\n=== A2: coverage by alpha over all dataset/score rows ===")
cov = cp.groupby(["dataset", "score", "variant", "alpha"])["coverage"].mean().reset_index()
print(cov.groupby("alpha")["coverage"].agg(["min", "mean", "max"]).round(4))
print("\n=== A3: gates and error detection ===")
print(gt.groupby("dataset")[["share", "fid_all", "fid_conformal", "fid_confidence", "fid_fuzziness",
                             "fid_random", "auc_fuzziness", "auc_distance"]].mean().round(4).to_string())