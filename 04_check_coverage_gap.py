"""Step 4: how far below the 0.90 target the pooled coverage falls, in standard errors;
then 30 extra splits (seeds 10-39) on Wilt. Run: python 04_check_coverage_gap.py"""
import pandas as pd, numpy as np
import ccl_helpers as H
names = ["shill", "wilt", "htru2", "magic", "pageblocks", "drybean"]
b1 = pd.concat([pd.read_csv(str(H.RESULTS_DIR / f"final_B1_{n}.csv")) for n in names])
x = b1.groupby(["dataset", "model", "variant"])["coverage"].agg(["mean", "std", "count"])
x["se"] = x["std"] / np.sqrt(x["count"])
x["gap_in_se"] = (0.90 - x["mean"]) / x["se"]
print(x.round(4).sort_values("gap_in_se", ascending=False).head(8))

# ---- extra splits on Wilt ----
import warnings
import numpy as np, pandas as pd
import ccl_helpers as H
warnings.filterwarnings("ignore")

a, b = H.part_b("wilt", H.FILES["wilt"], seeds=range(10, 40))
extra = pd.DataFrame(a, columns=H.B1_COLS)
extra.to_csv(str(H.RESULTS_DIR / "final_B1_wilt_extra.csv"), index=False)

old = pd.read_csv(str(H.RESULTS_DIR / "final_B1_wilt.csv"))
allw = pd.concat([old, extra])
x = allw.groupby(["model", "variant"])["coverage"].agg(["mean", "std", "count"])
x["se"] = x["std"] / np.sqrt(x["count"])
x["gap_in_se"] = (0.90 - x["mean"]) / x["se"]
print(x.round(4))