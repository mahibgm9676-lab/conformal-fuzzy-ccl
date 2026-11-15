"""Simulation: does each gate keep the TRUE unfaithful rate among accepted cases at most delta with probability >= 1 - eps?
Data come from a Gaussian mixture, so fresh calibration samples and a huge test population are available."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import ccl_helpers as H
from ua_gate import gate_inputs, gate_fixed_grid, gate_empirical
from ua_fccl import controlled_gate
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def make_sampler(K, d, sep, probs, seed):
    r = np.random.RandomState(seed)
    mu = r.randn(K, d) * sep
    def sample(n, rng):
        y = rng.choice(K, size=n, p=probs)
        return mu[y] + rng.randn(n, d), y
    return sample

def one_setting(label, K, d, sep, probs, sseed, n_cals, R, deltas, eps=0.1, G=20, n_pop=60000):
    rng = np.random.RandomState(sseed)
    sample = make_sampler(K, d, sep, probs, sseed)
    Xtr, ytr = sample(6000, rng)
    sc = MinMaxScaler().fit(Xtr)
    f = lambda X: np.clip(sc.transform(X), 0, 1)
    Xtr_f = f(Xtr)
    Xpop, ypop = sample(n_pop, rng); Xpop_f = f(Xpop)
    Xc_big, yc_big = sample(max(n_cals) * R, rng)         # fresh calibration data for every repetition
    Xc_big_f = f(Xc_big)
    sel_all, loss_c_all, loss_pop = gate_inputs(Xtr_f, ytr, Xc_big_f, Xpop_f, K, sseed)
    rows = []
    for n in n_cals:
        for r in range(R):
            idx = np.arange(r * n, (r + 1) * n)           # disjoint calibration samples
            for sname, (s_tr, s_ca, s_pop) in sel_all.items():
                for delta in deltas:
                    masks = {"fixed_grid": gate_fixed_grid(s_tr, s_ca[idx], loss_c_all[idx], s_pop, delta, eps, G)[0],
                             "cal_quantile": controlled_gate(s_ca[idx], loss_c_all[idx], s_pop, delta, eps, G),
                             "no_bound": gate_empirical(s_ca[idx], loss_c_all[idx], s_pop, delta, G)}
                    for g, acc in masks.items():
                        p = loss_pop[acc].mean() if acc.any() else np.nan
                        rows.append([label, n, r, sname, delta, g, acc.mean(), p, float(p > delta) if acc.any() else 0.0])
    return pd.DataFrame(rows, columns=["setting", "n_cal", "rep", "selector", "delta", "gate", "share", "true_risk", "violation"])

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = []
    # (label, K, d, sep, class probabilities, seed)
    for lab, K, d, sep, pr, sd in [("easy, 3 classes", 3, 6, 1.5, [.5, .3, .2], 1), ("hard, 3 classes", 3, 6, 0.7, [.5, .3, .2], 2),
                                   ("hard, 2 classes", 2, 5, 0.5, [.8, .2], 3)]:
        out.append(one_setting(lab, K, d, sep, pr, sd, n_cals=(100, 300, 1000), R=R, deltas=(0.05, 0.10, 0.20)))
        print(lab, "done", flush=True)
    df = pd.concat(out); df.to_csv("sim_gate_results.csv", index=False)
    t = df.groupby(["setting", "n_cal", "gate"]).agg(violation_rate=("violation", "mean"), share=("share", "mean")).round(3)
    print(t.to_string())
    print("\nOVERALL violation rate by gate (target <= 0.10):")
    print(df.groupby("gate").violation.mean().round(3).to_string())
