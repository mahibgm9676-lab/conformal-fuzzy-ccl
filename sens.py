"""Sensitivity of UA-FCCL to the two fixed constants: lambda (weight of Euclidean vs cosine distance) and p (fuzzifier).
For every split the forest is fitted once. For each (lambda, p) the pseudo-concept distances, the memberships,
the class-conditional sets with the adaptive rule, and the controlled gate (fixed grid) are recomputed.
Measured: rare-class coverage and overall coverage of the adaptive sets (distance score, alpha = 0.1),
agreement of the nearest pseudo-concept with the forest, and the gate (combined selector, delta = 0.10)."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import ccl_helpers as H
from ua_fccl import sets_adaptive, ecdf_rank
from ua_gate import gate_fixed_grid


def run_sens(name, Xd, yd, seeds, lams=(0.25, 0.5, 0.75), ps=(1.5, 2.0, 3.0), alpha=0.1, delta=0.10, eps=0.1, G=20):
    K = len(np.unique(yd)); rare = int(np.bincount(yd).argmin()); rows = []
    for seed in seeds:
        Xtr, Xca, Xte, ytr, yca, yte = H.make_split(Xd, yd, seed)
        Bk = H.prototypes(Xtr, ytr, K)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1, oob_score=True).fit(Xtr, ytr)
        Pca, Pte, oob = rf.predict_proba(Xca), rf.predict_proba(Xte), rf.oob_decision_function_
        okr = ~np.isnan(oob).any(axis=1)
        c_tr, c_ca, c_te = 1 - oob[okr].max(1), 1 - Pca.max(1), 1 - Pte.max(1)
        for lam in lams:
            Dtr, Dca, Dte = H.rho(Xtr, Bk, lam), H.rho(Xca, Bk, lam), H.rho(Xte, Bk, lam)
            sc = np.array([np.median(Dtr[ytr == k, k]) for k in range(K)])      # same scaling rule as the main runs
            S = sets_adaptive(Dca / sc, yca, Dte / sc, alpha)
            hit = S[np.arange(len(yte)), yte]; m = yte == rare
            loss_ca = (Dca.argmin(1) != Pca.argmax(1)).astype(int)
            loss_te = (Dte.argmin(1) != Pte.argmax(1)).astype(int)
            for p in ps:
                f = lambda D: 1 - (H.membership(D, p) ** 2).sum(axis=1)
                f_tr, f_ca, f_te = f(Dtr)[okr], f(Dca), f(Dte)
                comb = lambda ff, cc: (ecdf_rank(f_tr, ff) + ecdf_rank(c_tr, cc)) / 2
                acc, _ = gate_fixed_grid(comb(f_tr, c_tr), comb(f_ca, c_ca), loss_ca, comb(f_te, c_te), delta, eps, G)
                u = loss_te[acc].mean() if acc.any() else np.nan
                rows.append([name, seed, lam, p, hit.mean(), hit[m].sum(), m.sum(), 1 - loss_te.mean(),
                             acc.mean(), u, float(u > delta) if acc.any() else np.nan])
    return pd.DataFrame(rows, columns=["dataset", "seed", "lam", "p", "coverage", "rare_hit", "rare_n", "agree_forest",
                                       "gate_share", "gate_unfaith", "gate_over_delta"])


def summarise(df):
    g = df.groupby(["dataset", "lam", "p"]).agg(coverage=("coverage", "mean"), rare_hit=("rare_hit", "sum"),
            rare_n=("rare_n", "sum"), agree=("agree_forest", "mean"), gate_share=("gate_share", "mean"),
            gate_unfaith=("gate_unfaith", "mean"), over=("gate_over_delta", "mean"))
    g["rare_cov"] = g.rare_hit / g.rare_n
    return g.drop(columns=["rare_hit", "rare_n"]).round(3)
