"""Controlled explanation gate with a threshold grid that is fixed before the calibration data are seen.

Why: a Clopper-Pearson bound is exact for a threshold that does not depend on the data used for the bound.
Here the G candidate thresholds are quantiles of the selector on the TRAINING data (forest confidence uses
out-of-bag probabilities), so they are fixed before calibration. Then, for every candidate g, the number of
unfaithful cases among the m_g calibration cases below the threshold is Binomial(m_g, p_g) given m_g,
where p_g = P(unfaithful | selector <= threshold_g). With a union bound over the grid, the chosen level
satisfies p_g <= delta with probability at least 1 - eps (same argument as Learn then Test, ref 42).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ccl_helpers as H
from ua_fccl import cp_upper, ecdf_rank, controlled_gate


def gate_inputs(Xtr_f, ytr, Xca_f, Xte_f, K, seed, n_trees=100):
    """Selectors (train, calibration, test) and the unfaithfulness loss (calibration, test)."""
    Bk = H.prototypes(Xtr_f, ytr, K)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1, oob_score=True).fit(Xtr_f, ytr)
    Dtr, Dca, Dte = H.rho(Xtr_f, Bk), H.rho(Xca_f, Bk), H.rho(Xte_f, Bk)
    Pca, Pte = rf.predict_proba(Xca_f), rf.predict_proba(Xte_f)
    oob = rf.oob_decision_function_
    ok = ~np.isnan(oob).any(axis=1)                      # rows that have an out-of-bag prediction
    f = lambda D: 1 - (H.membership(D) ** 2).sum(axis=1)
    f_tr, f_ca, f_te = f(Dtr)[ok], f(Dca), f(Dte)
    c_tr, c_ca, c_te = 1 - oob[ok].max(axis=1), 1 - Pca.max(axis=1), 1 - Pte.max(axis=1)
    comb = lambda ff, cc: (ecdf_rank(f_tr, ff) + ecdf_rank(c_tr, cc)) / 2
    sel = {"fuzziness": (f_tr, f_ca, f_te), "confidence": (c_tr, c_ca, c_te),
           "combined": (comb(f_tr, c_tr), comb(f_ca, c_ca), comb(f_te, c_te))}
    loss_ca = (Dca.argmin(axis=1) != Pca.argmax(axis=1)).astype(int)
    loss_te = (Dte.argmin(axis=1) != Pte.argmax(axis=1)).astype(int)
    return sel, loss_ca, loss_te


def gate_fixed_grid(s_tr, s_ca, loss_ca, s_te, delta=0.10, eps=0.1, G=20):
    """Gate with a grid fixed from the training data. Returns (accept mask for test cases, threshold or None)."""
    taus = np.quantile(s_tr, np.arange(1, G + 1) / G)
    best = None
    for tau in taus:
        acc = s_ca <= tau
        if cp_upper(int(loss_ca[acc].sum()), int(acc.sum()), eps / G) <= delta:
            best = tau
    if best is None:
        return np.zeros(len(s_te), dtype=bool), None
    return s_te <= best, best


def gate_empirical(s_cal, loss_ca, s_te, delta=0.10, G=20):
    """Baseline without any bound: the largest level whose observed calibration rate is at most delta."""
    best = None
    for g in range(1, G + 1):
        tau = np.quantile(s_cal, g / G)
        acc = s_cal <= tau
        if acc.sum() > 0 and loss_ca[acc].mean() <= delta:
            best = tau
    if best is None:
        return np.zeros(len(s_te), dtype=bool)
    return s_te <= best


def run_gates(name, Xd, yd, seeds, deltas=(0.05, 0.10, 0.20), eps=0.1, G=20, cal_fracs=(1.0,)):
    """Real-data runner: for every split, the fixed-grid gate, the older calibration-quantile gate and the
    no-bound baseline. cal_fracs < 1 keeps only that share of the calibration cases (calibration-size check)."""
    import pandas as pd
    K = len(np.unique(yd)); rows = []
    for seed in seeds:
        Xtr_f, Xca_f, Xte_f, ytr, yca, yte = H.make_split(Xd, yd, seed)
        sel, loss_ca, loss_te = gate_inputs(Xtr_f, ytr, Xca_f, Xte_f, K, seed)
        rng = np.random.RandomState(1000 + seed)
        for cf in cal_fracs:
            keep = np.sort(rng.choice(len(yca), int(round(cf * len(yca))), replace=False)) if cf < 1 else np.arange(len(yca))
            for sname, (s_tr, s_ca, s_te) in sel.items():
                for delta in deltas:
                    masks = {"fixed_grid": gate_fixed_grid(s_tr, s_ca[keep], loss_ca[keep], s_te, delta, eps, G)[0],
                             "cal_quantile": controlled_gate(s_ca[keep], loss_ca[keep], s_te, delta, eps, G),
                             "no_bound": gate_empirical(s_ca[keep], loss_ca[keep], s_te, delta, G)}
                    for gname, acc in masks.items():
                        u = loss_te[acc].mean() if acc.any() else np.nan
                        rows.append([name, seed, cf, sname, delta, gname, acc.mean(), u,
                                     float(u > delta) if acc.any() else np.nan, loss_te.mean()])
    return pd.DataFrame(rows, columns=["dataset", "seed", "cal_frac", "selector", "delta", "gate", "share",
                                       "unfaith_among_shown", "over_delta", "base_unfaith"])
