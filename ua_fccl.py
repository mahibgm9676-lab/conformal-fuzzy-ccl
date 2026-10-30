"""UA-FCCL pilot: two new modules on top of ccl_helpers.py.

Module 1  sets_grouped      concept-grouped class-conditional calibration (rare classes)
Module 2  controlled_gate   explanation gate with a guarantee on unfaithful explanations

Needs ccl_helpers.py in the same folder.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier
import ccl_helpers as H


# ---------------------------------------------------------------- Module 1
def class_scale(Xtr_f, ytr, Bk):
    """Median training distance of each class to its own pseudo-concept (puts scores on one scale)."""
    D = H.rho(Xtr_f, Bk)
    return np.array([np.median(D[ytr == k, k]) for k in range(len(Bk))])


def sets_grouped(score_cal, y_cal, score_te, Bk, alpha=0.1):
    """A class with at least n_min calibration cases keeps its own threshold.
    A smaller class borrows the calibration scores of the classes whose pseudo-concepts
    are nearest (distance rho between pseudo-concepts) until the group has n_min cases."""
    K = score_cal.shape[1]
    n_min = int(np.ceil((1 - alpha) / alpha))
    counts = np.bincount(y_cal, minlength=K)
    Dpc = H.rho(Bk, Bk)
    np.fill_diagonal(Dpc, np.inf)
    cols = []
    for k in range(K):
        group, total = [k], counts[k]
        for j in np.argsort(Dpc[k]):
            if total >= n_min or not np.isfinite(Dpc[k, j]):
                break
            group.append(j)
            total += counts[j]
        s = np.sort(np.concatenate([score_cal[y_cal == g, g] for g in group]))
        n = len(s)
        r = int(np.ceil((n + 1) * (1 - alpha)))
        q = np.inf if (n == 0 or r > n) else s[r - 1]
        cols.append(score_te[:, k] <= q)
    return np.stack(cols, axis=1)


def sets_adaptive(score_cal, y_cal, score_te, alpha=0.1):
    """Class-conditional sets that never use q = infinity.
    A class with n calibration cases gets the level alpha_k = max(alpha, 1/(n+1)), so its
    own rank j = ceil((n+1)(1-alpha_k)) always exists (j <= n). Its coverage is then at least
    1 - alpha_k. Only the class's own data are used, so it also works for two-class data."""
    K = score_cal.shape[1]
    cols = []
    for k in range(K):
        s = np.sort(score_cal[y_cal == k, k])
        n = len(s)
        if n == 0:
            cols.append(np.ones(len(score_te), dtype=bool))
            continue
        a_k = max(alpha, 1.0 / (n + 1))
        r = int(np.ceil((n + 1) * (1 - a_k) - 1e-12))
        cols.append(score_te[:, k] <= s[min(r, n) - 1])
    return np.stack(cols, axis=1)


def shrink_rare(y_cal, rare, n_rare, seed):
    """Keep only n_rare calibration cases of the rare class (simulates scarce calibration data)."""
    if n_rare is None:
        return np.arange(len(y_cal))
    rng = np.random.RandomState(seed)
    ir = np.where(y_cal == rare)[0]
    if len(ir) <= n_rare:
        return np.arange(len(y_cal))
    keep_r = rng.choice(ir, n_rare, replace=False)
    return np.sort(np.concatenate([np.where(y_cal != rare)[0], keep_r]))


M1_COLS = ["dataset", "seed", "n_rare_cal", "score", "variant", "coverage", "rare_cov",
           "min_class_cov", "answered", "avg_size"]


def pilot_module1(name, Xd, yd, seeds=range(20), alpha=0.1, rare_counts=(None, 12, 8, 6, 4)):
    K = len(np.unique(yd))
    rare = int(np.bincount(yd).argmin())
    rows = []
    for seed in seeds:
        Xtr_f, Xca_f, Xte_f, ytr, yca, yte = H.make_split(Xd, yd, seed)
        Bk = H.prototypes(Xtr_f, ytr, K)
        sc = class_scale(Xtr_f, ytr, Bk)
        Dca, Dte = H.rho(Xca_f, Bk), H.rho(Xte_f, Bk)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr_f, ytr)
        scores = {"distance": (Dca / sc, Dte / sc),
                  "membership": (1 - H.membership(Dca), 1 - H.membership(Dte)),
                  "forest": (1 - rf.predict_proba(Xca_f), 1 - rf.predict_proba(Xte_f))}
        for n_rare in rare_counts:
            keep = shrink_rare(yca, rare, n_rare, seed)
            n_have = int((yca[keep] == rare).sum())
            for sname, (s_cal, s_te) in scores.items():
                for variant in ["pooled", "classcond", "grouped", "adaptive"]:
                    if variant == "pooled":
                        S = H.sets_pooled(s_cal[keep], yca[keep], s_te, alpha)
                    elif variant == "classcond":
                        S = H.sets_class(s_cal[keep], yca[keep], s_te, alpha)
                    elif variant == "adaptive":
                        S = sets_adaptive(s_cal[keep], yca[keep], s_te, alpha)
                    else:
                        S = sets_grouped(s_cal[keep], yca[keep], s_te, Bk, alpha)
                    m = H.set_metrics(S, yte, K)
                    rc = S[yte == rare, rare].mean() if (yte == rare).any() else np.nan
                    rows.append([name, seed, n_have, sname, variant, m["coverage"], rc,
                                 m["min_class_cov"], m["answered"], m["avg_size"]])
    return pd.DataFrame(rows, columns=M1_COLS)


# ---------------------------------------------------------------- Module 2
def cp_upper(k, m, e):
    """One-sided Clopper-Pearson upper bound for a rate, k unfaithful out of m shown."""
    if m == 0:
        return 1.0
    return 1.0 if k >= m else float(beta.ppf(1 - e, k + 1, m - k))


def controlled_gate(sel_cal, loss_cal, sel_te, delta=0.05, eps=0.1, grid=20):
    """Lower selector value = more trustworthy. Candidate acceptance levels r = 1/grid ... 1.
    For each level, bound the unfaithful rate among accepted calibration cases with a
    Clopper-Pearson bound at confidence 1 - eps/grid (union bound over the grid).
    Choose the largest level whose bound is at most delta.
    With probability at least 1 - eps, at most a share delta of the accepted cases is unfaithful."""
    best = None
    for g in range(1, grid + 1):
        tau = np.quantile(sel_cal, g / grid)
        acc = sel_cal <= tau
        ub = cp_upper(int(loss_cal[acc].sum()), int(acc.sum()), eps / grid)
        if ub <= delta:
            best = tau
    if best is None:
        return np.zeros(len(sel_te), dtype=bool)
    return sel_te <= best


M2_COLS = ["dataset", "seed", "selector", "delta", "share", "unfaith_among_shown", "ok", "base_unfaith"]


def ecdf_rank(cal, x):
    """Share of calibration values that are at most x (0 = most trustworthy)."""
    return np.searchsorted(np.sort(cal), x, side="right") / len(cal)


def pilot_module2(name, Xd, yd, seeds=range(20), deltas=(0.05, 0.10, 0.20, 0.30), eps=0.1):
    K = len(np.unique(yd))
    rows = []
    for seed in seeds:
        Xtr_f, Xca_f, Xte_f, ytr, yca, yte = H.make_split(Xd, yd, seed)
        Bk = H.prototypes(Xtr_f, ytr, K)
        Dca, Dte = H.rho(Xca_f, Bk), H.rho(Xte_f, Bk)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr_f, ytr)
        Pca, Pte = rf.predict_proba(Xca_f), rf.predict_proba(Xte_f)
        loss_ca = (Dca.argmin(axis=1) != Pca.argmax(axis=1)).astype(int)   # needs no true labels
        loss_te = (Dte.argmin(axis=1) != Pte.argmax(axis=1)).astype(int)
        sel = {"fuzziness": (1 - (H.membership(Dca) ** 2).sum(axis=1), 1 - (H.membership(Dte) ** 2).sum(axis=1)),
               "confidence": (1 - Pca.max(axis=1), 1 - Pte.max(axis=1))}
        f_ca, f_te = sel["fuzziness"]
        c_ca, c_te = sel["confidence"]
        sel["combined"] = ((ecdf_rank(f_ca, f_ca) + ecdf_rank(c_ca, c_ca)) / 2,
                           (ecdf_rank(f_ca, f_te) + ecdf_rank(c_ca, c_te)) / 2)
        for sname, (s_cal, s_te) in sel.items():
            for delta in deltas:
                acc = controlled_gate(s_cal, loss_ca, s_te, delta, eps)
                share = acc.mean()
                u = loss_te[acc].mean() if acc.any() else np.nan
                rows.append([name, seed, sname, delta, share, u,
                             float(u <= delta) if acc.any() else np.nan, loss_te.mean()])
    return pd.DataFrame(rows, columns=M2_COLS)


# ---------------------------------------------------------------- Baselines and final runner
DING_DIR = os.environ.get("CCL_DING_DIR", "/content/class-conditional-conformal")   # authors' code (Ding et al., NeurIPS 2023)


def sets_default_pooled(score_cal, y_cal, score_te, alpha=0.1):
    """Class-by-class threshold when the class has at least n_min cases, otherwise the pooled threshold."""
    K = score_cal.shape[1]
    n_min = int(np.ceil((1 - alpha) / alpha))
    counts = np.bincount(y_cal, minlength=K)
    n = len(y_cal)
    sp = np.sort(score_cal[np.arange(n), y_cal])
    jp = int(np.ceil((n + 1) * (1 - alpha)))
    qp = np.inf if jp > n else sp[jp - 1]
    cols = []
    for k in range(K):
        if counts[k] >= n_min:
            s = np.sort(score_cal[y_cal == k, k])
            r = int(np.ceil((len(s) + 1) * (1 - alpha)))
            q = np.inf if r > len(s) else s[r - 1]
        else:
            q = qp
        cols.append(score_te[:, k] <= q)
    return np.stack(cols, axis=1)


def sets_clustered(score_cal, y_cal, score_te, alpha=0.1, seed=0, **kw):
    """Clustered conformal prediction, run with the authors' own code (Ding et al. 2023)."""
    import sys, io, contextlib
    if DING_DIR not in sys.path:
        sys.path.insert(0, DING_DIR)
    from utils.conformal_utils import clustered_conformal
    with contextlib.redirect_stdout(io.StringIO()):
        q = clustered_conformal(score_cal, y_cal, alpha, seed=seed, **kw)
    return score_te <= np.asarray(q, dtype=float)[None, :]


def make_variants(alpha):
    return {
        "pooled": lambda s, y, st, Bk, seed: H.sets_pooled(s, y, st, alpha),
        "classcond": lambda s, y, st, Bk, seed: H.sets_class(s, y, st, alpha),
        "default_pooled": lambda s, y, st, Bk, seed: sets_default_pooled(s, y, st, alpha),
        "clustered_auto": lambda s, y, st, Bk, seed: sets_clustered(s, y, st, alpha, seed),
        "clustered_manual": lambda s, y, st, Bk, seed: sets_clustered(
            s, y, st, alpha, seed, frac_clustering=0.5, num_clusters=2),
        "adaptive": lambda s, y, st, Bk, seed: sets_adaptive(s, y, st, alpha),
        "grouped": lambda s, y, st, Bk, seed: sets_grouped(s, y, st, Bk, alpha),
    }


SPLIT_COLS = ["dataset", "seed", "target", "n_rare_cal", "score", "variant", "coverage", "answered",
              "avg_size", "empty", "acc_on_answered"]
CLASS_COLS = ["dataset", "seed", "target", "score", "variant", "cls", "is_rare", "n_cal", "n_test", "n_hit"]
GATE_COLS2 = ["dataset", "seed", "selector", "delta", "share", "unfaith_among_shown", "ok", "base_unfaith"]


def run_all(name, Xd, yd, seeds, alpha=0.1, rare_targets=(None, 8, 4), deltas=(0.05, 0.10, 0.20), eps=0.1,
            skip=()):
    """One pass per split: fit once, then rare-class calibration experiments and the controlled gate."""
    K = len(np.unique(yd))
    rare = int(np.bincount(yd).argmin())
    variants = {k: v for k, v in make_variants(alpha).items() if k not in skip}
    srows, crows, grows, fails = [], [], [], {}
    for seed in seeds:
        Xtr_f, Xca_f, Xte_f, ytr, yca, yte = H.make_split(Xd, yd, seed)
        Bk = H.prototypes(Xtr_f, ytr, K)
        sc = class_scale(Xtr_f, ytr, Bk)
        Dca, Dte = H.rho(Xca_f, Bk), H.rho(Xte_f, Bk)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr_f, ytr)
        Pca, Pte = rf.predict_proba(Xca_f), rf.predict_proba(Xte_f)
        scores = {"distance": (Dca / sc, Dte / sc),
                  "membership": (1 - H.membership(Dca), 1 - H.membership(Dte)),
                  "forest": (1 - Pca, 1 - Pte)}
        for tgt in rare_targets:
            keep = shrink_rare(yca, rare, tgt, seed)
            yk = yca[keep]
            n_rare_cal = int((yk == rare).sum())
            tname = "natural" if tgt is None else str(tgt)
            for sname, (s_cal, s_te) in scores.items():
                for vname, fn in variants.items():
                    try:
                        S = fn(s_cal[keep], yk, s_te, Bk, seed)
                    except Exception as e:           # a baseline that cannot run is counted, not hidden
                        fails[vname] = fails.get(vname, 0) + 1
                        continue
                    size = S.sum(axis=1)
                    one = size == 1
                    hit = S[np.arange(len(yte)), yte]
                    acc = (S[one].argmax(axis=1) == yte[one]).mean() if one.any() else np.nan
                    srows.append([name, seed, tname, n_rare_cal, sname, vname, hit.mean(), one.mean(),
                                  size.mean(), (size == 0).mean(), acc])
                    for k in range(K):
                        m = yte == k
                        crows.append([name, seed, tname, sname, vname, k, int(k == rare),
                                      int((yk == k).sum()), int(m.sum()), int(hit[m].sum())])
        # controlled gate (natural calibration size)
        loss_ca = (Dca.argmin(axis=1) != Pca.argmax(axis=1)).astype(int)
        loss_te = (Dte.argmin(axis=1) != Pte.argmax(axis=1)).astype(int)
        f_ca = 1 - (H.membership(Dca) ** 2).sum(axis=1)
        f_te = 1 - (H.membership(Dte) ** 2).sum(axis=1)
        c_ca, c_te = 1 - Pca.max(axis=1), 1 - Pte.max(axis=1)
        sel = {"fuzziness": (f_ca, f_te), "confidence": (c_ca, c_te),
               "combined": ((ecdf_rank(f_ca, f_ca) + ecdf_rank(c_ca, c_ca)) / 2,
                            (ecdf_rank(f_ca, f_te) + ecdf_rank(c_ca, c_te)) / 2)}
        for sname, (s_cal, s_te) in sel.items():
            for delta in deltas:
                acc_m = controlled_gate(s_cal, loss_ca, s_te, delta, eps)
                u = loss_te[acc_m].mean() if acc_m.any() else np.nan
                grows.append([name, seed, sname, delta, acc_m.mean(), u,
                              float(u <= delta) if acc_m.any() else np.nan, loss_te.mean()])
        one = H.sets_pooled(1 - Pca, yca, 1 - Pte, alpha).sum(axis=1) == 1   # the earlier conformal gate
        u = loss_te[one].mean() if one.any() else np.nan
        grows.append([name, seed, "naive_singleton", np.nan, one.mean(), u, np.nan, loss_te.mean()])
    return (pd.DataFrame(srows, columns=SPLIT_COLS), pd.DataFrame(crows, columns=CLASS_COLS),
            pd.DataFrame(grows, columns=GATE_COLS2), fails)


# ---------------------------------------------------------------- Summaries
def summarise_module1(splits, classes):
    """One row per dataset, calibration size, score and method.
    Per-class coverage is pooled over all splits (hits / test cases), so rare classes are not noisy."""
    keys = ["dataset", "target", "score", "variant"]
    a = splits.groupby(keys)[["coverage", "answered", "avg_size"]].mean()
    c = classes[classes.n_test > 0].groupby(keys + ["cls", "is_rare"])[["n_test", "n_hit"]].sum().reset_index()
    c["cov"] = c.n_hit / c.n_test
    worst = c.groupby(keys)["cov"].min().rename("worst_class_cov")
    r = c[c.is_rare == 1].groupby(keys).agg(rare_cov=("cov", "mean"), rare_test_cases=("n_test", "sum"))
    return a.join(worst).join(r).round(3)


def summarise_gate(g):
    g = g.copy()
    g["shown"] = g.share > 0
    keys = ["dataset", "selector", "delta"]
    out = g.groupby(keys, dropna=False).agg(share=("share", "mean"), unfaith_among_shown=("unfaith_among_shown", "mean"),
                                           splits_with_explanations=("shown", "mean"),
                                           base_unfaith=("base_unfaith", "mean"))
    viol = g[g.shown & g.ok.notna()].groupby(keys).ok.apply(lambda x: float((x == 0).mean())).rename("share_of_splits_over_delta")
    return out.join(viol).round(3)
