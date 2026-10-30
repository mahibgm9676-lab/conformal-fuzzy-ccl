import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import os
from pathlib import Path

# Folder that holds the nine csv files (see 01_prepare_data.py). Change it with the
# environment variable CCL_DATA_DIR, or leave it as "data" next to these scripts.
DATA_DIR = Path(os.environ.get("CCL_DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("CCL_RESULTS_DIR", "results"))

FILES = {"shill": str(DATA_DIR / "Shill Bidding Dataset.csv"),
         "wilt": str(DATA_DIR / "wilt.csv"),
         "htru2": str(DATA_DIR / "htru2.csv"),
         "magic": str(DATA_DIR / "magic.csv"),
         "pageblocks": str(DATA_DIR / "pageblocks.csv"),
         "drybean": str(DATA_DIR / "drybean.csv"),
         "shuttle": str(DATA_DIR / "shuttle.csv"),
         "yeast": str(DATA_DIR / "yeast.csv"),
         "ecoli": str(DATA_DIR / "ecoli.csv")}

METRIC_COLS = ["coverage", "min_class_cov", "answered", "acc_on_answered", "avg_size", "empty"]
CP_COLS = ["dataset", "seed", "alpha", "score", "variant"] + METRIC_COLS
GATE_COLS = ["dataset", "seed", "share", "fid_all", "fid_conformal", "fid_confidence",
             "fid_fuzziness", "fid_random", "auc_fuzziness", "auc_distance"]
B1_COLS = ["dataset", "seed", "model", "variant", "accuracy", "bal_acc"] + METRIC_COLS + ["fidelity"]
B2_COLS = ["dataset", "seed", "model", "share_defined", "top1_agree", "random_top1", "spearman"]

def load(path):
    d = pd.read_csv(path)
    cls = d.columns[-1]
    feats = [c for c in d.columns[:-1] if not c.lower().endswith("_id")]
    return d[feats].astype(float).values, LabelEncoder().fit_transform(d[cls].astype(str))

def rho(Xf, B, lam=0.5):
    eucl = np.linalg.norm(Xf[:, None, :] - B[None, :, :], axis=2)
    num = Xf @ B.T
    den = (np.linalg.norm(Xf, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :] + 1e-12)
    return lam * eucl + (1 - lam) * (1 - num / den)

def membership(D, p=2):
    D = np.maximum(D, 1e-12)
    ratio = D[:, :, None] / D[:, None, :]
    return 1.0 / (ratio ** (2 / (p - 1))).sum(axis=2)

def concept_feats(Xf, Bk):
    D = rho(Xf, Bk)
    return np.hstack([D, membership(D)])

def sets_pooled(score_cal, y_cal, score_te, alpha=0.1):
    n = len(y_cal)
    s = np.sort(score_cal[np.arange(n), y_cal])
    j = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.inf if j > n else s[j - 1]
    return score_te <= q

def sets_class(score_cal, y_cal, score_te, alpha=0.1):
    cols = []
    for k in range(score_cal.shape[1]):
        s = np.sort(score_cal[y_cal == k, k])
        n = len(s)
        j = int(np.ceil((n + 1) * (1 - alpha)))
        q = np.inf if (n == 0 or j > n) else s[j - 1]
        cols.append(score_te[:, k] <= q)
    return np.stack(cols, axis=1)

def set_metrics(s, yte, K):
    size = s.sum(axis=1)
    one = size == 1
    covered = s[np.arange(len(yte)), yte]
    percls = [covered[yte == k].mean() for k in range(K) if (yte == k).any()]
    acc = (s[one].argmax(axis=1) == yte[one]).mean() if one.any() else np.nan
    return {"coverage": covered.mean(), "min_class_cov": min(percls), "answered": one.mean(),
            "acc_on_answered": acc, "avg_size": size.mean(), "empty": (size == 0).mean()}

def make_split(Xd, yd, seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(Xd, yd, test_size=0.4, stratify=yd, random_state=seed)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    sc = MinMaxScaler()
    Xtr_f = sc.fit_transform(Xtr)
    Xca_f = np.clip(sc.transform(Xca), 0, 1)
    Xte_f = np.clip(sc.transform(Xte), 0, 1)
    return Xtr_f, Xca_f, Xte_f, ytr, yca, yte

def prototypes(Xtr_f, ytr, K):
    return np.array([Xtr_f[ytr == k].mean(axis=0) for k in range(K)])

def balance_idx(y_, seed, cap=10):
    rng = np.random.RandomState(seed)
    counts = np.bincount(y_)
    target = counts.max()
    parts = []
    for k, c in enumerate(counts):
        ik = np.where(y_ == k)[0]
        want = min(target, c * cap)
        extra = rng.choice(ik, want - c, replace=True) if want > c else np.array([], dtype=int)
        parts.append(np.concatenate([ik, extra]))
    return np.concatenate(parts)

COMBOS = [("distance", "classcond"), ("distance", "pooled"), ("membership", "classcond"),
          ("membership", "pooled"), ("logreg", "pooled"), ("forest", "pooled")]

def part_a(name, path, seeds=range(20), alphas=(0.05, 0.10, 0.20)):
    Xd, yd = load(path)
    K = len(np.unique(yd))
    cp_rows, gate_rows = [], []
    for seed in seeds:
        Xtr_f, Xca_f, Xte_f, ytr, yca, yte = make_split(Xd, yd, seed)
        Bk = prototypes(Xtr_f, ytr, K)
        Dca, Dte = rho(Xca_f, Bk), rho(Xte_f, Bk)
        lr = LogisticRegression(max_iter=1000).fit(Xtr_f, ytr)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr_f, ytr)
        P_ca, P_te = rf.predict_proba(Xca_f), rf.predict_proba(Xte_f)
        scores = {"distance": (Dca, Dte),
                  "membership": (1 - membership(Dca), 1 - membership(Dte)),
                  "logreg": (1 - lr.predict_proba(Xca_f), 1 - lr.predict_proba(Xte_f)),
                  "forest": (1 - P_ca, 1 - P_te)}
        for alpha in alphas:
            for sname, variant in COMBOS:
                s_cal, s_te = scores[sname]
                fn = sets_class if variant == "classcond" else sets_pooled
                m = set_metrics(fn(s_cal, yca, s_te, alpha), yte, K)
                cp_rows.append([name, seed, alpha, sname, variant] + [m[c] for c in METRIC_COLS])
        proto_pred = Dte.argmin(axis=1)
        agree = proto_pred == P_te.argmax(axis=1)
        one = sets_pooled(1 - P_ca, yca, 1 - P_te, 0.10).sum(axis=1) == 1
        n_sel = max(1, int(round(one.mean() * len(yte))))
        sel_conf = np.argsort(-P_te.max(axis=1), kind="stable")[:n_sel]
        fuzz = 1 - (membership(Dte) ** 2).sum(axis=1)
        sel_fuzz = np.argsort(fuzz, kind="stable")[:n_sel]
        sel_rand = np.random.RandomState(seed).choice(len(yte), n_sel, replace=False)
        err = proto_pred != yte
        ok = 0 < err.sum() < len(err)
        gate_rows.append([name, seed, one.mean(), agree.mean(),
                          agree[one].mean() if one.any() else np.nan,
                          agree[sel_conf].mean(), agree[sel_fuzz].mean(), agree[sel_rand].mean(),
                          roc_auc_score(err, fuzz) if ok else np.nan,
                          roc_auc_score(err, Dte.min(axis=1)) if ok else np.nan])
    return cp_rows, gate_rows

def part_b(name, path, seeds=range(10), alpha=0.1, n_attr=300):
    Xd, yd = load(path)
    K = len(np.unique(yd))
    d = Xd.shape[1]
    b1, b2 = [], []
    for seed in seeds:
        Xtr_f, Xca_f, Xte_f, ytr, yca, yte = make_split(Xd, yd, seed)
        Bk = prototypes(Xtr_f, ytr, K)
        idx = balance_idx(ytr, seed)
        proto_pred = rho(Xte_f, Bk).argmin(axis=1)
        mu = Xtr_f.mean(axis=0)
        raw = lambda Xf: Xf
        both = lambda Xf: np.hstack([Xf, concept_feats(Xf, Bk)])
        conc = lambda Xf: concept_feats(Xf, Bk)

        def fit_net(feat):
            m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed).fit(feat(Xtr_f)[idx], ytr[idx])
            return lambda Xf: m.predict_proba(feat(Xf))

        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr_f, ytr)
        proba = {"forest": rf.predict_proba, "mlp_raw": fit_net(raw),
                 "mlp_both": fit_net(both), "mlp_concept": fit_net(conc)}
        sub = np.random.RandomState(seed).choice(len(yte), min(n_attr, len(yte)), replace=False)
        Xs = Xte_f[sub]
        for mname, pf in proba.items():
            P_ca, P_te = pf(Xca_f), pf(Xte_f)
            pred = P_te.argmax(axis=1)
            for variant, fn in [("pooled", sets_pooled), ("classcond", sets_class)]:
                m = set_metrics(fn(1 - P_ca, yca, 1 - P_te, alpha), yte, K)
                b1.append([name, seed, mname, variant, (pred == yte).mean(),
                           balanced_accuracy_score(yte, pred)] + [m[c] for c in METRIC_COLS]
                          + [(pred == proto_pred).mean()])
            Ps = pf(Xs)
            order = np.argsort(-Ps, axis=1)
            k1, k2 = order[:, 0], order[:, 1]
            e = np.abs(Xs - Bk[k2]) - np.abs(Xs - Bk[k1])
            base = Ps[np.arange(len(Xs)), k1]
            o = np.zeros_like(Xs)
            for a in range(d):
                Xo = Xs.copy()
                Xo[:, a] = mu[a]
                o[:, a] = base - pf(Xo)[np.arange(len(Xs)), k1]
            ok = (np.ptp(o, axis=1) > 1e-12) & (np.ptp(e, axis=1) > 1e-12)
            if ok.any():
                re, ro = rankdata(e[ok], axis=1), rankdata(o[ok], axis=1)
                re = re - re.mean(axis=1, keepdims=True)
                ro = ro - ro.mean(axis=1, keepdims=True)
                sp = (re * ro).sum(axis=1) / np.sqrt((re ** 2).sum(axis=1) * (ro ** 2).sum(axis=1))
                top1 = (e[ok].argmax(axis=1) == o[ok].argmax(axis=1)).mean()
                b2.append([name, seed, mname, ok.mean(), top1, 1.0 / d, np.nanmean(sp)])
            else:
                b2.append([name, seed, mname, 0.0, np.nan, 1.0 / d, np.nan])
    return b1, b2