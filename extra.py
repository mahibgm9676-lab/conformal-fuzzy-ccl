"""Two extra checks.
(A) rare_check: rare-class coverage over many splits (forest score, alpha = 0.1) for the pooled and the adaptive rule.
(B) lattice_check: a real concept lattice of the alpha-cut context, and how well its concepts agree with the forest."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import ccl_helpers as H
from ua_fccl import sets_adaptive


# ---------------------------------------------------------------- (A) more splits
def rare_check(name, Xd, yd, seeds, alpha=0.1):
    K = len(np.unique(yd)); rare = int(np.bincount(yd).argmin()); rows = []
    for seed in seeds:
        Xtr, Xca, Xte, ytr, yca, yte = H.make_split(Xd, yd, seed)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        s_ca, s_te = 1 - rf.predict_proba(Xca), 1 - rf.predict_proba(Xte)
        m = yte == rare
        for v, S in (("pooled", H.sets_pooled(s_ca, yca, s_te, alpha)), ("adaptive", sets_adaptive(s_ca, yca, s_te, alpha))):
            hit = S[np.arange(len(yte)), yte]
            rows.append([name, seed, v, hit.mean(), int(hit[m].sum()), int(m.sum())])
    return pd.DataFrame(rows, columns=["dataset", "seed", "variant", "coverage", "rare_hit", "rare_n"])


def summarise_rare(df, B=2000, seed=0):
    rng = np.random.RandomState(seed); out = []
    for (n, v), g in df.groupby(["dataset", "variant"]):
        g = g.sort_values("seed"); h, t = g.rare_hit.values, g.rare_n.values; S = len(g)
        bs = [h[i].sum() / t[i].sum() for i in (rng.randint(0, S, S) for _ in range(B))]
        lo, hi = np.percentile(bs, [2.5, 97.5]); m = S // 2
        out.append([n, v, S, int(t.sum()), round(h.sum() / t.sum(), 3), round(lo, 3), round(hi, 3),
                    round(abs(h[:m].sum() / t[:m].sum() - h[m:].sum() / t[m:].sum()), 3)])
    return pd.DataFrame(out, columns=["dataset", "variant", "splits", "rare_test_cases", "rare_cov", "ci_lo", "ci_hi", "half_diff"])


# ---------------------------------------------------------------- (B) concept lattice
def to_masks(Xb):
    """Binary matrix (objects x attributes) -> integer bitmask per object."""
    return (Xb.astype(np.int64) * (1 << np.arange(Xb.shape[1]))).sum(axis=1)


def closure(masks, S):
    """Formal-concept closure of attribute set S (bitmask): returns (extent indices, intent bitmask)."""
    ext = np.where((masks & S) == S)[0]
    if len(ext) == 0:
        return ext, None
    return ext, int(np.bitwise_and.reduce(masks[ext]))


def lattice_size(masks, m):
    """Number of formal concepts = number of distinct closed attribute sets (all intents), for m <= 12 attributes.
    The top concept (empty extent excluded) is included. Computed by closing every attribute subset."""
    top = int(np.bitwise_and.reduce(masks)); intents = set()
    for S in range(1 << m):
        ext, it = closure(masks, S)
        if it is not None:
            intents.add(it)
    return len(intents)


def lattice_check(name, Xd, yd, seeds, thetas=(0.3, 0.5, 0.7), count_max_attr=10):
    K = len(np.unique(yd)); m = Xd.shape[1]; rows = []
    for seed in seeds:
        Xtr, Xca, Xte, ytr, yca, yte = H.make_split(Xd, yd, seed)
        Bk = H.prototypes(Xtr, ytr, K)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        f_te = rf.predict(Xte)
        near = H.rho(Xte, Bk).argmin(axis=1)
        for th in thetas:
            mt = to_masks(Xtr >= th); me = to_masks(Xte >= th)
            nconc = lattice_size(mt, m) if (m <= count_max_attr and seed == seeds[0]) else np.nan
            sup = agree = pure = size = agree_near = 0
            for i in range(len(Xte)):
                ext, it = closure(mt, int(me[i]))
                if len(ext) == 0:
                    continue
                sup += 1; size += len(ext)
                cnt = np.bincount(ytr[ext], minlength=K)
                maj = cnt.argmax(); pure += cnt.max() / len(ext)
                agree += int(maj == f_te[i]); agree_near += int(near[i] == f_te[i])
            n = len(Xte)
            rows.append([name, seed, th, nconc, sup / n, size / max(sup, 1), pure / max(sup, 1),
                         agree / max(sup, 1), agree_near / max(sup, 1)])
    return pd.DataFrame(rows, columns=["dataset", "seed", "theta", "n_concepts", "support", "mean_extent",
                                       "purity", "agree_forest_concept", "agree_forest_pseudo"])
