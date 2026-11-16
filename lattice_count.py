"""Count the formal concepts of the alpha-cut context of the TRAINING objects for any dataset with up to about 20
attributes, and write the context as a ConExp-readable Burmeister file (.cxt).

Attribute k of an object is present when its min-max scaled value is at least theta (same rule as extra.py and
cxt_export.py). The number of formal concepts is the number of distinct closed attribute sets. The count that ConExp
shows includes the bottom concept even when its extent is empty (with_bottom). The older count in extra.py leaves the
bottom concept out when no object has all attributes (without_empty_bottom). The two are equal when some training
object has all attributes (for example Wilt at theta = 0.3).
Method: every attribute subset A (2**m of them) is closed when the intersection of all objects that contain A equals A.
The intersections are computed over the distinct attribute sets of the objects only, so Dry Bean (16 attributes) takes
seconds."""
import numpy as np, pandas as pd
import ccl_helpers as H


def to_masks(Xb):
    return (Xb.astype(np.int64) * (1 << np.arange(Xb.shape[1]))).sum(axis=1)


def concept_counts(Xb):
    m = Xb.shape[1]; masks = np.unique(to_masks(Xb)); full = (1 << m) - 1
    A = np.arange(1 << m, dtype=np.int64); cl = np.full(1 << m, full, dtype=np.int64)
    for u in masks:
        sel = (A & u) == A
        cl[sel] &= u
    with_bottom = int((cl == A).sum())
    has_full = bool((masks == full).any())
    return with_bottom, with_bottom - (0 if has_full else 1)


def export_cxt_all(csv_path, theta, seed, out_path):
    """Write the clarified context (identical attribute sets merged; rows named P{i}_n{size}_c{majority class}).
    Returns (object groups, attributes, concepts with bottom concept, concepts without an empty bottom concept)."""
    d = pd.read_csv(csv_path)
    feats = [c for c in d.columns[:-1] if not c.lower().endswith("_id")]
    X, y = H.load(csv_path)
    Xtr, Xca, Xte, ytr, yca, yte = H.make_split(X, y, seed)
    B = (Xtr >= theta).astype(int)
    pats, inv = np.unique(B, axis=0, return_inverse=True)
    inv = inv.ravel(); K = int(y.max()) + 1
    names = []
    for i in range(len(pats)):
        cnt = np.bincount(ytr[inv == i], minlength=K)
        names.append(f"P{i}_n{cnt.sum()}_c{cnt.argmax()}")
    attrs = [str(f).replace(" ", "_") for f in feats]
    with open(out_path, "w") as f:
        f.write("B\n\n%d\n%d\n\n" % (len(pats), len(attrs)))
        f.write("\n".join(names) + "\n")
        f.write("\n".join(attrs) + "\n")
        for row in pats:
            f.write("".join("X" if v else "." for v in row) + "\n")
    w, wo = concept_counts(B)
    return len(pats), len(attrs), w, wo
