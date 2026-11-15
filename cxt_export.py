"""Write the binary alpha-cut context of the TRAINING objects as a ConExp-readable Burmeister file (.cxt).
Objects with the same attribute set are merged (clarified context); each merged object is named with its size and
majority class, e.g. P3_n41_c2. Clarifying does not change the concept lattice. Attribute names come from the csv header."""
import numpy as np, pandas as pd
import ccl_helpers as H
from extra import to_masks, lattice_size


def export_cxt(csv_path, theta, seed, out_path, class_names=None):
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
    m = Xtr.shape[1]
    full = lattice_size(to_masks(B), m); clar = lattice_size(to_masks(pats), m)
    return len(pats), len(attrs), full, clar


def read_cxt(path):
    L = open(path).read().split("\n")
    no, na = int(L[2]), int(L[3])
    rows = L[5 + no + na: 5 + no + na + no]
    return np.array([[c == "X" for c in r] for r in rows])
