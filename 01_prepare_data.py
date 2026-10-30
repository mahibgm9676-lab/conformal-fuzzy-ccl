"""Step 1: download the nine public UCI datasets and save them as csv files in data/.

Run:  python 01_prepare_data.py
The class column is always the LAST column (ccl_helpers.load() relies on this).
Shill Bidding must be downloaded by hand (see below).
"""
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(os.environ.get("CCL_DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

# HTRU2, MAGIC, Page Blocks and Dry Bean come from the UCI repository by id.
ids = {"htru2": 372, "magic": 159, "pageblocks": 78, "drybean": 602}
for name, i in ids.items():
    d = fetch_ucirepo(id=i)
    out = pd.concat([d.data.features, d.data.targets], axis=1)
    out.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(name, out.shape, "| class column:", list(d.data.targets.columns))

# Wilt: the training and testing files are merged into one pool (4,839 cases).
req = urllib.request.Request("https://archive.ics.uci.edu/static/public/285/wilt.zip",
                             headers={"User-Agent": "Mozilla/5.0"})
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req).read()))
raw = DATA_DIR / "wilt_raw"
z.extractall(raw)
tr = pd.read_csv(raw / "training.csv")
te = pd.read_csv(raw / "testing.csv")
w = pd.concat([tr, te], ignore_index=True)
cls = [c for c in w.columns if c.strip().lower() == "class"][0]
w = w[[c for c in w.columns if c != cls] + [cls]]   # class goes last
w.to_csv(DATA_DIR / "wilt.csv", index=False)
print("wilt", w.shape)

# Shuttle, Yeast and Ecoli: numeric attributes only; classes with fewer than 8 cases are dropped
# (the paper counts the datasets after this step).
for name, i in {"shuttle": 148, "yeast": 110, "ecoli": 39}.items():
    d = fetch_ucirepo(id=i)
    Xf = d.data.features.select_dtypes("number")
    yt = d.data.targets.iloc[:, 0].astype(str)
    out = pd.concat([Xf, yt.rename("class")], axis=1)
    cnt = out["class"].value_counts()
    drop = cnt[cnt < 8].index.tolist()
    out = out[~out["class"].isin(drop)]
    out.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(name, out.shape, "dropped:", drop, "| class sizes:", out["class"].value_counts().tolist())

# Shill Bidding: download "Shill Bidding Dataset.csv" from
# https://doi.org/10.24432/C5Z611 and save it as data/Shill Bidding Dataset.csv.
# Columns ending in "_id" are dropped automatically by ccl_helpers.load().
print("Now put 'Shill Bidding Dataset.csv' into", DATA_DIR.resolve())
