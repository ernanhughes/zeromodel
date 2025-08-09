# tests/test_titanic_autotask.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from zeromodel import ZeroModel  # your package

RANDOM = 42

import seaborn as sns
import pandas as pd

def load_titanic():
    """
    Load Titanic dataset from Seaborn and return (X_raw, y).
    """
    df = sns.load_dataset("titanic")
    # Target: survived (0 or 1)
    y = df["survived"].astype(int)
    # Features: everything except 'survived'
    X_raw = df.drop(columns=["survived"])
    return X_raw, y

def featurize(X):
    num = ["age","sibsp","parch","fare"]
    cat = ["pclass","sex","embarked"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median"))
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(drop=None, sparse_output=False))
            ]), cat),
        ],
        remainder="drop"
    )

    X_proc = pre.fit_transform(X)
    # build feature names
    cat_names = pre.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat)
    feat_names = num + list(cat_names)
    return X_proc.astype(float), feat_names, pre

def rank_metrics(X, y, feature_names):
    # 1) Mutual Information
    mi = mutual_info_classif(X, y, random_state=RANDOM)
    mi = (mi - mi.min()) / (np.ptp(mi) + 1e-9)

    # 2) Logistic absolute coefficients (univariate-ish, fast)
    lr = LogisticRegression(max_iter=200, n_jobs=None, random_state=RANDOM)
    lr.fit(X, y)
    lr_w = np.abs(lr.coef_[0])
    lr_w = lr_w / (lr_w.max() + 1e-9)

    # 3) RF feature importance
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM, n_jobs=-1)
    rf.fit(X, y)
    rf_w = rf.feature_importances_ / (rf.feature_importances_.max() + 1e-9)

    # Combine (you can tweak)
    w = 0.5*mi + 0.3*rf_w + 0.2*lr_w
    order = np.argsort(-w)  # descending
    return w, order, [feature_names[i] for i in order]

def build_sql(metric_names, weights):
    # ORDER BY weighted sum desc
    # NOTE: quote identifiers for safety; DuckDB supports double quotes
    terms = [f"{weights[i]:.6f}*\"{name}\"" for i, name in enumerate(metric_names)]
    expr = " + ".join(terms)
    return f'SELECT * FROM virtual_index ORDER BY ({expr}) DESC'

def build_json_task(metric_names, weights):
    # db-less mode
    return {
        "order": [{"name": n, "weight": float(w)} for n, w in zip(metric_names, weights)],
        "mode": "weighted_sum_desc"
    }

def test_titanic_autotask():
    X_raw, y = load_titanic()
    X, feat_names, pre = featurize(X_raw)

    # train/test split for evaluation of enrichment
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM, stratify=y)

    # Normalize 0..1 per feature using train stats
    mn, mx = Xtr.min(0), Xtr.max(0)
    rng = np.where((mx - mn) == 0, 1.0, (mx - mn))
    tr = (Xtr - mn)/rng
    te = (Xte - mn)/rng

    # Rank metrics
    weights, order_idx, ordered_names = rank_metrics(tr, ytr.values, feat_names)

    # Sanity: top few should contain sex_male / pclass_3 / age (or proxies)
    joined = " ".join(ordered_names[:6])
    assert any("sex_male" in fn for fn in ordered_names[:6])
    assert any("pclass_3" in fn for fn in ordered_names[:6])
    assert ("age" in joined)  # age itself is numeric

    # Choose engine: SQL (DuckDB) or JSON (db-less)
    use_sql = True

    zm = ZeroModel(metric_names=feat_names)
    if use_sql:
        sql = build_sql(feat_names, weights)
        zm.prepare(tr, sql)                  # your single-entry method
    else:
        task = build_json_task(feat_names, weights)
        zm.prepare(tr, task)                 # same API, detect dict → db-less

    # Inspect top-k on TRAIN (where ranking was learned)
    k = 50
    top_docs = zm.doc_order[:k]
    # Map some interpretable columns back from processed features
    # sex_male is a one-hot column:
    sex_cols = [i for i,n in enumerate(feat_names) if n.endswith("sex_male")]
    pclass3_cols = [i for i,n in enumerate(feat_names) if n.endswith("pclass_3")]
    age_idx = feat_names.index("age")

    top_mean_male = tr[top_docs][:, sex_cols].mean()
    top_mean_pclass3 = tr[top_docs][:, pclass3_cols].mean()
    top_mean_age = tr[top_docs][:, age_idx].mean()

    global_mean_male = tr[:, sex_cols].mean()
    global_mean_pclass3 = tr[:, pclass3_cols].mean()
    global_mean_age = tr[:, age_idx].mean()

    # Enrichment: top-k should skew toward “likely to die”: male, 3rd class, older
    assert top_mean_male >= global_mean_male
    assert top_mean_pclass3 <= global_mean_pclass3
    assert top_mean_age >= global_mean_age

    # Optional: inference on TEST → take the top-left decision as "more likely to die"
    # Compare the label of the very top document index (on test batch prepared separately)
    zm_te = ZeroModel(metric_names=feat_names)
    if use_sql:
        zm_te.prepare(te, build_sql(feat_names, weights))
    else:
        zm_te.prepare(te, build_json_task(feat_names, weights))
    idx, rel = zm_te.get_decision()
    # Just sanity: the top test doc should have high male/pclass3/age
    test_top = te[idx]
    assert test_top[sex_cols].mean() >= global_mean_male * 0.8  # soft checks
