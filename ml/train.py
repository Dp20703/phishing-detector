"""
Train the phishing URL classifier.

Run from the project root:
    python ml/train.py

What it does
------------
1. Loads ml/data/urlset.csv, cleans it and removes duplicate URLs.
2. Splits 80/20 BY SITE (registered domain), so the test set only contains
   websites the model never saw. A plain random split lets the same site
   appear in train and test, which inflates scores.
3. Adds legitimate "homepage" examples (host + "/") derived from the legit
   training URLs. The raw dataset has no legit homepages at all, so without
   this the model flags every bare domain such as google.com as phishing.
   (Training split only - test data is never augmented.)
4. Compares Logistic Regression and Random Forest with grouped
   cross-validation on the TRAINING data, picks the better one by F1, refits
   it on all training data, and evaluates on the test set once.
5. Checks the final model on ml/eval/real_world_urls.csv (hand-made, small).
6. Saves backend/model/phishing_model.joblib and ml/metrics.json.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from features import ( 
    FEATURE_NAMES, _host_of, extract_features, normalize_url, registered_domain_of,
)

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = ROOT / "ml" / "data" / "urlset.csv"
REAL_WORLD_PATH = ROOT / "ml" / "eval" / "real_world_urls.csv"
MODEL_PATH = ROOT / "backend" / "model" / "phishing_model.joblib"
METRICS_PATH = ROOT / "ml" / "metrics.json"


# ---------------------------------------------------------------- data ------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, on_bad_lines="skip", dtype="str")
    df = df[["domain", "label"]].dropna()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["url"] = df["domain"].map(normalize_url)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    df["site"] = df["url"].map(registered_domain_of)
    df = df[df["site"] != ""].reset_index(drop=True)
    print(f"Rows after cleaning: {len(df):,}  "
          f"(legit={int((df.label == 0).sum()):,}, phishing={int((df.label == 1).sum()):,})")
    print(f"Unique sites: {df['site'].nunique():,}")
    return df


def build_features(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_features(u) for u in urls])[FEATURE_NAMES]


def homepage_augmentation(train_df: pd.DataFrame) -> pd.DataFrame:
    """Legit homepages: host + '/' for legit training sites (one per site)."""
    legit = train_df[train_df.label == 0]
    hosts = legit["url"].map(_host_of)
    homepages = pd.DataFrame({"url": (hosts + "/").map(normalize_url),
                              "site": legit["site"], "label": 0})
    homepages = homepages.drop_duplicates(subset="site")
    homepages = homepages[~homepages["url"].isin(set(train_df["url"]))]
    return homepages


# ---------------------------------------------------------------- models ----
def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        # Limited depth + min_samples_leaf keep the saved file small
        # (the unrestricted forest was ~200 MB).
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=SEED,
        ),
    }


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    df = load_data()

    # --- split by site -----------------------------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["site"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    print(f"Train: {len(train_df):,} URLs | Test: {len(test_df):,} URLs "
          f"(test sites unseen in training)")

    extra = homepage_augmentation(train_df)
    train_df = pd.concat([train_df[["url", "site", "label"]], extra], ignore_index=True)
    print(f"Added {len(extra):,} legit homepage examples to the TRAIN split")

    X_train, y_train = build_features(train_df["url"]), train_df["label"].values
    X_test, y_test = build_features(test_df["url"]), test_df["label"].values
    groups = train_df["site"].values

    # --- compare models with grouped CV on training data -------------------
    print("\nGrouped 3-fold CV on training data (F1 for the phishing class):")
    cv = GroupKFold(n_splits=3)
    cv_scores = {}
    for name in make_models():
        scores = []
        for tr, va in cv.split(X_train, y_train, groups):
            m = make_models()[name].fit(X_train.iloc[tr], y_train[tr])
            scores.append(f1_score(y_train[va], m.predict(X_train.iloc[va])))
        cv_scores[name] = round(float(np.mean(scores)), 4)
        print(f"  {name:<20} F1 = {cv_scores[name]}")
    best_name = max(cv_scores, key=cv_scores.get)
    print(f"Selected: {best_name}")

    # --- final evaluation on the untouched test set ------------------------
    results = {}
    fitted = {}
    for name in make_models():
        fitted[name] = make_models()[name].fit(X_train, y_train)
        results[name] = metrics(y_test, fitted[name].predict(X_test))
    print("\nTest-set results (unseen sites):")
    for name, r in results.items():
        print(f"  {name:<20} acc={r['accuracy']}  prec={r['precision']}  "
              f"rec={r['recall']}  f1={r['f1']}  cm={r['confusion_matrix']}")

    model = fitted[best_name]

    # --- feature importance (Random Forest) --------------------------------
    importance = None
    if best_name == "Random Forest":
        importance = dict(sorted(
            zip(FEATURE_NAMES, map(lambda v: round(float(v), 4), model.feature_importances_)),
            key=lambda kv: kv[1], reverse=True))
        print("\nFeature importance:")
        for k, v in importance.items():
            print(f"  {k:<22} {v}")

    # --- real-world sanity check -------------------------------------------
    rw = pd.read_csv(REAL_WORLD_PATH)
    X_rw = build_features(rw["url"])
    rw["prob"] = model.predict_proba(X_rw)[:, 1]
    rw["pred"] = (rw["prob"] >= 0.5).astype(int)
    legit_rw, phish_rw = rw[rw.label == 0], rw[rw.label == 1]
    real_world = {
        "n_legit": int(len(legit_rw)),
        "n_phishing_style": int(len(phish_rw)),
        "legit_flagged_as_phishing": int(legit_rw["pred"].sum()),
        "phishing_style_caught": int(phish_rw["pred"].sum()),
    }
    print("\nReal-world sanity set (hand-made, small - NOT a benchmark):")
    print(f"  Legit URLs wrongly flagged: {real_world['legit_flagged_as_phishing']}/{real_world['n_legit']}")
    print(f"  Phishing-style URLs caught: {real_world['phishing_style_caught']}/{real_world['n_phishing_style']}")
    wrong = rw[rw["pred"] != rw["label"]]
    for _, r in wrong.iterrows():
        print(f"    miss: p={r.prob:.2f} label={r.label} {r.url[:70]}")

    # --- save ----------------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "name": best_name},
                MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} ({size_mb:.1f} MB)")

    METRICS_PATH.write_text(json.dumps({
        "dataset_rows_after_cleaning": int(len(df)),
        "split": "80/20 grouped by registered domain, seed 42",
        "cv_f1": cv_scores,
        "selected_model": best_name,
        "test_results": results,
        "feature_importance": importance,
        "real_world_sanity_check": real_world,
        "model_size_mb": round(size_mb, 1),
    }, indent=2))
    print(f"Wrote {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
