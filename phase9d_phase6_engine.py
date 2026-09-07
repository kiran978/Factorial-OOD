#!/usr/bin/env python3
"""
FACTORIAL-OOD Phase 6 — Locked Baseline and Multimodal Model Benchmark
Protocol v0.2

Primary endpoints only (8).
Uses frozen Phase 2 labels, Phase 4 PRIMARY outer/inner folds, and Phase 5 leakage-safe features.
Does NOT use factorial-cell assignments or outer-test performance for tuning.

Learned model families (4-candidate equal tuning budget each):
  1) Elastic-net logistic regression
  2) Random forest
  3) Histogram gradient boosting
Modalities:
  STRUCTURE, HTTR, EARLY_FUSION, LATE_FUSION
Controls:
  outer-train prevalence, chemical 5-NN, biological 5-NN

Selection/calibration/thresholding:
  - hyperparameters: pooled parent-level inner OOF AUPRC
  - Platt calibration: pooled inner OOF raw parent probabilities only
  - MCC threshold: calibrated inner OOF, fixed 0.01..0.99 grid
  - Route B context probabilities: arithmetic mean to parent (primary)
  - late fusion: weight in {0,.25,.5,.75,1} selected by inner OOF AUPRC

Context-row fitting uses parent-normalized class-balanced sample weights so each
parent contributes equal total weight regardless of having 1/2/3 HTTr contexts.

Resume-safe: every endpoint × outer fold × family × modality is checkpointed.
"""

import os
# Set before numpy/sklearn imports. RF parallelism is separately controlled.
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("P6_OMP_THREADS", "8"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("P6_BLAS_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("P6_BLAS_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("P6_BLAS_THREADS", "1"))

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

PRIMARY_ENDPOINTS = [
    "TOX21_ERa_BLA_Antagonist_ratio",
    "TOX21_FXR_BLA_Antagonist_ratio",
    "TOX21_MMP_ratio",
    "TOX21_PPARg_BLA_Antagonist_ratio",
    "TOX21_RAR_LUC_Antagonist",
    "TOX21_RORg_LUC_CHO_Antagonist",
    "TOX21_SBE_BLA_Antagonist_ratio",
    "TOX21_hERG_U2OS_Antagonist",
]

MODEL_GRIDS = {
    "ELASTIC_NET": [
        {"candidate_id": "EN01", "C": 0.01, "l1_ratio": 0.5},
        {"candidate_id": "EN02", "C": 0.10, "l1_ratio": 0.5},
        {"candidate_id": "EN03", "C": 1.00, "l1_ratio": 0.5},
        {"candidate_id": "EN04", "C": 10.0, "l1_ratio": 0.5},
    ],
    "RANDOM_FOREST": [
        {"candidate_id": "RF01", "n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 1},
        {"candidate_id": "RF02", "n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5},
        {"candidate_id": "RF03", "n_estimators": 300, "max_features": 0.25, "min_samples_leaf": 1},
        {"candidate_id": "RF04", "n_estimators": 300, "max_features": 0.25, "min_samples_leaf": 5},
    ],
    "HIST_GRAD_BOOST": [
        {"candidate_id": "HGB01", "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.0},
        {"candidate_id": "HGB02", "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
        {"candidate_id": "HGB03", "learning_rate": 0.10, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"candidate_id": "HGB04", "learning_rate": 0.10, "max_leaf_nodes": 31, "l2_regularization": 0.0},
    ],
}

MODALITIES = ["STRUCTURE", "HTTR", "EARLY_FUSION"]
LEARNED_FAMILIES = list(MODEL_GRIDS)
LATE_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]  # weight on structure probability
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
EPS = 1e-6

DESCRIPTOR_COLS = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount",
    "FractionCSP3", "HeavyAtomCount", "FormalCharge"
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem",
        help="Project root containing Phase2, Phase4, Phase5, Phase5B_Output",
    )
    ap.add_argument("--rf-jobs", type=int, default=8)
    ap.add_argument("--only-endpoint", default=None, help="Optional exact endpoint for smoke/partial run")
    ap.add_argument("--only-outer", type=int, default=None, help="Optional outer fold 0..4")
    ap.add_argument("--only-family", choices=LEARNED_FAMILIES, default=None)
    ap.add_argument("--only-modality", choices=MODALITIES, default=None)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--skip-controls", action="store_true")
    return ap.parse_args()

ARGS = parse_args()
ROOT = Path(ARGS.root).resolve()
P2 = ROOT / "Phase2"
P4 = ROOT / "Phase4"
P5 = ROOT / "Phase5"
P5B = ROOT / "Phase5B_Output"
OUT = ROOT / "Phase6_Output"
CHECK = OUT / "checkpoints"
LOGS = OUT / "logs"
for d in [OUT, CHECK, LOGS]:
    d.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGS / "phase6_run.log"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def die(msg):
    log("FATAL: " + msg)
    raise RuntimeError(msg)


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_for(*parts):
    text = "|".join(map(str, ["FACTORIAL-OOD", "P6", "v1"] + list(parts)))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def safe_name(s):
    return s.replace("/", "__").replace(" ", "_")


def sanitize_prob(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def ap_score(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def balanced_parent_weights(y):
    y = np.asarray(y, dtype=int)
    n = len(y)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return np.ones(n, dtype=np.float64)
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return np.where(y == 1, w1, w0).astype(np.float64)


def context_parent_normalized_weights(parent_ids, y_parent_map):
    parent_ids = np.asarray(parent_ids, dtype=object)
    unique = pd.unique(parent_ids)
    ys = np.array([int(y_parent_map[p]) for p in unique], dtype=int)
    pw = balanced_parent_weights(ys)
    parent_w = dict(zip(unique, pw))
    counts = pd.Series(parent_ids).value_counts().to_dict()
    return np.array([parent_w[p] / counts[p] for p in parent_ids], dtype=np.float64)


def aggregate_context(parent_ids, probs, how="mean"):
    df = pd.DataFrame({"compound_id": np.asarray(parent_ids, dtype=object), "p": np.asarray(probs, dtype=float)})
    if how == "mean":
        out = df.groupby("compound_id", sort=False, as_index=False)["p"].mean()
    elif how == "median":
        out = df.groupby("compound_id", sort=False, as_index=False)["p"].median()
    else:
        raise ValueError(how)
    return out


class Platt1D:
    def __init__(self):
        self.status = "UNFIT"
        self.coef_ = 1.0
        self.intercept_ = 0.0
        self.model = None

    def fit(self, raw_p, y):
        raw_p = sanitize_prob(raw_p)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2 or float(np.std(raw_p)) < 1e-12:
            self.status = "IDENTITY_FALLBACK"
            return self
        x = logit(raw_p).reshape(-1, 1)
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=0)
        try:
            m.fit(x, y)
            self.model = m
            self.coef_ = float(m.coef_[0, 0])
            self.intercept_ = float(m.intercept_[0])
            self.status = "PLATT_OK"
        except Exception as e:
            self.status = f"IDENTITY_FALLBACK:{type(e).__name__}"
        return self

    def predict(self, raw_p):
        raw_p = sanitize_prob(raw_p)
        if self.model is None:
            return raw_p
        return self.model.predict_proba(logit(raw_p).reshape(-1, 1))[:, 1]


def select_mcc_threshold(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rows = []
    for t in THRESHOLD_GRID:
        pred = (p >= t).astype(int)
        m = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
        rows.append((float(m), float(t)))
    best_m = max(x[0] for x in rows)
    tied = [x[1] for x in rows if abs(x[0] - best_m) <= 1e-12]
    # deterministic tie-break: threshold closest to 0.5, then lower threshold
    t = sorted(tied, key=lambda x: (abs(x - 0.5), x))[0]
    return float(t), float(best_m)


def build_model(family, params, seed, rf_jobs):
    p = dict(params)
    p.pop("candidate_id", None)
    if family == "ELASTIC_NET":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight=None,
            max_iter=4000, tol=1e-3, random_state=seed, n_jobs=1, **p
        )
    if family == "RANDOM_FOREST":
        return RandomForestClassifier(
            max_depth=None, class_weight=None, bootstrap=True,
            n_jobs=rf_jobs, random_state=seed, **p
        )
    if family == "HIST_GRAD_BOOST":
        return HistGradientBoostingClassifier(
            max_iter=200, min_samples_leaf=20, early_stopping=False,
            class_weight=None, random_state=seed, **p
        )
    raise ValueError(family)


def fit_predict_prob(model, Xfit, yfit, wfit, Xeval):
    warning_names = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model.fit(Xfit, yfit, sample_weight=wfit)
        for w in rec:
            if issubclass(w.category, ConvergenceWarning):
                warning_names.append("ConvergenceWarning")
            else:
                warning_names.append(w.category.__name__)
    p = model.predict_proba(Xeval)[:, 1]
    return p, warning_names


def ece_equal_width(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            ece += (m.sum() / total) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def calibration_intercept_slope(y, p):
    y = np.asarray(y, dtype=int)
    p = sanitize_prob(p)
    if len(np.unique(y)) < 2 or np.std(p) < 1e-12:
        return np.nan, np.nan
    x = logit(p).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        m.fit(x, y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def metrics_from_oof(df):
    rows = []
    group_cols = ["endpoint", "model_family", "modality"]
    for keys, g in df.groupby(group_cols, sort=False):
        y = g["y_true"].astype(int).to_numpy()
        p = g["probability"].astype(float).to_numpy()
        pred = g["predicted_class"].astype(int).to_numpy()
        try:
            auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        except Exception:
            auroc = np.nan
        auprc = ap_score(y, p)
        mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0
        try:
            bal = float(balanced_accuracy_score(y, pred))
        except Exception:
            bal = np.nan
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        brier = float(brier_score_loss(y, p))
        ece = ece_equal_width(y, p, bins=10)
        cint, cslope = calibration_intercept_slope(y, p)
        rows.append({
            "endpoint": keys[0], "model_family": keys[1], "modality": keys[2],
            "n": len(g), "active_n": int(y.sum()), "inactive_n": int((y == 0).sum()),
            "prevalence": float(y.mean()), "AUPRC": auprc, "MCC": mcc,
            "AUROC": auroc, "balanced_accuracy": bal, "sensitivity": sens,
            "specificity": spec, "Brier": brier, "ECE_10bin": ece,
            "calibration_intercept": cint, "calibration_slope": cslope,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Validate inputs and load frozen objects
# ------------------------------------------------------------------
required = [
    P2 / "P2_04_endpoint_label_matrix.csv",
    P4 / "outputs/P4_01_primary_outer_split_manifest_FROZEN.csv",
    P4 / "outputs/P4_04_nested_inner_split_manifest_FROZEN.csv",
    P5 / "outputs/P5_01_Morgan_r2_2048_parent_matrix.npz",
    P5 / "outputs/P5_02_physicochemical_descriptors_raw.csv",
    P5 / "outputs/P5_04_descriptor_scaler_registry.csv",
    P5 / "outputs/P5_05_fusion_context_row_manifest.csv",
]
for x in required:
    if not x.exists():
        die(f"Missing required input: {x}")
for f in range(5):
    for suffix in ["FINAL_REFIT"] + [f"INNER{j}" for j in range(4)]:
        sid = f"P5B_OUTER{f}_{suffix}"
        p = P5B / "scores" / f"{sid}_scores.npz"
        if not p.exists():
            die(f"Missing Phase5B score file: {p}")

labels = pd.read_csv(P2 / "P2_04_endpoint_label_matrix.csv")
outer = pd.read_csv(P4 / "outputs/P4_01_primary_outer_split_manifest_FROZEN.csv")
inner = pd.read_csv(P4 / "outputs/P4_04_nested_inner_split_manifest_FROZEN.csv")
inner = inner[inner["split_set"].astype(str) == "PRIMARY"].copy()
inner["outer_fold"] = pd.to_numeric(inner["outer_fold"]).astype(int)
inner["inner_fold"] = pd.to_numeric(inner["inner_fold"]).astype(int)
outer["outer_fold"] = pd.to_numeric(outer["outer_fold"]).astype(int)

# Cross-version-safe Phase-5 structure loading.
# The Morgan NPZ contains an object-array compound_id created under NumPy 2.x;
# NumPy 1.x cannot unpickle that metadata (numpy._core). We therefore read the
# immutable parent order from the frozen CSV manifest and load ONLY the numeric
# Morgan bit matrix from the NPZ. This changes no scientific data or ordering.
structure_manifest = pd.read_csv(P5 / "outputs/P5_03_structure_feature_manifest.csv")
parent_ids = structure_manifest["compound_id"].astype(str).to_numpy()
zm = np.load(P5 / "outputs/P5_01_Morgan_r2_2048_parent_matrix.npz", allow_pickle=False)
MORGAN = np.asarray(zm["morgan_bits"], dtype=np.float32)
if MORGAN.shape != (2017, 2048):
    die(f"Unexpected Morgan shape: {MORGAN.shape}")
if len(parent_ids) != MORGAN.shape[0]:
    die("Structure manifest row count != Morgan matrix row count")
parent_row = {p: i for i, p in enumerate(parent_ids)}

desc_df = pd.read_csv(P5 / "outputs/P5_02_physicochemical_descriptors_raw.csv")
if list(desc_df["compound_id"].astype(str)) != list(parent_ids.astype(str)):
    die("Descriptor compound order != frozen structure-manifest order")
DESC = desc_df[DESCRIPTOR_COLS].to_numpy(dtype=np.float32)
scalers = pd.read_csv(P5 / "outputs/P5_04_descriptor_scaler_registry.csv")
fusion = pd.read_csv(P5 / "outputs/P5_05_fusion_context_row_manifest.csv")
fusion = fusion.sort_values("context_row_index").reset_index(drop=True)
if not np.array_equal(fusion["context_row_index"].to_numpy(), np.arange(len(fusion))):
    die("Fusion context row index is not 0..4020")
context_parent = fusion["compound_id"].astype(str).to_numpy(dtype=str)

label_maps = {}
for ep in PRIMARY_ENDPOINTS:
    if ep not in labels.columns:
        die(f"Missing primary endpoint: {ep}")
    s = labels.set_index("compound_id")[ep].dropna()
    label_maps[ep] = {str(k): int(v) for k, v in s.items()}

outer_fold_of = dict(zip(outer["compound_id"].astype(str), outer["outer_fold"].astype(int)))

# ------------------------------------------------------------------
# Prospective computational lock BEFORE model outcomes are examined
# ------------------------------------------------------------------
LOCK = {
    "phase": "6",
    "protocol": "v0.2",
    "created_before_model_results": True,
    "primary_endpoints": PRIMARY_ENDPOINTS,
    "outer_split": "Phase4 PRIMARY frozen 5-fold parent-group split",
    "inner_split": "Phase4 PRIMARY frozen 4-fold nested split inside each outer training fold",
    "modalities": ["STRUCTURE", "HTTR", "EARLY_FUSION", "LATE_FUSION"],
    "controls": {
        "PREVALENCE": "outer-training labeled-parent prevalence",
        "CHEMICAL_5NN": "5 nearest labeled training parents by Morgan/Tanimoto; similarity-weighted mean label",
        "BIOLOGICAL_5NN": "5 nearest labeled training parents by Euclidean distance between mean parent HTTr-PC vectors; inverse-distance-weighted mean label",
    },
    "model_grids": MODEL_GRIDS,
    "tuning": "maximize pooled parent-level inner OOF AUPRC; deterministic first candidate in declared order on exact tie",
    "sample_weighting": {
        "structure": "balanced class weight at parent level",
        "context_models": "balanced parent-level class weight divided by number of training contexts for that parent, so each parent has equal total contribution within class",
    },
    "calibration": "Platt logistic calibration on logit(raw probability), fitted on pooled inner OOF parent probabilities only; no class weighting",
    "threshold": "maximize MCC on calibrated inner OOF parent probabilities over fixed 0.01..0.99 grid; tie -> closest to 0.5 then lower",
    "route_B_aggregation": "arithmetic mean of context-row raw probabilities to parent before calibration (primary)",
    "late_fusion": {
        "formula": "w * calibrated_structure + (1-w) * calibrated_HTTr",
        "weight_grid": LATE_WEIGHTS,
        "selection": "inner OOF AUPRC only",
        "second_calibration": False,
    },
    "fixed_0_5_threshold": "stored as secondary sensitivity only; learned-model primary predicted class uses fold-specific inner-OOF MCC threshold",
    "outer_test_use": "prediction only; never model/hyperparameter/calibration/threshold/fusion-weight selection",
}
LOCK_PATH = OUT / "P6_00_locked_computational_registry.json"
if not LOCK_PATH.exists():
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(LOCK, f, indent=2)
LOCK_SHA = sha256_file(LOCK_PATH)
with open(OUT / "P6_00_locked_computational_registry.sha256.txt", "w") as f:
    f.write(f"{LOCK_SHA}  {LOCK_PATH.name}\n")
log(f"Phase 6 computational registry LOCKED: {LOCK_SHA}")

software = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    "sklearn": sklearn.__version__, "joblib": joblib.__version__,
    "rf_jobs": ARGS.rf_jobs,
}
with open(OUT / "P6_01_software_lock.json", "w") as f:
    json.dump(software, f, indent=2)


def scaler_for_scope(outer_fold, inner_fold=None):
    if inner_fold is None:
        sid = f"P5_OUTER{outer_fold}_FINAL_REFIT"
    else:
        sid = f"P5_OUTER{outer_fold}_INNER{inner_fold}"
    r = scalers[scalers["scope_id"] == sid]
    if len(r) != 1:
        die(f"Descriptor scaler row not unique: {sid}")
    r = r.iloc[0]
    means = np.array([r[f"{c}_mean"] for c in DESCRIPTOR_COLS], dtype=np.float32)
    scales = np.array([r[f"{c}_scale"] for c in DESCRIPTOR_COLS], dtype=np.float32)
    scales[scales == 0] = 1.0
    return means, scales


def structure_matrix(compounds, outer_fold, inner_fold=None):
    idx = np.array([parent_row[str(p)] for p in compounds], dtype=int)
    means, scales = scaler_for_scope(outer_fold, inner_fold)
    d = (DESC[idx] - means) / scales
    return np.concatenate([MORGAN[idx], d.astype(np.float32)], axis=1).astype(np.float32, copy=False)


def inner_parent_sets(f, j):
    outer_train = set(outer.loc[outer["outer_fold"] != f, "compound_id"].astype(str))
    sub = inner[(inner["outer_fold"] == f) & inner["compound_id"].astype(str).isin(outer_train)]
    val = set(sub.loc[sub["inner_fold"] == j, "compound_id"].astype(str))
    tr = set(sub.loc[sub["inner_fold"] != j, "compound_id"].astype(str))
    return tr, val


def outer_parent_sets(f):
    tr = set(outer.loc[outer["outer_fold"] != f, "compound_id"].astype(str))
    te = set(outer.loc[outer["outer_fold"] == f, "compound_id"].astype(str))
    return tr, te


def load_score_scope(f, j=None):
    sid = f"P5B_OUTER{f}_FINAL_REFIT" if j is None else f"P5B_OUTER{f}_INNER{j}"
    z = np.load(P5B / "scores" / f"{sid}_scores.npz")
    if j is None:
        return (
            z["fit_context_row_index"].astype(int), z["fit_scores"].astype(np.float32),
            z["outer_test_context_row_index"].astype(int), z["outer_test_scores"].astype(np.float32),
        )
    return (
        z["fit_context_row_index"].astype(int), z["fit_scores"].astype(np.float32),
        z["inner_val_context_row_index"].astype(int), z["inner_val_scores"].astype(np.float32),
    )


def data_for_modality(ep, f, modality, j=None):
    ymap = label_maps[ep]
    if j is None:
        tr_set, ev_set = outer_parent_sets(f)
    else:
        tr_set, ev_set = inner_parent_sets(f, j)

    if modality == "STRUCTURE":
        trp = sorted([p for p in tr_set if p in ymap])
        evp = sorted([p for p in ev_set if p in ymap])
        Xtr = structure_matrix(trp, f, j)
        Xev = structure_matrix(evp, f, j)
        ytr = np.array([ymap[p] for p in trp], dtype=int)
        yev = np.array([ymap[p] for p in evp], dtype=int)
        wtr = balanced_parent_weights(ytr)
        return {
            "Xtr": Xtr, "ytr": ytr, "wtr": wtr, "train_parent": np.asarray(trp, dtype=str),
            "Xev": Xev, "yev": yev, "eval_parent": np.asarray(evp, dtype=str),
            "eval_context_idx": None,
        }

    fit_idx, Zfit, eval_idx, Zeval = load_score_scope(f, j)
    fit_parent = context_parent[fit_idx]
    eval_parent = context_parent[eval_idx]
    fit_keep = np.array([(p in tr_set) and (p in ymap) for p in fit_parent], dtype=bool)
    eval_keep = np.array([(p in ev_set) and (p in ymap) for p in eval_parent], dtype=bool)
    fit_idx2 = fit_idx[fit_keep]
    eval_idx2 = eval_idx[eval_keep]
    trp = fit_parent[fit_keep]
    evp = eval_parent[eval_keep]
    Xtr = Zfit[fit_keep]
    Xev = Zeval[eval_keep]
    if modality == "EARLY_FUSION":
        Str = structure_matrix(trp, f, j)
        Sev = structure_matrix(evp, f, j)
        Xtr = np.concatenate([Str, Xtr], axis=1).astype(np.float32, copy=False)
        Xev = np.concatenate([Sev, Xev], axis=1).astype(np.float32, copy=False)
    ytr_rows = np.array([ymap[p] for p in trp], dtype=int)
    yev_rows = np.array([ymap[p] for p in evp], dtype=int)
    wtr = context_parent_normalized_weights(trp, ymap)
    return {
        "Xtr": Xtr, "ytr": ytr_rows, "wtr": wtr, "train_parent": trp,
        "Xev": Xev, "yev": yev_rows, "eval_parent": evp,
        "eval_context_idx": eval_idx2,
    }


def parent_eval_from_raw(data, raw_prob):
    if data["eval_context_idx"] is None:
        return pd.DataFrame({
            "compound_id": data["eval_parent"],
            "y_true": data["yev"].astype(int),
            "raw_probability": np.asarray(raw_prob, dtype=float),
        })
    agg = aggregate_context(data["eval_parent"], raw_prob, "mean").rename(columns={"p": "raw_probability"})
    ymap_local = {str(p): int(y) for p, y in zip(data["eval_parent"], data["yev"])}
    agg["y_true"] = agg["compound_id"].map(ymap_local).astype(int)
    return agg[["compound_id", "y_true", "raw_probability"]]


def context_prediction_frame(ep, f, family, modality, data, raw_prob):
    if data["eval_context_idx"] is None:
        return None
    return pd.DataFrame({
        "endpoint": ep, "outer_fold": f, "model_family": family, "modality": modality,
        "context_row_index": data["eval_context_idx"].astype(int),
        "compound_id": data["eval_parent"],
        "raw_context_probability": np.asarray(raw_prob, dtype=float),
    })


def checkpoint_paths(ep, f, family, modality):
    d = CHECK / safe_name(ep) / f"outer{f}"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{family}__{modality}"
    return d / f"{stem}.npz", d / f"{stem}.json"


def train_one(ep, f, family, modality):
    npz_path, js_path = checkpoint_paths(ep, f, family, modality)
    if npz_path.exists() and js_path.exists():
        meta = json.load(open(js_path, "r"))
        if meta.get("status") == "PASS" and meta.get("registry_sha256") == LOCK_SHA:
            log(f"SKIP {ep} outer{f} {family} {modality}")
            return

    t0 = time.time()
    log(f"FIT {ep} | outer={f} | {family} | {modality}")
    candidates = MODEL_GRIDS[family]
    cand_results = []
    cand_oof = {}

    for cidx, cand in enumerate(candidates):
        pred_parts = []
        warn_total = []
        failed = None
        for j in range(4):
            data = data_for_modality(ep, f, modality, j)
            if len(np.unique(data["ytr"])) < 2 or len(np.unique(data["yev"])) < 2:
                # Should not occur for Primary endpoints per Gate C, but log prospectively.
                failed = f"single-class inner fold j={j}"
                break
            model = build_model(
                family, cand,
                seed_for(ep, f, family, modality, cand["candidate_id"], j),
                ARGS.rf_jobs,
            )
            try:
                raw, warns = fit_predict_prob(model, data["Xtr"], data["ytr"], data["wtr"], data["Xev"])
                warn_total.extend(warns)
            except Exception as e:
                failed = f"{type(e).__name__}:{e}"
                break
            par = parent_eval_from_raw(data, raw)
            par["inner_fold"] = j
            pred_parts.append(par)
            del model, data, raw
            gc.collect()

        if failed is None:
            oof = pd.concat(pred_parts, ignore_index=True)
            # Every labeled outer-training parent should appear exactly once in inner OOF.
            if oof["compound_id"].duplicated().any():
                failed = "duplicate parent in pooled inner OOF"
            else:
                score = ap_score(oof["y_true"], oof["raw_probability"])
                cand_oof[cand["candidate_id"]] = oof
        else:
            score = np.nan

        cand_results.append({
            "candidate_id": cand["candidate_id"], "AUPRC": score,
            "failed": failed, "warning_count": len(warn_total),
            "warnings": sorted(set(warn_total)), "params": cand,
        })
        log(f"  candidate {cand['candidate_id']}: inner OOF AUPRC={score} failed={failed}")

    successful = [x for x in cand_results if x["failed"] is None and np.isfinite(x["AUPRC"])]
    if not successful:
        die(f"No successful candidate: {ep} outer{f} {family} {modality}")
    # Stable declared-grid tie breaker.
    declared_order = {c["candidate_id"]: i for i, c in enumerate(candidates)}
    best = sorted(successful, key=lambda x: (-x["AUPRC"], declared_order[x["candidate_id"]]))[0]
    best_cid = best["candidate_id"]
    best_params = next(c for c in candidates if c["candidate_id"] == best_cid)
    inner_oof = cand_oof[best_cid].copy()

    calibrator = Platt1D().fit(inner_oof["raw_probability"].to_numpy(), inner_oof["y_true"].to_numpy())
    inner_oof["calibrated_probability"] = calibrator.predict(inner_oof["raw_probability"].to_numpy())
    threshold, inner_mcc = select_mcc_threshold(inner_oof["y_true"], inner_oof["calibrated_probability"])

    final_data = data_for_modality(ep, f, modality, None)
    final_model = build_model(
        family, best_params,
        seed_for(ep, f, family, modality, best_cid, "FINAL"),
        ARGS.rf_jobs,
    )
    outer_raw_context_or_parent, final_warns = fit_predict_prob(
        final_model, final_data["Xtr"], final_data["ytr"], final_data["wtr"], final_data["Xev"]
    )
    outer_parent = parent_eval_from_raw(final_data, outer_raw_context_or_parent)
    outer_parent["probability"] = calibrator.predict(outer_parent["raw_probability"].to_numpy())
    outer_parent["predicted_class"] = (outer_parent["probability"] >= threshold).astype(int)
    outer_parent["predicted_class_0_5"] = (outer_parent["probability"] >= 0.5).astype(int)

    cframe = context_prediction_frame(ep, f, family, modality, final_data, outer_raw_context_or_parent)

    np_payload = {
        "inner_compound_id": inner_oof["compound_id"].astype(str).to_numpy(dtype=str),
        "inner_y_true": inner_oof["y_true"].to_numpy(dtype=np.int8),
        "inner_raw_probability": inner_oof["raw_probability"].to_numpy(dtype=np.float32),
        "inner_calibrated_probability": inner_oof["calibrated_probability"].to_numpy(dtype=np.float32),
        "outer_compound_id": outer_parent["compound_id"].astype(str).to_numpy(dtype=str),
        "outer_y_true": outer_parent["y_true"].to_numpy(dtype=np.int8),
        "outer_raw_probability": outer_parent["raw_probability"].to_numpy(dtype=np.float32),
        "outer_probability": outer_parent["probability"].to_numpy(dtype=np.float32),
        "outer_predicted_class": outer_parent["predicted_class"].to_numpy(dtype=np.int8),
        "outer_predicted_class_0_5": outer_parent["predicted_class_0_5"].to_numpy(dtype=np.int8),
    }
    if cframe is not None:
        np_payload.update({
            "outer_context_row_index": cframe["context_row_index"].to_numpy(dtype=np.int32),
            "outer_context_compound_id": cframe["compound_id"].astype(str).to_numpy(dtype=str),
            "outer_context_raw_probability": cframe["raw_context_probability"].to_numpy(dtype=np.float32),
        })
    np.savez_compressed(npz_path, **np_payload)

    meta = {
        "status": "PASS", "registry_sha256": LOCK_SHA,
        "endpoint": ep, "outer_fold": f, "model_family": family, "modality": modality,
        "selected_candidate": best_cid, "selected_params": best_params,
        "candidate_results": cand_results,
        "selection_metric": "inner_parent_OOF_AUPRC",
        "selected_inner_AUPRC": best["AUPRC"],
        "calibration_status": calibrator.status,
        "platt_coef": calibrator.coef_, "platt_intercept": calibrator.intercept_,
        "decision_threshold": threshold, "inner_MCC_at_threshold": inner_mcc,
        "final_fit_warnings": final_warns,
        "outer_labeled_parent_n": int(len(outer_parent)),
        "seconds": round(time.time() - t0, 3),
        "checkpoint_sha256": sha256_file(npz_path),
    }
    with open(js_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"DONE {ep} outer{f} {family} {modality}: {best_cid}, threshold={threshold:.2f}")
    del final_model, final_data, outer_raw_context_or_parent, outer_parent, inner_oof, cand_oof
    gc.collect()


def train_late_fusion(ep, f, family):
    modality = "LATE_FUSION"
    npz_path, js_path = checkpoint_paths(ep, f, family, modality)
    if npz_path.exists() and js_path.exists():
        meta = json.load(open(js_path, "r"))
        if meta.get("status") == "PASS" and meta.get("registry_sha256") == LOCK_SHA:
            log(f"SKIP {ep} outer{f} {family} LATE_FUSION")
            return
    sp, sj = checkpoint_paths(ep, f, family, "STRUCTURE")
    hp, hj = checkpoint_paths(ep, f, family, "HTTR")
    if not sp.exists() or not hp.exists():
        die(f"Late fusion prerequisites missing for {ep} outer{f} {family}")
    zs = np.load(sp, allow_pickle=True)
    zh = np.load(hp, allow_pickle=True)

    si = pd.DataFrame({
        "compound_id": zs["inner_compound_id"].astype(str),
        "y": zs["inner_y_true"].astype(int),
        "s": zs["inner_calibrated_probability"].astype(float),
    })
    hi = pd.DataFrame({
        "compound_id": zh["inner_compound_id"].astype(str),
        "y_h": zh["inner_y_true"].astype(int),
        "h": zh["inner_calibrated_probability"].astype(float),
    })
    inn = si.merge(hi, on="compound_id", how="inner", validate="one_to_one")
    if not np.array_equal(inn["y"].to_numpy(), inn["y_h"].to_numpy()):
        die("Late-fusion inner labels mismatch")
    wr = []
    for w in LATE_WEIGHTS:
        p = w * inn["s"].to_numpy() + (1 - w) * inn["h"].to_numpy()
        wr.append({"w_structure": w, "AUPRC": ap_score(inn["y"], p)})
    best = sorted(wr, key=lambda r: (-r["AUPRC"], abs(r["w_structure"] - 0.5), r["w_structure"]))[0]
    w = float(best["w_structure"])
    innp = w * inn["s"].to_numpy() + (1 - w) * inn["h"].to_numpy()
    threshold, inner_mcc = select_mcc_threshold(inn["y"].to_numpy(), innp)

    so = pd.DataFrame({
        "compound_id": zs["outer_compound_id"].astype(str),
        "y": zs["outer_y_true"].astype(int),
        "s": zs["outer_probability"].astype(float),
    })
    ho = pd.DataFrame({
        "compound_id": zh["outer_compound_id"].astype(str),
        "y_h": zh["outer_y_true"].astype(int),
        "h": zh["outer_probability"].astype(float),
    })
    out = so.merge(ho, on="compound_id", how="inner", validate="one_to_one")
    if not np.array_equal(out["y"].to_numpy(), out["y_h"].to_numpy()):
        die("Late-fusion outer labels mismatch")
    p = w * out["s"].to_numpy() + (1 - w) * out["h"].to_numpy()
    pred = (p >= threshold).astype(np.int8)
    pred05 = (p >= 0.5).astype(np.int8)
    np.savez_compressed(
        npz_path,
        inner_compound_id=inn["compound_id"].astype(str).to_numpy(dtype=str),
        inner_y_true=inn["y"].to_numpy(dtype=np.int8),
        inner_raw_probability=innp.astype(np.float32),
        inner_calibrated_probability=innp.astype(np.float32),
        outer_compound_id=out["compound_id"].astype(str).to_numpy(dtype=str),
        outer_y_true=out["y"].to_numpy(dtype=np.int8),
        outer_raw_probability=p.astype(np.float32),
        outer_probability=p.astype(np.float32),
        outer_predicted_class=pred,
        outer_predicted_class_0_5=pred05,
    )
    meta = {
        "status": "PASS", "registry_sha256": LOCK_SHA, "endpoint": ep, "outer_fold": f,
        "model_family": family, "modality": modality,
        "weight_grid_results": wr, "selected_structure_weight": w,
        "selected_inner_AUPRC": best["AUPRC"], "decision_threshold": threshold,
        "inner_MCC_at_threshold": inner_mcc, "second_calibration": False,
        "checkpoint_sha256": sha256_file(npz_path),
    }
    with open(js_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"DONE {ep} outer{f} {family} LATE_FUSION: w_structure={w:.2f}, threshold={threshold:.2f}")


# ------------------------------------------------------------------
# Fixed controls
# ------------------------------------------------------------------
def chemical_similarity_ranking(f):
    tr, te = outer_parent_sets(f)
    trp = np.asarray(sorted(tr), dtype=str)
    tep = np.asarray(sorted(te), dtype=str)
    Xi = MORGAN[np.array([parent_row[p] for p in trp])]
    Xq = MORGAN[np.array([parent_row[p] for p in tep])]
    inter = Xq @ Xi.T
    qsum = Xq.sum(axis=1)[:, None]
    isum = Xi.sum(axis=1)[None, :]
    denom = qsum + isum - inter
    sim = np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)
    order = np.argsort(-sim, axis=1)
    return trp, tep, sim, order


def biological_parent_embeddings(f):
    fit_idx, Zfit, test_idx, Ztest = load_score_scope(f, None)
    def agg(idx, z):
        d = pd.DataFrame({"compound_id": context_parent[idx]})
        # group row positions to average vectors
        groups = d.groupby("compound_id", sort=False).indices
        ids = []
        vecs = []
        for p, pos in groups.items():
            ids.append(str(p)); vecs.append(z[np.asarray(pos, dtype=int)].mean(axis=0))
        return np.asarray(ids, dtype=str), np.asarray(vecs, dtype=np.float32)
    return (*agg(fit_idx, Zfit), *agg(test_idx, Ztest))


def run_controls():
    rows = []
    for f in range(5):
        log(f"CONTROL preparation outer={f}")
        tr_set, te_set = outer_parent_sets(f)
        trp_chem, tep_chem, sim, order = chemical_similarity_ranking(f)
        trp_bio, Ztrp, tep_bio, Ztep = biological_parent_embeddings(f)
        # Euclidean distances parent-level, all training parents; small enough.
        # d^2 = a^2+b^2-2ab
        a2 = (Ztep ** 2).sum(axis=1)[:, None]
        b2 = (Ztrp ** 2).sum(axis=1)[None, :]
        d2 = np.maximum(a2 + b2 - 2.0 * (Ztep @ Ztrp.T), 0.0)
        bio_order = np.argsort(d2, axis=1)

        chem_test_pos = {p: i for i, p in enumerate(tep_chem)}
        bio_test_pos = {p: i for i, p in enumerate(tep_bio)}

        for ep in PRIMARY_ENDPOINTS:
            ymap = label_maps[ep]
            train_lab = [p for p in tr_set if p in ymap]
            test_lab = sorted([p for p in te_set if p in ymap])
            prev = float(np.mean([ymap[p] for p in train_lab]))
            for p in test_lab:
                y = int(ymap[p])
                # prevalence
                rows.append({
                    "endpoint": ep, "outer_fold": f, "compound_id": p, "y_true": y,
                    "model_family": "PREVALENCE", "modality": "CONTROL",
                    "probability": prev, "decision_threshold": 0.5,
                    "predicted_class": int(prev >= 0.5), "predicted_class_0_5": int(prev >= 0.5),
                })
                # chemical 5NN
                qi = chem_test_pos[p]
                chosen = []
                for ti in order[qi]:
                    nbr = str(trp_chem[ti])
                    if nbr in ymap:
                        chosen.append((nbr, float(sim[qi, ti])))
                        if len(chosen) == 5:
                            break
                if not chosen:
                    cp = prev
                else:
                    ws = np.array([max(s, EPS) for _, s in chosen], dtype=float)
                    ys = np.array([ymap[n] for n, _ in chosen], dtype=float)
                    cp = float(np.sum(ws * ys) / np.sum(ws))
                rows.append({
                    "endpoint": ep, "outer_fold": f, "compound_id": p, "y_true": y,
                    "model_family": "CHEMICAL_5NN", "modality": "STRUCTURE_CONTROL",
                    "probability": cp, "decision_threshold": 0.5,
                    "predicted_class": int(cp >= 0.5), "predicted_class_0_5": int(cp >= 0.5),
                })
                # biological 5NN
                qi = bio_test_pos[p]
                chosen = []
                for ti in bio_order[qi]:
                    nbr = str(trp_bio[ti])
                    if nbr in ymap:
                        dist = math.sqrt(float(d2[qi, ti]))
                        chosen.append((nbr, dist))
                        if len(chosen) == 5:
                            break
                if not chosen:
                    bp = prev
                else:
                    ws = np.array([1.0 / (d + 1e-6) for _, d in chosen], dtype=float)
                    ys = np.array([ymap[n] for n, _ in chosen], dtype=float)
                    bp = float(np.sum(ws * ys) / np.sum(ws))
                rows.append({
                    "endpoint": ep, "outer_fold": f, "compound_id": p, "y_true": y,
                    "model_family": "BIOLOGICAL_5NN", "modality": "HTTR_CONTROL",
                    "probability": bp, "decision_threshold": 0.5,
                    "predicted_class": int(bp >= 0.5), "predicted_class_0_5": int(bp >= 0.5),
                })
        del sim, order, Ztrp, Ztep, d2, bio_order
        gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "P6_02_control_OOF_predictions.csv", index=False)
    log(f"Controls complete: {len(out)} OOF rows")


# ------------------------------------------------------------------
# Execute selected scope
# ------------------------------------------------------------------
log("=" * 80)
log("FACTORIAL-OOD Phase 6 locked benchmark starting/resuming")
log(f"Root: {ROOT}")
log(f"Computational registry SHA256: {LOCK_SHA}")

endpoints_run = PRIMARY_ENDPOINTS
if ARGS.only_endpoint:
    if ARGS.only_endpoint not in PRIMARY_ENDPOINTS:
        die("--only-endpoint must be one of the 8 Primary endpoints")
    endpoints_run = [ARGS.only_endpoint]
outers_run = list(range(5)) if ARGS.only_outer is None else [ARGS.only_outer]
families_run = LEARNED_FAMILIES if ARGS.only_family is None else [ARGS.only_family]
modalities_run = MODALITIES if ARGS.only_modality is None else [ARGS.only_modality]

if not ARGS.skip_controls and not ARGS.controls_only:
    cp = OUT / "P6_02_control_OOF_predictions.csv"
    if not cp.exists():
        run_controls()
    else:
        log("SKIP controls: output already exists")
elif ARGS.controls_only:
    run_controls()

if not ARGS.controls_only:
    for ep in endpoints_run:
        for f in outers_run:
            for family in families_run:
                # If only one modality requested, late fusion is not attempted automatically.
                for modality in modalities_run:
                    train_one(ep, f, family, modality)
                if ARGS.only_modality is None:
                    train_late_fusion(ep, f, family)

# ------------------------------------------------------------------
# Consolidate only when complete full primary run exists
# ------------------------------------------------------------------
def all_primary_checkpoints_present():
    for ep in PRIMARY_ENDPOINTS:
        for f in range(5):
            for fam in LEARNED_FAMILIES:
                for mod in ["STRUCTURE", "HTTR", "EARLY_FUSION", "LATE_FUSION"]:
                    npz, js = checkpoint_paths(ep, f, fam, mod)
                    if not npz.exists() or not js.exists():
                        return False
    return True

if all_primary_checkpoints_present():
    log("All learned checkpoints present. Consolidating immutable OOF tables...")
    parent_rows = []
    context_rows = []
    tuning_rows = []
    for ep in PRIMARY_ENDPOINTS:
        for f in range(5):
            for fam in LEARNED_FAMILIES:
                for mod in ["STRUCTURE", "HTTR", "EARLY_FUSION", "LATE_FUSION"]:
                    npz, js = checkpoint_paths(ep, f, fam, mod)
                    z = np.load(npz, allow_pickle=True)
                    meta = json.load(open(js, "r"))
                    for cid, y, raw, p, pred, pred05 in zip(
                        z["outer_compound_id"].astype(str), z["outer_y_true"].astype(int),
                        z["outer_raw_probability"].astype(float), z["outer_probability"].astype(float),
                        z["outer_predicted_class"].astype(int), z["outer_predicted_class_0_5"].astype(int),
                    ):
                        parent_rows.append({
                            "endpoint": ep, "outer_fold": f, "compound_id": cid, "y_true": y,
                            "model_family": fam, "modality": mod,
                            "raw_probability": raw, "probability": p,
                            "decision_threshold": float(meta["decision_threshold"]),
                            "predicted_class": pred, "predicted_class_0_5": pred05,
                            "registry_sha256": LOCK_SHA,
                        })
                    if "outer_context_row_index" in z.files:
                        for rid, cid, rp in zip(
                            z["outer_context_row_index"].astype(int),
                            z["outer_context_compound_id"].astype(str),
                            z["outer_context_raw_probability"].astype(float),
                        ):
                            context_rows.append({
                                "endpoint": ep, "outer_fold": f, "model_family": fam, "modality": mod,
                                "context_row_index": rid, "compound_id": cid,
                                "raw_context_probability": rp, "registry_sha256": LOCK_SHA,
                            })
                    tuning_rows.append({
                        "endpoint": ep, "outer_fold": f, "model_family": fam, "modality": mod,
                        "selected_candidate": meta.get("selected_candidate"),
                        "selected_structure_weight": meta.get("selected_structure_weight"),
                        "selected_inner_AUPRC": meta.get("selected_inner_AUPRC"),
                        "decision_threshold": meta.get("decision_threshold"),
                        "calibration_status": meta.get("calibration_status", "NOT_APPLICABLE"),
                        "platt_coef": meta.get("platt_coef"), "platt_intercept": meta.get("platt_intercept"),
                        "checkpoint_sha256": meta.get("checkpoint_sha256"),
                    })

    learned = pd.DataFrame(parent_rows)
    controls_path = OUT / "P6_02_control_OOF_predictions.csv"
    controls = pd.read_csv(controls_path) if controls_path.exists() else pd.DataFrame()
    allpred = pd.concat([controls, learned], ignore_index=True, sort=False)
    allpred.to_csv(OUT / "P6_03_locked_parent_OOF_predictions.csv", index=False)
    pd.DataFrame(context_rows).to_csv(OUT / "P6_04_locked_context_OOF_raw_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "P6_05_tuning_calibration_threshold_registry.csv", index=False)
    metrics = metrics_from_oof(allpred)
    metrics.to_csv(OUT / "P6_06_global_OOF_performance.csv", index=False)

    # Integrity checks; no factorial-cell or model-result-driven decisions.
    tests = []
    def test(tid, desc, ok, detail):
        tests.append({"test_id": tid, "description": desc, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    test("P6-T01", "8 Primary endpoints represented", set(allpred.endpoint.unique()) == set(PRIMARY_ENDPOINTS), len(allpred.endpoint.unique()))
    test("P6-T02", "5 outer folds represented", set(allpred.outer_fold.unique()) == set(range(5)), sorted(allpred.outer_fold.unique()))
    test("P6-T03", "No duplicate learned parent OOF key", not learned.duplicated(["endpoint","model_family","modality","compound_id"]).any(), int(learned.duplicated(["endpoint","model_family","modality","compound_id"]).sum()))
    test("P6-T04", "All probabilities finite", np.isfinite(allpred.probability.astype(float)).all(), "finite")
    test("P6-T05", "All probabilities within [0,1]", ((allpred.probability.astype(float) >= 0) & (allpred.probability.astype(float) <= 1)).all(), "range")
    test("P6-T06", "All learned thresholds within [0.01,0.99]", ((learned.decision_threshold >= .01) & (learned.decision_threshold <= .99)).all(), "range")
    test("P6-T07", "All learned predictions use locked registry", (learned.registry_sha256 == LOCK_SHA).all(), LOCK_SHA)
    test("P6-T08", "Expected learned family-modality combinations = 12", learned[["model_family","modality"]].drop_duplicates().shape[0] == 12, learned[["model_family","modality"]].drop_duplicates().shape[0])
    test("P6-T09", "Expected controls = 3", controls[["model_family","modality"]].drop_duplicates().shape[0] == 3, controls[["model_family","modality"]].drop_duplicates().shape[0])
    test("P6-T10", "No endpoint label missing in prediction table", not allpred.y_true.isna().any(), int(allpred.y_true.isna().sum()))
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(OUT / "P6_07_prediction_integrity_tests.csv", index=False)
    if (tests_df.status != "PASS").any():
        die("Phase 6 prediction integrity test failed")

    # SHA manifest
    files = [p for p in OUT.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.name != "SHA256SUMS_PHASE6.txt"]
    with open(OUT / "SHA256SUMS_PHASE6.txt", "w") as fh:
        for p in sorted(files):
            fh.write(f"{sha256_file(p)}  {p.relative_to(OUT)}\n")
    status = {
        "status": "PASS",
        "phase": "6",
        "primary_endpoint_n": 8,
        "learned_family_n": 3,
        "learned_modalities_per_family": 4,
        "control_n": 3,
        "locked_parent_oof_rows": int(len(allpred)),
        "locked_context_oof_rows": int(len(context_rows)),
        "registry_sha256": LOCK_SHA,
        "prediction_integrity_tests": f"{len(tests_df)}/{len(tests_df)} PASS",
        "interpretation_boundary": "global benchmark metrics generated; factorial C0/B0-C1/B1 interpretation prohibited until Phase 7",
    }
    with open(OUT / "P6_FINAL_STATUS.json", "w") as fh:
        json.dump(status, fh, indent=2)

    # Upload-friendly package: consolidated outputs + checkpoint JSON metadata,
    # excluding the much larger per-scope NPZ checkpoint arrays.
    upload_zip = OUT / "Phase6_UPLOAD_TO_CHAT.zip"
    with zipfile.ZipFile(upload_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or p == upload_zip:
                continue
            rel = p.relative_to(OUT)
            if "checkpoints" in rel.parts and p.suffix.lower() == ".npz":
                continue
            zf.write(p, arcname=str(rel))
    upload_sha = sha256_file(upload_zip)
    status["upload_zip"] = str(upload_zip)
    status["upload_zip_sha256"] = upload_sha
    status["upload_zip_size_bytes"] = upload_zip.stat().st_size
    with open(OUT / "P6_FINAL_STATUS.json", "w") as fh:
        json.dump(status, fh, indent=2)

    log("=" * 80)
    log("PHASE 6 LOCAL COMPUTATION PASS")
    log(f"Locked parent OOF rows: {len(allpred)}")
    log(f"Prediction integrity: {len(tests_df)}/{len(tests_df)} PASS")
    log(f"Upload to ChatGPT: {upload_zip}")
    log(f"Upload ZIP SHA256: {upload_sha}")
    log(f"Output folder: {OUT}")
    log("Do not interpret factorial cells until Phase 7.")
    log("=" * 80)
else:
    log("Partial/resume run completed. Full consolidation waits until all primary checkpoints exist.")
