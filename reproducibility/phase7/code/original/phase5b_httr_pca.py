#!/usr/bin/env python3
"""
FACTORIAL-OOD — Phase 5B
Leakage-safe fold-specific HTTr PCA construction

Frozen scientific specification:
- Route B parent-context rows
- duplicate parent-context source rows collapsed featurewise by median
- 16,821 HTTr variables
- variance filtering fit inside each training scope
- StandardScaler fit inside each training scope
- randomized PCA fit inside each training scope
- retain min(256 PCs, number needed for >=80% training variance)
- five frozen outer folds
- four frozen inner folds inside every outer-training fold
- NO Tox21 endpoint labels are read or used

Designed for:
  /home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem

Outputs:
  Phase5B_Output/
      cache/
      transformers/
      scores/
      qa/
      logs/
      Phase5B_UPLOAD_TO_CHAT.zip
      Phase5B_FULL_LOCAL_ARCHIVE.zip  (optional; created at the end if disk permits)

The script is resume-safe. A completed scope is skipped on rerun.
"""

import os
# Limit BLAS oversubscription before importing NumPy/scikit-learn.
THREADS = os.environ.get("P5B_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", THREADS)
os.environ.setdefault("MKL_NUM_THREADS", THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", THREADS)

import sys
import gc
import json
import time
import math
import hashlib
import zipfile
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn
import scipy
import joblib

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path("/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem")
FEATHER = ROOT / "chemical_httr_assay_aggregated.feather"
P2 = ROOT / "Phase2"
P4 = ROOT / "Phase4"
OUT = ROOT / "Phase5B_Output"

CACHE = OUT / "cache"
TRANSFORMERS = OUT / "transformers"
SCORES = OUT / "scores"
QA = OUT / "qa"
LOGS = OUT / "logs"

for d in [OUT, CACHE, TRANSFORMERS, SCORES, QA, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOGFILE = LOGS / "phase5b_run.log"

EXPECTED_FEATHER_SHA256 = "480f3aad0e84acce7b5affb54fdd9027258e93275de09cf3f21b97971d17cc8a"
EXPECTED_RAW_ROWS = 4142
EXPECTED_CONTEXT_ROWS = 4021
EXPECTED_PARENTS = 2017
EXPECTED_HTTR = 16821
EXPECTED_OUTER_FOLDS = set(range(5))
EXPECTED_INNER_FOLDS = set(range(4))
MAX_PCS = 256
TARGET_VARIANCE = 0.80
PROTOCOL = "v0.2"
PCA_ENGINE = "sklearn.PCA(randomized, iterated_power=4, n_oversamples=10)"

# ---------------------------------------------------------------------
# Logging/utilities
# ---------------------------------------------------------------------
def log(msg):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
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

def deterministic_seed(scope_id):
    text = f"FACTORIAL-OOD-PHASE5B-PCA-v1|{scope_id}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

def disk_free_gib(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024**3)

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def add_test(tests, test_id, description, passed, detail):
    tests.append({
        "test_id": test_id,
        "description": description,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail)
    })
    if not passed:
        log(f"LEAKAGE/QA TEST FAIL [{test_id}]: {description} :: {detail}")

# ---------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------
log("=" * 78)
log("FACTORIAL-OOD Phase 5B starting/resuming")
log(f"Working directory: {ROOT}")
log(f"BLAS thread cap: {THREADS}")
log(f"Free disk at start: {disk_free_gib(ROOT):.2f} GiB")

required = [
    FEATHER,
    P2 / "P2_01_curated_compound_master.csv",
    P2 / "P2_02_curated_compound_context_table.csv",
    P2 / "P2_02a_curated_context_source_rows.csv",
    P4 / "outputs/P4_01_primary_outer_split_manifest_FROZEN.csv",
    P4 / "outputs/P4_04_nested_inner_split_manifest_FROZEN.csv",
    P4 / "outputs/P4_08_preprocessing_scope_registry.csv",
]
for p in required:
    if not p.exists():
        die(f"Required file missing: {p}")

log("Checking frozen Feather SHA-256...")
feather_sha = sha256_file(FEATHER)
if feather_sha != EXPECTED_FEATHER_SHA256:
    die(f"Feather checksum mismatch: {feather_sha}")
log("Feather SHA-256 PASS.")

# ---------------------------------------------------------------------
# Load frozen manifests
# ---------------------------------------------------------------------
master = pd.read_csv(P2 / "P2_01_curated_compound_master.csv")
ctx = pd.read_csv(P2 / "P2_02_curated_compound_context_table.csv")
src = pd.read_csv(P2 / "P2_02a_curated_context_source_rows.csv")
outer = pd.read_csv(P4 / "outputs/P4_01_primary_outer_split_manifest_FROZEN.csv")
inner = pd.read_csv(P4 / "outputs/P4_04_nested_inner_split_manifest_FROZEN.csv")

# Phase 4 stores PRIMARY plus four prospective V3 robustness split sets
# in the same nested-inner manifest. Phase 5 primary feature construction
# must use ONLY the frozen PRIMARY split set.
if "split_set" not in inner.columns:
    die("Nested inner manifest is missing required split_set column.")

available_split_sets = sorted(inner["split_set"].dropna().astype(str).unique())
log(f"Nested manifest split sets present: {available_split_sets}")

inner = inner[inner["split_set"].astype(str) == "PRIMARY"].copy()

if inner.empty:
    die("No PRIMARY records found in nested inner manifest.")

log(
    f"Restricted nested manifest to PRIMARY: "
    f"{len(inner)} rows across {inner['compound_id'].nunique()} compounds."
)

scope_registry = pd.read_csv(P4 / "outputs/P4_08_preprocessing_scope_registry.csv")

if len(master) != EXPECTED_PARENTS:
    die(f"Expected {EXPECTED_PARENTS} parents; found {len(master)}")
if len(ctx) != EXPECTED_CONTEXT_ROWS:
    die(f"Expected {EXPECTED_CONTEXT_ROWS} curated parent-context rows; found {len(ctx)}")
if len(src) != EXPECTED_RAW_ROWS:
    die(f"Expected {EXPECTED_RAW_ROWS} source rows; found {len(src)}")
if len(outer) != EXPECTED_PARENTS:
    die(f"Expected {EXPECTED_PARENTS} outer assignments; found {len(outer)}")

outer_folds = set(pd.to_numeric(outer["outer_fold"]).astype(int).unique())
if outer_folds != EXPECTED_OUTER_FOLDS:
    die(f"Outer folds mismatch: {outer_folds}")

if bool(outer["endpoint_outcomes_used"].astype(str).str.lower().isin(["true", "1"]).any()):
    die("Outer manifest indicates endpoint outcomes were used.")
if bool(inner["endpoint_outcomes_used"].astype(str).str.lower().isin(["true", "1"]).any()):
    die("Inner manifest indicates endpoint outcomes were used.")

fold_map = dict(zip(outer["compound_id"], pd.to_numeric(outer["outer_fold"]).astype(int)))
ctx["outer_fold"] = ctx["compound_id"].map(fold_map)
if ctx["outer_fold"].isna().any():
    die("At least one parent-context row has no frozen outer fold.")
ctx["outer_fold"] = ctx["outer_fold"].astype(int)

# Preserve exact Phase-2 row order as the canonical context-row index.
ctx = ctx.reset_index(drop=True)
ctx["context_row_index"] = np.arange(len(ctx), dtype=int)
ctx[[
    "context_row_index", "compound_id", "parent_compound_id",
    "modeling_context_id", "cell_type", "outer_fold"
]].to_csv(QA / "P5B_context_row_index_manifest.csv", index=False)

# ---------------------------------------------------------------------
# Discover HTTr schema WITHOUT reading endpoint columns into the PCA path
# ---------------------------------------------------------------------
try:
    mm = pa.memory_map(str(FEATHER), "r")
    reader = pa.ipc.open_file(mm)
    feather_names = list(reader.schema.names)
    del reader
    mm.close()
except Exception:
    # Feather V2 fallback
    schema_table = feather.read_table(FEATHER, memory_map=True)
    feather_names = list(schema_table.schema.names)
    del schema_table
    gc.collect()

required_meta = ["outcome_id", "epa_sample_id", "cell_type"]
for c in required_meta:
    if c not in feather_names:
        die(f"Required Feather metadata column absent: {c}")

# The source matrix has exactly three HTTr summaries per transcript:
# *_max, *_dose_at_max, *_AUC_neg
httr_cols = [
    c for c in feather_names
    if c.endswith("_max") or c.endswith("_AUC_neg")
]
# *_dose_at_max is already included by endswith("_max").
# Preserve original Feather column order.
if len(httr_cols) != EXPECTED_HTTR:
    die(f"Expected {EXPECTED_HTTR} HTTr columns; detected {len(httr_cols)}")

tox_cols = [c for c in feather_names if c.startswith("TOX21_")]
if len(tox_cols) != 41:
    die(f"Expected 41 Tox21 columns in source schema; detected {len(tox_cols)}")

with open(QA / "P5B_HTTr_feature_names.txt", "w", encoding="utf-8") as f:
    for c in httr_cols:
        f.write(c + "\n")

log(f"Schema PASS: {len(httr_cols)} HTTr columns detected.")
log("Tox21 columns were identified only for a schema-count QA check; their values are NEVER loaded.")

# ---------------------------------------------------------------------
# Build/reuse 4,021 x 16,821 parent-context median-collapsed HTTr cache
# ---------------------------------------------------------------------
CACHE_NPY = CACHE / "HTTr_parent_context_4021x16821_float32.npy"
CACHE_META = CACHE / "HTTr_parent_context_cache_metadata.json"

cache_key = {
    "feather_sha256": feather_sha,
    "p2_context_sha256": sha256_file(P2 / "P2_02_curated_compound_context_table.csv"),
    "p2_source_rows_sha256": sha256_file(P2 / "P2_02a_curated_context_source_rows.csv"),
    "rows": EXPECTED_CONTEXT_ROWS,
    "cols": EXPECTED_HTTR,
    "collapse_rule": "featurewise median across Phase-2 source rows for each parent-context"
}

reuse_cache = False
if CACHE_NPY.exists() and CACHE_META.exists():
    try:
        old_meta = json.load(open(CACHE_META, "r", encoding="utf-8"))
        if old_meta == cache_key:
            arr = np.load(CACHE_NPY, mmap_mode="r")
            if arr.shape == (EXPECTED_CONTEXT_ROWS, EXPECTED_HTTR):
                reuse_cache = True
                del arr
    except Exception:
        reuse_cache = False

if not reuse_cache:
    log("Building parent-context HTTr cache from frozen Feather source...")
    # Only metadata + HTTr columns are loaded. Endpoint values are not loaded.
    table = feather.read_table(
        FEATHER,
        columns=required_meta + httr_cols,
        memory_map=True,
        use_threads=True
    )
    if table.num_rows != EXPECTED_RAW_ROWS:
        die(f"Feather row count mismatch: {table.num_rows}")

    raw_meta = table.select(required_meta).to_pandas()
    # Numeric HTTr block -> float32. No Tox21 values enter this matrix.
    log("Converting HTTr Arrow table to float32 matrix...")
    hdf = table.select(httr_cols).to_pandas(use_threads=True)
    Xraw = hdf.to_numpy(dtype=np.float32, copy=True)
    del hdf, table
    gc.collect()

    if Xraw.shape != (EXPECTED_RAW_ROWS, EXPECTED_HTTR):
        die(f"Raw HTTr shape mismatch: {Xraw.shape}")
    if not np.isfinite(Xraw).all():
        die("Non-finite value detected in raw HTTr matrix.")

    raw_meta = raw_meta.reset_index(drop=True)
    raw_meta["raw_feather_row_index"] = np.arange(len(raw_meta), dtype=int)

    src_map = src[[
        "outcome_id", "epa_sample_id", "cell_type", "parent_compound_id"
    ]].copy()

    merged = raw_meta.merge(
        src_map,
        on=["outcome_id", "epa_sample_id", "cell_type"],
        how="left",
        validate="one_to_one"
    )
    if len(merged) != EXPECTED_RAW_ROWS or merged["parent_compound_id"].isna().any():
        die("Raw Feather rows could not be mapped one-to-one to Phase-2 source provenance.")

    # Build raw-index lookup for every curated parent-context.
    grp = merged.groupby(["parent_compound_id", "cell_type"], sort=False)["raw_feather_row_index"].apply(list)
    lookup = grp.to_dict()

    outmm = np.lib.format.open_memmap(
        CACHE_NPY,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_CONTEXT_ROWS, EXPECTED_HTTR)
    )

    duplicate_context_n = 0
    max_source_rows = 0
    for i, r in ctx.iterrows():
        key = (r["parent_compound_id"], r["cell_type"])
        inds = lookup.get(key)
        if not inds:
            die(f"No raw source row found for curated context: {key}")
        max_source_rows = max(max_source_rows, len(inds))
        if len(inds) == 1:
            outmm[i, :] = Xraw[inds[0], :]
        else:
            duplicate_context_n += 1
            outmm[i, :] = np.median(Xraw[np.asarray(inds, dtype=int), :], axis=0).astype(np.float32)

        if (i + 1) % 500 == 0:
            outmm.flush()
            log(f"  collapsed {i+1}/{EXPECTED_CONTEXT_ROWS} parent-context rows")

    outmm.flush()
    del outmm, Xraw, merged, raw_meta
    gc.collect()

    if duplicate_context_n != 116:
        log(f"WARNING: expected 116 duplicated parent-context keys; observed {duplicate_context_n}.")
    log(f"Parent-context collapse complete; duplicated keys={duplicate_context_n}, max source rows/key={max_source_rows}.")
    write_json(CACHE_META, cache_key)
else:
    log("Reusing validated parent-context HTTr cache.")

Xctx = np.load(CACHE_NPY, mmap_mode="r")
if Xctx.shape != (EXPECTED_CONTEXT_ROWS, EXPECTED_HTTR):
    die(f"Context cache shape mismatch: {Xctx.shape}")

# ---------------------------------------------------------------------
# Frozen inner assignments
# ---------------------------------------------------------------------
inner["outer_fold"] = pd.to_numeric(inner["outer_fold"]).astype(int)
inner["inner_fold"] = pd.to_numeric(inner["inner_fold"]).astype(int)

# Validate nested assignments using ONLY the frozen outer-training parents.
# The Phase-4 manifest intentionally carries the complete parent universe for
# each outer fold; outer-test parents must therefore be filtered out before
# interpreting the inner-fold assignment.
for f in sorted(EXPECTED_OUTER_FOLDS):
    outer_test_parents = set(
        outer.loc[outer["outer_fold"] == f, "compound_id"]
    )
    outer_train_parents = set(
        outer.loc[outer["outer_fold"] != f, "compound_id"]
    )

    sub_all = inner[inner["outer_fold"] == f].copy()
    sub = sub_all[sub_all["compound_id"].isin(outer_train_parents)].copy()

    if sub["compound_id"].duplicated().any():
        die(f"Outer fold {f}: duplicate outer-training parent in inner manifest.")

    if set(sub["compound_id"]) != outer_train_parents:
        missing = outer_train_parents - set(sub["compound_id"])
        extra = set(sub["compound_id"]) - outer_train_parents
        die(
            f"Outer fold {f}: inner parent coverage mismatch; "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    if set(sub["inner_fold"].unique()) != EXPECTED_INNER_FOLDS:
        die(
            f"Outer fold {f}: inner-fold domain mismatch after "
            f"outer-training restriction: {sorted(sub['inner_fold'].unique())}"
        )

    log(
        f"Outer fold {f}: nested manifest PASS — "
        f"{len(outer_train_parents)} training parents, "
        f"{len(outer_test_parents)} held-out parents."
    )

# ---------------------------------------------------------------------
# Scope fitter
# ---------------------------------------------------------------------
scope_summaries = []

def fit_pca_scope(scope_id, fit_idx, eval_sets, seed, outer_fold, inner_fold_label):
    """
    eval_sets: dict label -> context row indices
    """
    transformer_path = TRANSFORMERS / f"{scope_id}_transformer.npz"
    scores_path = SCORES / f"{scope_id}_scores.npz"
    done_path = QA / f"{scope_id}.DONE.json"

    if transformer_path.exists() and scores_path.exists() and done_path.exists():
        try:
            done = json.load(open(done_path, "r", encoding="utf-8"))
            if done.get("status") == "PASS":
                log(f"SKIP completed scope: {scope_id}")
                scope_summaries.append(done["summary"])
                return
        except Exception:
            pass

    t0 = time.time()
    log("-" * 78)
    log(f"FIT {scope_id}")
    log(f"  fit context rows: {len(fit_idx)}")
    for label, idx in eval_sets.items():
        log(f"  {label} context rows: {len(idx)}")
    log(f"  seed: {seed}")

    # Advanced row indexing intentionally creates only the current scope matrix.
    Xfit_full = np.asarray(Xctx[np.asarray(fit_idx, dtype=int), :], dtype=np.float32)
    if not np.isfinite(Xfit_full).all():
        die(f"{scope_id}: non-finite fit values.")

    # Variance filter learned strictly from fit scope.
    var = np.var(Xfit_full, axis=0, dtype=np.float64)
    mask = var > 0.0
    retained = int(mask.sum())
    zero_var = int((~mask).sum())
    if retained <= MAX_PCS:
        die(f"{scope_id}: too few retained HTTr variables ({retained}).")

    Xfit = np.ascontiguousarray(Xfit_full[:, mask], dtype=np.float32)
    del Xfit_full, var
    gc.collect()

    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    Xfit = scaler.fit_transform(Xfit)
    if not np.isfinite(Xfit).all():
        die(f"{scope_id}: non-finite standardized fit values.")

    ncomp = min(MAX_PCS, Xfit.shape[0] - 1, Xfit.shape[1])
    pca = PCA(
        n_components=ncomp,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=4,
        n_oversamples=10,
        copy=False
    )
    Zfit_all = pca.fit_transform(Xfit)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    hit = np.flatnonzero(cumulative >= TARGET_VARIANCE)
    k = int(hit[0] + 1) if len(hit) else int(ncomp)

    Zfit = np.asarray(Zfit_all[:, :k], dtype=np.float32)
    cum_at_retained = float(cumulative[k - 1])
    cum_at_256 = float(cumulative[-1])

    score_payload = {
        "fit_context_row_index": np.asarray(fit_idx, dtype=np.int32),
        "fit_scores": Zfit
    }

    # Transform validation/test sets using fit-scope scaler/PCA only.
    for label, idx in eval_sets.items():
        idx = np.asarray(idx, dtype=int)
        Xe = np.asarray(Xctx[idx, :][:, mask], dtype=np.float32)
        Xe = scaler.transform(Xe)
        Ze = pca.transform(Xe)[:, :k]
        Ze = np.asarray(Ze, dtype=np.float32)
        if not np.isfinite(Ze).all():
            die(f"{scope_id}: non-finite {label} PCA scores.")
        score_payload[f"{label}_context_row_index"] = idx.astype(np.int32)
        score_payload[f"{label}_scores"] = Ze
        del Xe, Ze
        gc.collect()

    # Save a version-portable numerical transformer, not a pickled sklearn object.
    np.savez(
        transformer_path,
        feature_mask=mask.astype(np.bool_),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_[:k], dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_[:k], dtype=np.float32),
        retained_components=np.asarray([k], dtype=np.int32),
        random_seed=np.asarray([seed], dtype=np.uint32)
    )
    np.savez(scores_path, **score_payload)

    summary = {
        "scope_id": scope_id,
        "outer_fold": int(outer_fold),
        "inner_fold": str(inner_fold_label),
        "fit_context_rows": int(len(fit_idx)),
        "fit_parent_n": int(ctx.iloc[np.asarray(fit_idx, dtype=int)]["compound_id"].nunique()),
        "retained_httr_features": retained,
        "zero_variance_httr_features": zero_var,
        "pca_components_retained": k,
        "target_variance": TARGET_VARIANCE,
        "cumulative_variance_retained": cum_at_retained,
        "cumulative_variance_at_max_components": cum_at_256,
        "max_components": MAX_PCS,
        "pca_engine": PCA_ENGINE,
        "random_seed": int(seed),
        "finite_fit_scores": bool(np.isfinite(Zfit).all()),
        "seconds": round(time.time() - t0, 3),
        "status": "PASS"
    }

    done_obj = {
        "status": "PASS",
        "transformer_sha256": sha256_file(transformer_path),
        "scores_sha256": sha256_file(scores_path),
        "summary": summary
    }
    write_json(done_path, done_obj)
    scope_summaries.append(summary)

    log(
        f"DONE {scope_id}: retained_features={retained}, zero_var={zero_var}, "
        f"PCs={k}, cumulative_var={cum_at_256:.6f}"
    )

    del Xfit, scaler, pca, Zfit_all, Zfit, score_payload
    gc.collect()

# ---------------------------------------------------------------------
# 5 OUTER transforms
# ---------------------------------------------------------------------
log("=" * 78)
log("Starting/resuming 5 OUTER HTTr transforms")

for f in range(5):
    fit_idx = ctx.index[ctx["outer_fold"] != f].to_numpy(dtype=int)
    test_idx = ctx.index[ctx["outer_fold"] == f].to_numpy(dtype=int)

    # Parent leakage assertion
    fit_par = set(ctx.iloc[fit_idx]["compound_id"])
    test_par = set(ctx.iloc[test_idx]["compound_id"])
    if fit_par & test_par:
        die(f"Outer {f}: parent leakage before PCA.")

    scope_id = f"P5B_OUTER{f}_FINAL_REFIT"
    seed = deterministic_seed(scope_id)
    fit_pca_scope(
        scope_id=scope_id,
        fit_idx=fit_idx,
        eval_sets={"outer_test": test_idx},
        seed=seed,
        outer_fold=f,
        inner_fold_label="FINAL_REFIT"
    )

# ---------------------------------------------------------------------
# 20 INNER transforms
# ---------------------------------------------------------------------
log("=" * 78)
log("Starting/resuming 20 INNER HTTr transforms")

for f in range(5):
    outer_test_parents = set(
        outer.loc[outer["outer_fold"] == f, "compound_id"]
    )
    outer_train_parents = set(
        outer.loc[outer["outer_fold"] != f, "compound_id"]
    )

    sub = inner[
        (inner["outer_fold"] == f)
        & (inner["compound_id"].isin(outer_train_parents))
    ][["compound_id", "inner_fold"]].copy()

    imap = dict(zip(sub["compound_id"], sub["inner_fold"]))

    for j in range(4):
        val_parents = set(sub.loc[sub["inner_fold"] == j, "compound_id"])
        fit_parents = set(sub.loc[sub["inner_fold"] != j, "compound_id"])

        if fit_parents & val_parents:
            die(f"Outer {f} inner {j}: inner train/validation parent overlap.")
        if fit_parents & outer_test_parents:
            die(f"Outer {f} inner {j}: inner train/outer test overlap.")
        if val_parents & outer_test_parents:
            die(f"Outer {f} inner {j}: inner validation/outer test overlap.")

        fit_idx = ctx.index[ctx["compound_id"].isin(fit_parents)].to_numpy(dtype=int)
        val_idx = ctx.index[ctx["compound_id"].isin(val_parents)].to_numpy(dtype=int)

        scope_id = f"P5B_OUTER{f}_INNER{j}"
        seed = deterministic_seed(scope_id)
        fit_pca_scope(
            scope_id=scope_id,
            fit_idx=fit_idx,
            eval_sets={"inner_val": val_idx},
            seed=seed,
            outer_fold=f,
            inner_fold_label=j
        )

# ---------------------------------------------------------------------
# Consolidated QA / leakage validation
# ---------------------------------------------------------------------
log("=" * 78)
log("Running consolidated Phase 5B leakage/QA tests")

# Reload done summaries so a resumed run is complete and deterministic.
scope_summaries = []
for f in range(5):
    ids = [f"P5B_OUTER{f}_FINAL_REFIT"] + [f"P5B_OUTER{f}_INNER{j}" for j in range(4)]
    for sid in ids:
        p = QA / f"{sid}.DONE.json"
        if not p.exists():
            die(f"Missing completed-scope marker: {sid}")
        obj = json.load(open(p, "r", encoding="utf-8"))
        if obj.get("status") != "PASS":
            die(f"Scope not PASS: {sid}")
        scope_summaries.append(obj["summary"])

summary_df = pd.DataFrame(scope_summaries).sort_values(
    ["outer_fold", "inner_fold"], kind="stable"
)
summary_df.to_csv(QA / "P5B_PCA_scope_summary.csv", index=False)

tests = []
add_test(tests, "P5B-T01", "Frozen Feather SHA-256 matches Phase 1", feather_sha == EXPECTED_FEATHER_SHA256, feather_sha)
add_test(tests, "P5B-T02", "Curated parent count is 2,017", len(master) == 2017, len(master))
add_test(tests, "P5B-T03", "Curated parent-context row count is 4,021", len(ctx) == 4021, len(ctx))
add_test(tests, "P5B-T04", "Raw source row count is 4,142", len(src) == 4142, len(src))
add_test(tests, "P5B-T05", "HTTr feature count is 16,821", len(httr_cols) == 16821, len(httr_cols))
add_test(tests, "P5B-T06", "Exactly 41 Tox21 columns exist but values were not loaded", len(tox_cols) == 41, len(tox_cols))
add_test(tests, "P5B-T07", "Outer manifest uses no endpoint outcomes", not bool(outer["endpoint_outcomes_used"].astype(str).str.lower().isin(["true","1"]).any()), "endpoint_outcomes_used=False")
add_test(tests, "P5B-T08", "Inner manifest uses no endpoint outcomes", not bool(inner["endpoint_outcomes_used"].astype(str).str.lower().isin(["true","1"]).any()), "endpoint_outcomes_used=False")
add_test(tests, "P5B-T09", "All five frozen outer folds represented", set(ctx["outer_fold"].unique()) == set(range(5)), sorted(ctx["outer_fold"].unique()))
add_test(tests, "P5B-T10", "All 25 required PCA scopes completed", len(summary_df) == 25, len(summary_df))
add_test(tests, "P5B-T11", "All PCA scopes finite", bool(summary_df["finite_fit_scores"].all()), summary_df["finite_fit_scores"].value_counts().to_dict())
add_test(tests, "P5B-T12", "No scope retained >256 PCs", bool((summary_df["pca_components_retained"] <= 256).all()), summary_df["pca_components_retained"].max())
add_test(tests, "P5B-T13", "Each scope retained at least 1 PC", bool((summary_df["pca_components_retained"] >= 1).all()), summary_df["pca_components_retained"].min())

# Explicit parent-overlap tests for all outer and inner scopes.
outer_overlap_total = 0
inner_train_val_overlap_total = 0
inner_train_test_overlap_total = 0
inner_val_test_overlap_total = 0

for f in range(5):
    tr = set(outer.loc[outer["outer_fold"] != f, "compound_id"])
    te = set(outer.loc[outer["outer_fold"] == f, "compound_id"])
    outer_overlap_total += len(tr & te)

    sub = inner[
        (inner["outer_fold"] == f)
        & (inner["compound_id"].isin(tr))
    ].copy()

    for j in range(4):
        itr = set(sub.loc[sub["inner_fold"] != j, "compound_id"])
        iva = set(sub.loc[sub["inner_fold"] == j, "compound_id"])
        inner_train_val_overlap_total += len(itr & iva)
        inner_train_test_overlap_total += len(itr & te)
        inner_val_test_overlap_total += len(iva & te)

add_test(tests, "P5B-T14", "Zero outer train/test parent overlap", outer_overlap_total == 0, outer_overlap_total)
add_test(tests, "P5B-T15", "Zero inner train/validation parent overlap", inner_train_val_overlap_total == 0, inner_train_val_overlap_total)
add_test(tests, "P5B-T16", "Zero inner-train/outer-test parent overlap", inner_train_test_overlap_total == 0, inner_train_test_overlap_total)
add_test(tests, "P5B-T17", "Zero inner-validation/outer-test parent overlap", inner_val_test_overlap_total == 0, inner_val_test_overlap_total)

# Transformer and score file count + checksums.
transformer_files = sorted(TRANSFORMERS.glob("*_transformer.npz"))
score_files = sorted(SCORES.glob("*_scores.npz"))
add_test(tests, "P5B-T18", "Exactly 25 transformer files exist", len(transformer_files) == 25, len(transformer_files))
add_test(tests, "P5B-T19", "Exactly 25 score files exist", len(score_files) == 25, len(score_files))

# Verify each score row index belongs to the intended frozen scope.
score_scope_mismatch = 0
for f in range(5):
    sid = f"P5B_OUTER{f}_FINAL_REFIT"
    dat = np.load(SCORES / f"{sid}_scores.npz")
    fitrows = dat["fit_context_row_index"].astype(int)
    testrows = dat["outer_test_context_row_index"].astype(int)
    score_scope_mismatch += int(np.sum(ctx.iloc[fitrows]["outer_fold"].to_numpy() == f))
    score_scope_mismatch += int(np.sum(ctx.iloc[testrows]["outer_fold"].to_numpy() != f))

    outer_train_parents = set(
        outer.loc[outer["outer_fold"] != f, "compound_id"]
    )
    sub = inner[
        (inner["outer_fold"] == f)
        & (inner["compound_id"].isin(outer_train_parents))
    ].copy()
    imap = dict(zip(sub["compound_id"], sub["inner_fold"]))

    for j in range(4):
        sid = f"P5B_OUTER{f}_INNER{j}"
        dat = np.load(SCORES / f"{sid}_scores.npz")
        fitrows = dat["fit_context_row_index"].astype(int)
        valrows = dat["inner_val_context_row_index"].astype(int)
        fitcomp = ctx.iloc[fitrows]["compound_id"].map(imap).to_numpy()
        valcomp = ctx.iloc[valrows]["compound_id"].map(imap).to_numpy()
        score_scope_mismatch += int(np.sum(fitcomp == j))
        score_scope_mismatch += int(np.sum(valcomp != j))

add_test(tests, "P5B-T20", "All saved score row indices obey frozen fit/evaluation scopes", score_scope_mismatch == 0, score_scope_mismatch)

tests_df = pd.DataFrame(tests)
tests_df.to_csv(QA / "P5B_feature_leakage_and_integrity_tests.csv", index=False)

all_pass = bool((tests_df["status"] == "PASS").all())
if not all_pass:
    die("At least one consolidated Phase 5B QA/leakage test failed.")

# ---------------------------------------------------------------------
# Software / seed / transformer registries
# ---------------------------------------------------------------------
software = {
    "generated_at": datetime.now().isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "pyarrow": pa.__version__,
    "joblib": joblib.__version__,
    "protocol": PROTOCOL,
    "threads": THREADS,
    "pca_engine": PCA_ENGINE,
    "max_pcs": MAX_PCS,
    "target_variance": TARGET_VARIANCE,
    "feather_sha256": feather_sha
}
write_json(QA / "P5B_software_and_method_lock.json", software)

registry = []
for p in transformer_files:
    scope_id = p.name.replace("_transformer.npz", "")
    registry.append({
        "scope_id": scope_id,
        "transformer_file": p.name,
        "transformer_sha256": sha256_file(p),
        "size_bytes": p.stat().st_size
    })
pd.DataFrame(registry).to_csv(QA / "P5B_transformer_checksum_registry.csv", index=False)

score_registry = []
for p in score_files:
    scope_id = p.name.replace("_scores.npz", "")
    score_registry.append({
        "scope_id": scope_id,
        "score_file": p.name,
        "score_sha256": sha256_file(p),
        "size_bytes": p.stat().st_size
    })
pd.DataFrame(score_registry).to_csv(QA / "P5B_score_checksum_registry.csv", index=False)

# ---------------------------------------------------------------------
# Global SHA256 manifest
# ---------------------------------------------------------------------
manifest_targets = []
for folder in [QA, SCORES, TRANSFORMERS]:
    manifest_targets.extend([p for p in folder.rglob("*") if p.is_file()])

sha_lines = []
for p in sorted(manifest_targets):
    rel = p.relative_to(OUT)
    sha_lines.append(f"{sha256_file(p)}  {rel}")
with open(OUT / "SHA256SUMS_PHASE5B.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sha_lines) + "\n")

readme = f"""FACTORIAL-OOD Phase 5B
Protocol: {PROTOCOL}

PASS status: {all_pass}
HTTr source: {FEATHER.name}
HTTr source SHA256: {feather_sha}
Curated parent-context rows: {EXPECTED_CONTEXT_ROWS}
HTTr variables: {EXPECTED_HTTR}
Outer PCA scopes: 5
Inner PCA scopes: 20
Total PCA scopes: 25
PCA: randomized, max {MAX_PCS} components, retain first components reaching {TARGET_VARIANCE:.0%} variance if achieved before the cap.
All variance filtering, scaling and PCA fitting are training-scope-only.
No Tox21 endpoint values are loaded or used by the PCA pipeline.

UPLOAD PACKAGE:
Phase5B_UPLOAD_TO_CHAT.zip contains QA/manifests + all fold score matrices.
It intentionally excludes the large transformer component archives, but includes their SHA256 registry.

FULL LOCAL OUTPUT:
The transformers/ directory contains all 25 numerical PCA transformer archives and should be retained locally for reproducibility.
"""
with open(OUT / "README_PHASE5B.txt", "w", encoding="utf-8") as f:
    f.write(readme)

# ---------------------------------------------------------------------
# Create upload-friendly ZIP (exclude large transformer arrays/cache)
# ---------------------------------------------------------------------
upload_zip = OUT / "Phase5B_UPLOAD_TO_CHAT.zip"
log("Creating upload-friendly Phase 5B ZIP...")
with zipfile.ZipFile(upload_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(OUT / "README_PHASE5B.txt", arcname="README_PHASE5B.txt")
    z.write(OUT / "SHA256SUMS_PHASE5B.txt", arcname="SHA256SUMS_PHASE5B.txt")
    for p in sorted(QA.rglob("*")):
        if p.is_file():
            z.write(p, arcname=str(Path("qa") / p.name))
    for p in sorted(SCORES.rglob("*")):
        if p.is_file():
            z.write(p, arcname=str(Path("scores") / p.name))

# ---------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------
final_summary = {
    "status": "PASS",
    "phase": "5B",
    "protocol": PROTOCOL,
    "outer_scopes": 5,
    "inner_scopes": 20,
    "total_scopes": 25,
    "all_tests_pass": all_pass,
    "qa_test_n": int(len(tests_df)),
    "upload_zip": str(upload_zip),
    "upload_zip_sha256": sha256_file(upload_zip),
    "upload_zip_size_bytes": upload_zip.stat().st_size,
    "transformer_directory": str(TRANSFORMERS),
    "free_disk_gib_end": round(disk_free_gib(ROOT), 3)
}
write_json(OUT / "P5B_FINAL_STATUS.json", final_summary)

log("=" * 78)
log("PHASE 5B LOCAL COMPUTATION PASS")
log(f"QA tests: {len(tests_df)}/{len(tests_df)} PASS")
log(f"Upload this file to ChatGPT: {upload_zip}")
log(f"Upload ZIP size: {upload_zip.stat().st_size / (1024**2):.1f} MiB")
log(f"Upload ZIP SHA256: {final_summary['upload_zip_sha256']}")
log(f"Retain locally: {TRANSFORMERS}")
log(f"Free disk at end: {final_summary['free_disk_gib_end']:.2f} GiB")
log("=" * 78)
