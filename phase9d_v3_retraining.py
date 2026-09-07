#!/usr/bin/env python3
"""
FACTORIAL-OOD Phase 9D — four frozen V3 grouped-split robustness retraining

Scientific status
-----------------
This is a robustness-only execution of the already frozen Phase-9 Gate-D plan.
It does NOT replace the Phase-6/7/7B/8 primary analysis.

Core rules
----------
1. Run ALL four prospectively frozen V3 split sets (V3_ALT01..V3_ALT04).
2. For each split set, refit every split-dependent preprocessing object inside
   the relevant training scope:
     - structure descriptor StandardScaler (5 outer + 20 inner scopes)
     - HTTr variance filter / StandardScaler / randomized PCA (5 + 20 scopes)
3. Run the byte-identical validated Phase-6 model engine against each V3
   workspace. Thus model grids, tuning metric/budget, sample weighting, Platt
   calibration, MCC thresholding, Route-B mean aggregation and late-fusion
   weight selection remain unchanged.
4. Recompute the PRIMARY v0.2 factorial support labels for each V3 split from
   that split's training data only:
     chemical: C0 if represented scaffold OR max Morgan Tanimoto >= 0.50
     biological: B0 if median test-parent HTTr 5NN distance <= outer-training
                 parent leave-one-out q75; B1 otherwise.
   Biological support uses the frozen Phase-3 q75 construction (full 256-PC
   Phase-3 support PCA, not the Phase-5B modeling PCA).
5. Store immutable OOF predictions for every V3 split before robustness stats.
6. Use 2,000 paired parent bootstraps per endpoint and split set.
7. No favorable split, endpoint, threshold, model or failed direction may be
   selected or omitted.

Resume safety
-------------
- Preprocessing scopes are checkpointed.
- The Phase-6 engine is checkpointed per endpoint/fold/family/modality.
- Rerunning the same command skips completed work.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("P9D_OMP_THREADS", "8"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("P9D_BLAS_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("P9D_BLAS_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("P9D_BLAS_THREADS", "1"))

import argparse
import gc
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import pyarrow as pa
    import pyarrow.feather as feather
except Exception:
    pa = None
    feather = None

# -----------------------------------------------------------------------------
# Frozen identifiers / hashes
# -----------------------------------------------------------------------------
PHASE9_LOCK_SHA = "55f769210fa70d1be54904e1e59c7a639175811018a8373264fe3f3226027a01"
PHASE6_REGISTRY_SHA = "10005b76d389dcc73466ef377cf4f5d2ee0787ff71b95523d08ab2b988276b57"
PHASE6_ENGINE_SHA = "d0aace6c46a764a4e2ff4ecdb1081d38cd307a7a7a9939aa3824c91eaab8939a"

HASHES = {
    "P4_V3": "6fc68b0a51e1d696643969433dc6caea46a6eb6fd52075d00963f9113833aff1",
    "P4_INNER": "a009416c11ba18c5d62e86c8aa91a3b1f5cfdb3c8bba84d1793eb01537dda6d2",
    "P2_MASTER": "df4c35d971b3fd0de99d10a1a955959592177feec04d45899902f810e4188ebd",
    "P2_CONTEXT": "48a08466f04a5a295b9312c813669133f49417b848f2315e01e97d4ee51da1c8",
    "P2_SOURCE": "f90db9d7c757d27d3baf3584516a7b2f9c461dca35bcb0ec85e85be7e2f8320e",
    "P2_LABELS": "1179aa99487f95958e9906013e7f3711948914bd3e4dbfc5ae75c4610c44e061",
    "P5_MORGAN": "88ae993f00250578caba4855460db0d9eaa9bc89aefbbb2b1fff7a66d76d99e6",
    "P5_DESC": "cdb8ac38ccf0433914230dad83e1c72f1095078027b35106efb7e1b39cf03062",
    "P5_STRUCT_MANIFEST": "6652b49f959fcd8f3d729969bb9caf9e34fe71c0df97aa064b11f5b7661dd718",
    "P5_FUSION": "a7dcf10f70036337d539744eea7d6ba6e7082ec5fa1b1f1a7ba21c1207a85dff",
    "FEATHER": "480f3aad0e84acce7b5affb54fdd9027258e93275de09cf3f21b97971d17cc8a",
    "P9A_RE": "f5a4a563866f43ee28bf36572dbde8b5d60e63012ab19e3802e056487ca95bff",
}

SPLIT_SETS = ["V3_ALT01", "V3_ALT02", "V3_ALT03", "V3_ALT04"]
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
FAMILIES = ["ELASTIC_NET", "RANDOM_FOREST", "HIST_GRAD_BOOST"]
MODALITIES = ["STRUCTURE", "HTTR", "EARLY_FUSION", "LATE_FUSION"]
MODEL_KEYS = [(f, m) for f in FAMILIES for m in MODALITIES]
CELLS = ["C0/B0", "C1/B0", "C0/B1", "C1/B1"]
CI = {x: i for i, x in enumerate(CELLS)}
METRICS = [
    "AUROC", "AUPRC", "MCC", "balanced_MCC", "Brier", "Brier_skill_score",
    "ECE_10bin", "balanced_accuracy", "sensitivity", "specificity",
]
DESCRIPTOR_COLS = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount",
    "FractionCSP3", "HeavyAtomCount", "FormalCharge",
]
BOOTSTRAP_N = 2000
MAX_PCS = 256
TARGET_VARIANCE = 0.80
EPS = 1e-12

# -----------------------------------------------------------------------------
# CLI / paths
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem")
    ap.add_argument("--split-set", choices=["ALL"] + SPLIT_SETS, default="ALL")
    ap.add_argument("--rf-jobs", type=int, default=8)
    ap.add_argument("--analysis-jobs", type=int, default=8)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--keep-workspaces", action="store_true", default=True)
    return ap.parse_args()

ARGS = parse_args()
ROOT = Path(ARGS.root).resolve()
P2 = ROOT / "Phase2"
P4 = ROOT / "Phase4" / "outputs"
P5 = ROOT / "Phase5" / "outputs"
P5B = ROOT / "Phase5B_Output"
OUT = ROOT / "Phase9_Output" / "P9D"
WORK = OUT / "workspaces"
SPLIT_OUT = OUT / "split_outputs"
LOGS = OUT / "logs"
for d in [OUT, WORK, SPLIT_OUT, LOGS]:
    d.mkdir(parents=True, exist_ok=True)
LOG = LOGS / "phase9d_run.log"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def link_or_copy(src, dst):
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    if not len(idx):
        return q
    pv = p[idx]
    order = np.argsort(pv)
    r = pv[order]
    m = len(r)
    a = r * m / np.arange(1, m + 1)
    a = np.minimum.accumulate(a[::-1])[::-1]
    a = np.minimum(a, 1.0)
    q[idx[order]] = a
    return q


def boot_summary(est, vals):
    vals = np.asarray(vals, float)
    v = vals[np.isfinite(vals)]
    frac = len(v) / len(vals)
    if not np.isfinite(est) or not len(v):
        return dict(
            estimate=est, ci_low=np.nan, ci_high=np.nan, bootstrap_se=np.nan,
            bootstrap_p_two_sided=np.nan, bootstrap_valid_n=len(v),
            bootstrap_valid_fraction=frac, bootstrap_supported=False,
        )
    lo, hi = np.quantile(v, [0.025, 0.975])
    se = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    dev = v - est
    p = (1 + np.sum(np.abs(dev) >= abs(est))) / (len(v) + 1)
    return dict(
        estimate=float(est), ci_low=float(lo), ci_high=float(hi), bootstrap_se=se,
        bootstrap_p_two_sided=float(min(1, p)), bootstrap_valid_n=len(v),
        bootstrap_valid_fraction=float(frac), bootstrap_supported=bool(frac >= 0.90),
    )


def seed_for(split_set, endpoint):
    text = "|".join(map(str, ["FACTORIAL-OOD", "P9D", "v1", split_set, endpoint, "PARENT_CLUSTER_BOOTSTRAP"]))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def pca_seed(scope_id):
    # Same deterministic Phase-5B seed principle; scope IDs in each isolated V3
    # workspace intentionally retain the original OUTER/INNER nomenclature.
    text = f"FACTORIAL-OOD-PHASE5B-PCA-v1|{scope_id}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def support_pca_seed(outer_fold):
    # Exact Phase-3 support-PCA seed rule.
    return 20260831 + int(outer_fold)


def balanced_mcc_from_counts(tp, fn, tn, fp):
    sen = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fn) > 0)
    spe = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan, dtype=float), where=(tn + fp) > 0)
    den2 = 1 - (sen - spe) ** 2
    out = np.full_like(sen, np.nan, dtype=float)
    ok = np.isfinite(sen) & np.isfinite(spe) & (den2 > 1e-14)
    out[ok] = (sen[ok] + spe[ok] - 1) / np.sqrt(den2[ok])
    return out, sen, spe


def weighted_rank_metrics(y, p, W):
    """Exact tie-aware weighted AUROC/AP; vectorized over bootstrap rows."""
    y = np.asarray(y, int); p = np.asarray(p, float); W = np.asarray(W, float)
    Bn = W.shape[0]; auc = np.full(Bn, np.nan); ap = np.full(Bn, np.nan)
    if len(y) == 0:
        return auc, ap
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]; ys = y[order]; Ws = W[:, order]
    starts = np.r_[0, 1 + np.where(ps[1:] != ps[:-1])[0]]
    pos_g = np.add.reduceat(Ws * ys[None, :], starts, axis=1)
    neg_g = np.add.reduceat(Ws * (1 - ys)[None, :], starts, axis=1)
    Ptot = pos_g.sum(axis=1); Ntot = neg_g.sum(axis=1); ok = (Ptot > 0) & (Ntot > 0)
    if np.any(ok):
        cpos = np.cumsum(pos_g, axis=1); call = np.cumsum(pos_g + neg_g, axis=1)
        prec = np.divide(cpos, call, out=np.zeros_like(cpos), where=call > 0)
        ap[ok] = np.sum(prec[ok] * pos_g[ok], axis=1) / Ptot[ok]
        cneg = np.cumsum(neg_g, axis=1)
        neg_below = Ntot[:, None] - cneg
        num = np.sum(pos_g * (neg_below + 0.5 * neg_g), axis=1)
        auc[ok] = num[ok] / (Ptot[ok] * Ntot[ok])
    return auc, ap


def weighted_ece(y, p, W, nbins=10):
    y = np.asarray(y, float); p = np.asarray(p, float); W = np.asarray(W, float)
    Bn = W.shape[0]; tot = W.sum(axis=1); out = np.zeros(Bn, float); good = tot > 0
    bins = np.minimum((np.clip(p, 0, 1) * nbins).astype(int), nbins - 1)
    for k in range(nbins):
        idx = np.where(bins == k)[0]
        if not len(idx):
            continue
        wb = W[:, idx]; cnt = wb.sum(axis=1); m = cnt > 0
        if not np.any(m):
            continue
        ys = wb @ y[idx]; ps = wb @ p[idx]
        obs = np.divide(ys, cnt, out=np.zeros_like(cnt), where=m)
        pred = np.divide(ps, cnt, out=np.zeros_like(cnt), where=m)
        out[m] += cnt[m] / tot[m] * np.abs(obs[m] - pred[m])
    out[~good] = np.nan
    return out


def weighted_metrics(y, P, Q, W):
    """Return B x J x K metrics for fixed rows and bootstrap integer weights."""
    y = np.asarray(y, int); P = np.asarray(P, float); Q = np.asarray(Q, int); W = np.asarray(W, float)
    Bn, n = W.shape; J = P.shape[1]; K = len(METRICS)
    out = np.full((Bn, J, K), np.nan, float)
    tot = W.sum(axis=1); y1 = y.astype(float)
    prev = np.divide(W @ y1, tot, out=np.full(Bn, np.nan), where=tot > 0)
    for j in range(J):
        q = Q[:, j]; p = P[:, j]
        tp = W @ ((y == 1) & (q == 1)).astype(float)
        fn = W @ ((y == 1) & (q == 0)).astype(float)
        tn = W @ ((y == 0) & (q == 0)).astype(float)
        fp = W @ ((y == 0) & (q == 1)).astype(float)
        bm, sen, spe = balanced_mcc_from_counts(tp, fn, tn, fp); ba = (sen + spe) / 2
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(tp * tn - fp * fn, den, out=np.full(Bn, np.nan), where=den > 0)
        err = (p - y) ** 2
        brier = np.divide(W @ err, tot, out=np.full(Bn, np.nan), where=tot > 0)
        ref = prev * (1 - prev)
        bss = np.divide(brier, ref, out=np.full(Bn, np.nan), where=ref > 0); bss = 1 - bss
        auc, ap = weighted_rank_metrics(y, p, W); ece = weighted_ece(y, p, W, 10)
        vals = {
            "AUROC": auc, "AUPRC": ap, "MCC": mcc, "balanced_MCC": bm,
            "Brier": brier, "Brier_skill_score": bss, "ECE_10bin": ece,
            "balanced_accuracy": ba, "sensitivity": sen, "specificity": spe,
        }
        for k, met in enumerate(METRICS):
            out[:, j, k] = vals[met]
    return out

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
def locate_inputs():
    return {
        "lock": ROOT / "P9_00_robustness_lock_v1.0.json",
        "engine": ROOT / "phase9d_phase6_engine.py",
        "v3": P4 / "P4_03_v3_alternative_outer_splits_FROZEN.csv",
        "inner": P4 / "P4_04_nested_inner_split_manifest_FROZEN.csv",
        "master": P2 / "P2_01_curated_compound_master.csv",
        "context": P2 / "P2_02_curated_compound_context_table.csv",
        "source": P2 / "P2_02a_curated_context_source_rows.csv",
        "labels": P2 / "P2_04_endpoint_label_matrix.csv",
        "morgan": P5 / "P5_01_Morgan_r2_2048_parent_matrix.npz",
        "desc": P5 / "P5_02_physicochemical_descriptors_raw.csv",
        "struct_manifest": P5 / "P5_03_structure_feature_manifest.csv",
        "fusion": P5 / "P5_05_fusion_context_row_manifest.csv",
        "feather": ROOT / "chemical_httr_assay_aggregated.feather",
        "p7_re": ROOT / "Phase7_Output" / "P7_09_random_effects_endpoint_synthesis.csv",
        "p7b_re": ROOT / "Phase7B_Output" / "P7B_05_random_effects_synthesis.csv",
        "p9a_re": ROOT / "Phase9_Output" / "P9A" / "P9A_04_random_effects.csv",
    }


def preflight():
    files = locate_inputs()
    tests = []
    def T(tid, desc, ok, detail=""):
        tests.append({"test_id": tid, "description": desc, "status": "PASS" if bool(ok) else "FAIL", "detail": str(detail)})

    for k, p in files.items():
        T(f"P9D-P-{k}", f"Required input exists: {p.name}", p.exists(), p)
    if any(x["status"] == "FAIL" for x in tests):
        df = pd.DataFrame(tests)
        df.to_csv(OUT / "P9D_PREFLIGHT_TESTS.csv", index=False)
        return df

    T("P9D-P01", "Phase9 scientific lock hash matches", sha256(files["lock"]) == PHASE9_LOCK_SHA, sha256(files["lock"]))
    T("P9D-P02", "Bundled Phase6 engine hash matches validated v1.1", sha256(files["engine"]) == PHASE6_ENGINE_SHA, sha256(files["engine"]))
    for key, fk in [
        ("P4_V3", "v3"), ("P4_INNER", "inner"), ("P2_MASTER", "master"),
        ("P2_CONTEXT", "context"), ("P2_SOURCE", "source"), ("P2_LABELS", "labels"),
        ("P5_MORGAN", "morgan"), ("P5_DESC", "desc"), ("P5_STRUCT_MANIFEST", "struct_manifest"),
        ("P5_FUSION", "fusion"), ("FEATHER", "feather"),
    ]:
        T(f"P9D-H-{key}", f"Frozen input hash matches: {key}", sha256(files[fk]) == HASHES[key], sha256(files[fk]))

    v3 = pd.read_csv(files["v3"])
    inn = pd.read_csv(files["inner"])
    T("P9D-P03", "Exactly four V3 split sets x 2017 parents", len(v3) == 8068 and v3.split_set.nunique() == 4 and (v3.groupby("split_set").size() == 2017).all(), v3.groupby("split_set").size().to_dict())
    T("P9D-P04", "V3 split sets are exactly ALT01..ALT04", set(v3.split_set) == set(SPLIT_SETS), sorted(v3.split_set.unique()))
    T("P9D-P05", "V3 assignments remain outcome-blind", not v3.endpoint_outcomes_used.astype(bool).any(), v3.endpoint_outcomes_used.value_counts().to_dict())
    alti = inn[inn.split_set.isin(SPLIT_SETS)]
    T("P9D-P06", "All 4x5 V3 outer scopes have four frozen inner folds", alti.groupby(["split_set", "outer_fold"]).inner_fold.nunique().eq(4).all() and alti.groupby(["split_set", "outer_fold"]).ngroups == 20, alti.groupby(["split_set", "outer_fold"]).inner_fold.nunique().value_counts().to_dict())
    T("P9D-P07", "V3 inner assignments remain outcome-blind", not alti.endpoint_outcomes_used.astype(bool).any(), alti.endpoint_outcomes_used.value_counts().to_dict())

    labels = pd.read_csv(files["labels"]).set_index("compound_id")
    # Feasibility only: no performance and no split modification.
    support_issues = []
    for ss in SPLIT_SETS:
        o = v3[v3.split_set == ss].set_index("compound_id").outer_fold.astype(int)
        ii = alti[alti.split_set == ss]
        for ep in PRIMARY_ENDPOINTS:
            y = labels[ep].dropna()
            for f in range(5):
                tr = o[o != f].index.intersection(y.index)
                te = o[o == f].index.intersection(y.index)
                if y.loc[tr].nunique() < 2 or y.loc[te].nunique() < 2:
                    support_issues.append((ss, ep, f, "outer"))
                sub = ii[ii.outer_fold == f].set_index("compound_id").inner_fold.astype(int)
                for j in range(4):
                    itr = sub[sub != j].index.intersection(y.index)
                    iva = sub[sub == j].index.intersection(y.index)
                    if y.loc[itr].nunique() < 2 or y.loc[iva].nunique() < 2:
                        support_issues.append((ss, ep, f, f"inner{j}"))
    T("P9D-P08", "All V3 primary endpoint outer/inner scopes contain both classes", len(support_issues) == 0, support_issues[:10])

    sm = pd.read_csv(files["struct_manifest"])
    z = np.load(files["morgan"], allow_pickle=False)
    T("P9D-P09", "Morgan matrix is 2017x2048 and structure manifest is 2017", z["morgan_bits"].shape == (2017, 2048) and len(sm) == 2017, (z["morgan_bits"].shape, len(sm)))
    fus = pd.read_csv(files["fusion"])
    T("P9D-P10", "Fusion context manifest has exactly 4021 indexed rows", len(fus) == 4021 and fus.context_row_index.nunique() == 4021 and set(fus.context_row_index) == set(range(4021)), len(fus))
    T("P9D-P11", "P9A primary-definition random-effects hash matches independently validated package", sha256(files["p9a_re"]) == HASHES["P9A_RE"], sha256(files["p9a_re"]))

    df = pd.DataFrame(tests)
    df.to_csv(OUT / "P9D_PREFLIGHT_TESTS.csv", index=False)
    status = {
        "phase": "9D", "type": "preflight", "performance_computed": False,
        "status": "PASS" if (df.status == "PASS").all() else "FAIL",
        "pass_n": int((df.status == "PASS").sum()), "total_n": len(df),
        "phase9_lock_sha256": PHASE9_LOCK_SHA, "phase6_engine_sha256": PHASE6_ENGINE_SHA,
    }
    write_json(OUT / "P9D_PREFLIGHT_STATUS.json", status)
    execution_registry = {
        "project": "FACTORIAL-OOD", "phase": "9D", "version": "v1.0",
        "phase9_scientific_lock_sha256": PHASE9_LOCK_SHA,
        "p9d_driver_sha256": sha256(Path(__file__).resolve()),
        "phase6_engine_sha256": PHASE6_ENGINE_SHA,
        "phase6_computational_registry_sha256": PHASE6_REGISTRY_SHA,
        "v3_outer_manifest_sha256": HASHES["P4_V3"],
        "v3_inner_manifest_sha256": HASHES["P4_INNER"],
        "v3_split_sets": SPLIT_SETS,
        "structure_preprocessing": "StandardScaler descriptors refit in every V3 outer/inner training scope; Morgan bits invariant",
        "httr_model_preprocessing": "Phase5B rule refit in every V3 outer/inner training scope: zero-variance filter + StandardScaler + randomized PCA, max 256 PCs, retain first PCs reaching 80% if achieved",
        "factorial_support": "Recomputed within each V3 outer fold from training data only; chemical hybrid 0.50; biological Phase3-style q75 median 5NN distance",
        "modeling": "byte-identical validated Phase6 v1.1 engine",
        "bootstrap": {"unit": "unique parent compound", "resamples": BOOTSTRAP_N, "paired_across_models_within_endpoint_split": True},
        "gateD": ">=3/4 V3 split-level across-endpoint effects must retain the primary direction; >=2/4 reverse => downgrade",
        "governance": "All four frozen V3 splits must be retained; no performance-based split/endpoint/model selection or retuning.",
    }
    write_json(OUT / "P9D_00_execution_registry.json", execution_registry)
    return df

# -----------------------------------------------------------------------------
# Build/reuse invariant 4021x16821 HTTr cache
# -----------------------------------------------------------------------------
def ensure_httr_cache(files):
    cache_dir = P5B / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "HTTr_parent_context_4021x16821_float32.npy"
    if cache.exists():
        try:
            x = np.load(cache, mmap_mode="r")
            if x.shape == (4021, 16821) and x.dtype == np.float32:
                log(f"Reusing Phase5B HTTr parent-context cache: {cache}")
                return cache
        except Exception:
            pass
    if pa is None or feather is None:
        raise RuntimeError("HTTr cache missing and pyarrow unavailable. Restore Phase5B_Output/cache or install pyarrow.")
    log("HTTr cache missing; rebuilding the invariant Phase5B parent-context cache from frozen Feather source.")
    ctx = pd.read_csv(files["context"]).reset_index(drop=True)
    src = pd.read_csv(files["source"])
    try:
        mm = pa.memory_map(str(files["feather"]), "r")
        reader = pa.ipc.open_file(mm)
        names = list(reader.schema.names)
        del reader
        mm.close()
    except Exception:
        schema_table = feather.read_table(files["feather"], memory_map=True)
        names = list(schema_table.schema.names)
        del schema_table
        gc.collect()
    httr_cols = [c for c in names if c.endswith("_max") or c.endswith("_AUC_neg")]
    if len(httr_cols) != 16821:
        raise RuntimeError(f"Expected 16821 HTTr columns; found {len(httr_cols)}")
    meta_cols = ["outcome_id", "epa_sample_id", "cell_type"]
    table = feather.read_table(files["feather"], columns=meta_cols + httr_cols, memory_map=True, use_threads=True)
    raw_meta = table.select(meta_cols).to_pandas().reset_index(drop=True)
    Xraw = table.select(httr_cols).to_pandas(use_threads=True).to_numpy(dtype=np.float32, copy=True)
    del table
    gc.collect()
    raw_meta["raw_feather_row_index"] = np.arange(len(raw_meta), dtype=int)
    src_map = src[["outcome_id", "epa_sample_id", "cell_type", "parent_compound_id"]].copy()
    merged = raw_meta.merge(src_map, on=meta_cols, how="left", validate="one_to_one")
    if merged.parent_compound_id.isna().any() or len(merged) != 4142:
        raise RuntimeError("Could not reconstruct one-to-one raw source provenance for HTTr cache")
    lookup = merged.groupby(["parent_compound_id", "cell_type"], sort=False)["raw_feather_row_index"].apply(list).to_dict()
    outmm = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(4021, 16821))
    for i, r in ctx.iterrows():
        inds = lookup.get((r["parent_compound_id"], r["cell_type"]))
        if not inds:
            raise RuntimeError(f"No raw HTTr source for context row {i}")
        outmm[i] = Xraw[inds[0]] if len(inds) == 1 else np.median(Xraw[np.asarray(inds, int)], axis=0).astype(np.float32)
        if (i + 1) % 500 == 0:
            outmm.flush(); log(f"  rebuilt HTTr cache {i+1}/4021 rows")
    outmm.flush()
    del outmm, Xraw, merged, raw_meta
    gc.collect()
    return cache

# -----------------------------------------------------------------------------
# V3 workspace preprocessing
# -----------------------------------------------------------------------------
def workspace_paths(ss):
    w = WORK / ss
    return {
        "root": w,
        "p4": w / "Phase4" / "outputs",
        "p5": w / "Phase5" / "outputs",
        "p5b": w / "Phase5B_Output" / "scores",
        "phase6": w / "Phase6_Output",
        "prep": w / "P9D_preprocess",
    }


def create_workspace_manifests(ss, files, v3, inner):
    wp = workspace_paths(ss)
    for d in wp.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    # Phase2 immutable files are linked as a directory when possible.
    w_p2 = wp["root"] / "Phase2"
    if not w_p2.exists():
        try:
            w_p2.symlink_to(P2.resolve(), target_is_directory=True)
        except Exception:
            w_p2.mkdir(parents=True, exist_ok=True)
            for name in ["P2_04_endpoint_label_matrix.csv"]:
                link_or_copy(P2 / name, w_p2 / name)

    o = v3[v3.split_set == ss].copy().sort_values("compound_id").reset_index(drop=True)
    # The byte-identical P6 engine expects a file playing the PRIMARY role; this
    # isolated workspace maps the V3 assignment into that role without changing
    # any parent/fold value.
    o.to_csv(wp["p4"] / "P4_01_primary_outer_split_manifest_FROZEN.csv", index=False)
    ii = inner[inner.split_set == ss].copy()
    ii["source_split_set"] = ii["split_set"]
    ii["split_set"] = "PRIMARY"
    ii.to_csv(wp["p4"] / "P4_04_nested_inner_split_manifest_FROZEN.csv", index=False)

    # Invariant Phase5 raw structure/context resources.
    for name in [
        "P5_01_Morgan_r2_2048_parent_matrix.npz",
        "P5_02_physicochemical_descriptors_raw.csv",
        "P5_03_structure_feature_manifest.csv",
        "P5_05_fusion_context_row_manifest.csv",
    ]:
        link_or_copy(P5 / name, wp["p5"] / name)
    return o, ii


def build_descriptor_scalers(ss, wp, outer_ss, inner_ss):
    outpath = wp["p5"] / "P5_04_descriptor_scaler_registry.csv"
    done = wp["prep"] / "descriptor_scalers.DONE.json"
    if outpath.exists() and done.exists():
        obj = json.load(open(done))
        if obj.get("status") == "PASS" and obj.get("sha256") == sha256(outpath):
            log(f"{ss}: descriptor scaler registry already PASS")
            return
    desc = pd.read_csv(P5 / "P5_02_physicochemical_descriptors_raw.csv")
    X = desc[DESCRIPTOR_COLS].to_numpy(dtype=np.float64)
    ids = desc.compound_id.astype(str).to_numpy()
    row = {p: i for i, p in enumerate(ids)}
    fold_map = dict(zip(outer_ss.compound_id.astype(str), outer_ss.outer_fold.astype(int)))
    rows = []
    def add_scope(scope_id, f, ilabel, fit_par, eval_par):
        idx = np.asarray([row[p] for p in fit_par], int)
        sc = StandardScaler().fit(X[idx])
        r = {"scope_id": scope_id, "outer_fold": f, "inner_fold": ilabel, "fit_parent_n": len(fit_par), "eval_parent_n": len(eval_par)}
        for k, c in enumerate(DESCRIPTOR_COLS):
            r[f"{c}_mean"] = float(sc.mean_[k])
            r[f"{c}_scale"] = float(sc.scale_[k]) if sc.scale_[k] != 0 else 1.0
        rows.append(r)
    for f in range(5):
        fit_par = sorted(outer_ss.loc[outer_ss.outer_fold != f, "compound_id"].astype(str))
        eval_par = sorted(outer_ss.loc[outer_ss.outer_fold == f, "compound_id"].astype(str))
        add_scope(f"P5_OUTER{f}_FINAL_REFIT", f, "FINAL_REFIT", fit_par, eval_par)
        sub = inner_ss[inner_ss.outer_fold.astype(int) == f]
        for j in range(4):
            fit = sorted(sub.loc[sub.inner_fold.astype(int) != j, "compound_id"].astype(str))
            val = sorted(sub.loc[sub.inner_fold.astype(int) == j, "compound_id"].astype(str))
            add_scope(f"P5_OUTER{f}_INNER{j}", f, j, fit, val)
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    if len(df) != 25:
        raise RuntimeError(f"{ss}: expected 25 descriptor scaler scopes, got {len(df)}")
    write_json(done, {"status": "PASS", "rows": len(df), "sha256": sha256(outpath)})


def fit_model_pca_scope(Xctx, context_parent, fit_idx, eval_idx, eval_label, score_path, summary_row, scope_id):
    if score_path.exists():
        try:
            z = np.load(score_path)
            required = {"fit_context_row_index", "fit_scores", f"{eval_label}_context_row_index", f"{eval_label}_scores"}
            if required.issubset(z.files):
                return None
        except Exception:
            pass
    t0 = time.time()
    Xfit_full = np.asarray(Xctx[np.asarray(fit_idx, int), :], dtype=np.float32)
    var = np.var(Xfit_full, axis=0, dtype=np.float64)
    mask = var > 0.0
    retained = int(mask.sum())
    Xfit = np.ascontiguousarray(Xfit_full[:, mask], dtype=np.float32)
    del Xfit_full, var
    gc.collect()
    sc = StandardScaler(copy=False, with_mean=True, with_std=True)
    Xfit = sc.fit_transform(Xfit)
    ncomp = min(MAX_PCS, Xfit.shape[0] - 1, Xfit.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=pca_seed(scope_id), iterated_power=4, n_oversamples=10, copy=False)
    Zfit_all = pca.fit_transform(Xfit)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    hit = np.flatnonzero(cumulative >= TARGET_VARIANCE)
    k = int(hit[0] + 1) if len(hit) else int(ncomp)
    Zfit = np.asarray(Zfit_all[:, :k], dtype=np.float32)
    Xe = np.asarray(Xctx[np.asarray(eval_idx, int), :][:, mask], dtype=np.float32)
    Xe = sc.transform(Xe)
    Ze = np.asarray(pca.transform(Xe)[:, :k], dtype=np.float32)
    np.savez(
        score_path,
        fit_context_row_index=np.asarray(fit_idx, np.int32), fit_scores=Zfit,
        **{f"{eval_label}_context_row_index": np.asarray(eval_idx, np.int32), f"{eval_label}_scores": Ze},
    )
    summary_row.update({
        "retained_httr_features": retained,
        "zero_variance_httr_features": int((~mask).sum()),
        "pca_components_retained": k,
        "cumulative_variance_at_max_components": float(cumulative[-1]),
        "pca_seed": int(pca_seed(scope_id)),
        "seconds": round(time.time() - t0, 3),
    })
    del Xfit, Zfit_all, Zfit, Xe, Ze, sc, pca, mask
    gc.collect()
    return summary_row


def build_model_pca_scores(ss, wp, outer_ss, inner_ss, Xctx, fusion):
    done = wp["prep"] / "model_pca_scores.DONE.json"
    context_parent = fusion.sort_values("context_row_index").compound_id.astype(str).to_numpy()
    all_expected = [wp["p5b"] / f"P5B_OUTER{f}_FINAL_REFIT_scores.npz" for f in range(5)] + [wp["p5b"] / f"P5B_OUTER{f}_INNER{j}_scores.npz" for f in range(5) for j in range(4)]
    if done.exists() and all(p.exists() for p in all_expected):
        obj = json.load(open(done))
        if obj.get("status") == "PASS" and obj.get("score_file_n") == 25:
            log(f"{ss}: model HTTr PCA scores already PASS")
            return
    fmap = dict(zip(outer_ss.compound_id.astype(str), outer_ss.outer_fold.astype(int)))
    cf = np.asarray([fmap[p] for p in context_parent], int)
    summaries = []
    for f in range(5):
        fit_idx = np.where(cf != f)[0]
        te_idx = np.where(cf == f)[0]
        sid = f"P5B_OUTER{f}_FINAL_REFIT"
        r = {"split_set": ss, "scope_id": sid, "outer_fold": f, "inner_fold": "FINAL_REFIT", "fit_context_rows": len(fit_idx), "eval_context_rows": len(te_idx)}
        rr = fit_model_pca_scope(Xctx, context_parent, fit_idx, te_idx, "outer_test", wp["p5b"] / f"{sid}_scores.npz", r, sid)
        if rr is not None:
            summaries.append(rr); log(f"{ss}: fitted {sid}, PCs={rr['pca_components_retained']}")
        sub = inner_ss[inner_ss.outer_fold.astype(int) == f]
        imap = dict(zip(sub.compound_id.astype(str), sub.inner_fold.astype(int)))
        outer_train_mask = cf != f
        for j in range(4):
            fit_par = {p for p, jj in imap.items() if jj != j}
            val_par = {p for p, jj in imap.items() if jj == j}
            fi = np.where(np.array([(p in fit_par) for p in context_parent], bool))[0]
            vi = np.where(np.array([(p in val_par) for p in context_parent], bool))[0]
            sid = f"P5B_OUTER{f}_INNER{j}"
            r = {"split_set": ss, "scope_id": sid, "outer_fold": f, "inner_fold": j, "fit_context_rows": len(fi), "eval_context_rows": len(vi)}
            rr = fit_model_pca_scope(Xctx, context_parent, fi, vi, "inner_val", wp["p5b"] / f"{sid}_scores.npz", r, sid)
            if rr is not None:
                summaries.append(rr); log(f"{ss}: fitted {sid}, PCs={rr['pca_components_retained']}")
    # Reconstruct summary from saved arrays even after resume.
    reg = []
    for f in range(5):
        for suffix in ["FINAL_REFIT"] + [f"INNER{j}" for j in range(4)]:
            sid = f"P5B_OUTER{f}_{suffix}"
            p = wp["p5b"] / f"{sid}_scores.npz"
            z = np.load(p)
            reg.append({
                "split_set": ss, "scope_id": sid, "outer_fold": f, "inner_fold": suffix,
                "fit_context_rows": len(z["fit_context_row_index"]),
                "fit_pca_components": int(z["fit_scores"].shape[1]),
                "score_sha256": sha256(p),
            })
    pd.DataFrame(reg).to_csv(wp["prep"] / "P9D_model_PCA_score_registry.csv", index=False)
    write_json(done, {"status": "PASS", "score_file_n": 25, "registry_sha256": sha256(wp["prep"] / "P9D_model_PCA_score_registry.csv")})


def compute_chemical_support(ss, outer_ss, master, ids, MORGAN):
    rows = []
    rowidx = {p: i for i, p in enumerate(ids)}
    scaf_map = dict(zip(master.compound_id.astype(str), master.bemis_murcko_scaffold))
    for f in range(5):
        trp = sorted(outer_ss.loc[outer_ss.outer_fold.astype(int) != f, "compound_id"].astype(str))
        tep = sorted(outer_ss.loc[outer_ss.outer_fold.astype(int) == f, "compound_id"].astype(str))
        Xi = MORGAN[np.asarray([rowidx[p] for p in trp], int)].astype(np.float32, copy=False)
        Xq = MORGAN[np.asarray([rowidx[p] for p in tep], int)].astype(np.float32, copy=False)
        si = Xi.sum(axis=1)
        train_scaf = {str(scaf_map[p]) for p in trp if pd.notna(scaf_map[p]) and str(scaf_map[p]).strip()}
        for st in range(0, len(tep), 128):
            Q = Xq[st:st + 128]
            inter = Q @ Xi.T
            union = Q.sum(axis=1)[:, None] + si[None, :] - inter
            sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            mx = sim.max(axis=1)
            for p, v in zip(tep[st:st + len(mx)], mx):
                s = scaf_map[p]
                rep = bool(pd.notna(s) and str(s).strip() and str(s) in train_scaf)
                state = "C0" if (rep or float(v) >= 0.50) else "C1"
                rows.append({
                    "split_set": ss, "compound_id": p, "outer_fold": f,
                    "max_train_morgan_tanimoto": float(v),
                    "scaffold_represented_in_train": rep,
                    "chemical_state": state,
                    "chemical_rule": "C0=scaffold represented OR max Morgan>=0.50; C1=unseen scaffold AND max Morgan<0.50",
                })
    return pd.DataFrame(rows)


def compute_biological_support(ss, outer_ss, Xctx, fusion, jobs=8):
    fusion = fusion.sort_values("context_row_index").reset_index(drop=True)
    parents = fusion.compound_id.astype(str).to_numpy()
    contexts = fusion.cell_type.astype(str).to_numpy()
    fmap = dict(zip(outer_ss.compound_id.astype(str), outer_ss.outer_fold.astype(int)))
    folds = np.asarray([fmap[p] for p in parents], int)
    parent_rows = []
    threshold_rows = []
    train_dist_rows = []
    for f in range(5):
        t0 = time.time()
        tr = np.where(folds != f)[0]
        te = np.where(folds == f)[0]
        log(f"{ss}: Phase3-style biological support PCA outer{f} fit_context={len(tr)} test_context={len(te)}")
        sc = StandardScaler(copy=True)
        Xtr = sc.fit_transform(np.asarray(Xctx[tr, :], dtype=np.float32))
        Xte = sc.transform(np.asarray(Xctx[te, :], dtype=np.float32))
        pca = PCA(n_components=256, svd_solver="randomized", random_state=support_pca_seed(f), iterated_power=2)
        Ztr = pca.fit_transform(Xtr)
        Zte = pca.transform(Xte)
        nn = NearestNeighbors(n_neighbors=min(12, len(tr)), metric="euclidean", algorithm="brute", n_jobs=jobs).fit(Ztr)
        dtr, itr = nn.kneighbors(Ztr)
        trp = parents[tr]
        rd = []
        for i in range(len(tr)):
            ds = [float(d) for d, j in zip(dtr[i], itr[i]) if trp[j] != trp[i]][:5]
            if len(ds) < 5:
                raise RuntimeError(f"{ss} outer{f}: fewer than 5 non-self-parent HTTr neighbors")
            rd.append(float(np.mean(ds)))
        tpdist = pd.DataFrame({"parent": trp, "d": rd}).groupby("parent", sort=False).d.median()
        q75 = float(tpdist.quantile(0.75))
        q90 = float(tpdist.quantile(0.90))
        q95 = float(tpdist.quantile(0.95))
        q975 = float(tpdist.quantile(0.975))
        for p, d in tpdist.items():
            train_dist_rows.append({"split_set": ss, "outer_fold": f, "parent_compound_id": p, "train_bio_distance": float(d)})
        dte, _ = nn.kneighbors(Zte, n_neighbors=5)
        td = pd.DataFrame({"parent": parents[te], "d": dte.mean(axis=1), "context": contexts[te]})
        for pc, g in td.groupby("parent", sort=False):
            med = float(g.d.median())
            worst = float(g.d.max())
            parent_rows.append({
                "split_set": ss, "compound_id": pc, "outer_fold": f, "n_contexts": len(g),
                "bio_distance_median": med, "bio_distance_worst": worst,
                "train_q75": q75, "biological_state": "B0" if med <= q75 else "B1",
            })
        threshold_rows.append({
            "split_set": ss, "outer_fold": f, "train_context_rows": len(tr), "test_context_rows": len(te),
            "train_parents": int(len(tpdist)), "test_parents": int(pd.Series(parents[te]).nunique()),
            "pca_components": 256, "cum_var_256": float(pca.explained_variance_ratio_.sum()),
            "q75": q75, "q90": q90, "q95": q95, "q975": q975,
            "pca_seed": support_pca_seed(f), "seconds": round(time.time() - t0, 3),
        })
        del Xtr, Xte, Ztr, Zte, sc, pca, nn, dtr, itr, dte
        gc.collect()
    return pd.DataFrame(parent_rows), pd.DataFrame(threshold_rows), pd.DataFrame(train_dist_rows)


def build_support_and_preprocess(ss, files, v3, inner, cache):
    wp = workspace_paths(ss)
    outer_ss, inner_ss = create_workspace_manifests(ss, files, v3, inner)
    build_descriptor_scalers(ss, wp, outer_ss, inner_ss)
    fusion = pd.read_csv(files["fusion"]).sort_values("context_row_index").reset_index(drop=True)
    Xctx = np.load(cache, mmap_mode="r")
    build_model_pca_scores(ss, wp, outer_ss, inner_ss, Xctx, fusion)

    support_file = wp["prep"] / "P9D_primary_factorial_assignment.csv"
    thresh_file = wp["prep"] / "P9D_biological_support_thresholds.csv"
    traind_file = wp["prep"] / "P9D_training_bio_distances.csv"
    if support_file.exists() and thresh_file.exists() and traind_file.exists():
        log(f"{ss}: V3 training-derived support assignment already present")
    else:
        master = pd.read_csv(files["master"])
        sm = pd.read_csv(files["struct_manifest"])
        ids = sm.compound_id.astype(str).to_numpy()
        MORGAN = np.asarray(np.load(files["morgan"], allow_pickle=False)["morgan_bits"], dtype=np.float32)
        chem = compute_chemical_support(ss, outer_ss, master, ids, MORGAN)
        bio, th, td = compute_biological_support(ss, outer_ss, Xctx, fusion, jobs=ARGS.rf_jobs)
        sup = chem.merge(bio, on=["split_set", "compound_id", "outer_fold"], validate="one_to_one")
        sup["factorial_cell"] = sup.chemical_state.astype(str) + "/" + sup.biological_state.astype(str)
        if len(sup) != 2017 or sup.compound_id.nunique() != 2017 or set(sup.factorial_cell) - set(CELLS):
            raise RuntimeError(f"{ss}: invalid V3 support assignment")
        sup.to_csv(support_file, index=False)
        th.to_csv(thresh_file, index=False)
        td.to_csv(traind_file, index=False)
        log(f"{ss}: primary V3 factorial support counts {sup.factorial_cell.value_counts().to_dict()}")

    prep_status = {
        "split_set": ss, "status": "PASS", "support_rule": "PRIMARY_v0.2_CHEM_HYBRID050__BIO_Q75_MEDIAN",
        "descriptor_scaler_scopes": 25, "model_pca_score_scopes": 25,
        "factorial_assignment_sha256": sha256(support_file),
        "bio_thresholds_sha256": sha256(thresh_file),
        "phase6_engine_sha256": PHASE6_ENGINE_SHA,
    }
    write_json(wp["prep"] / "P9D_PREPROCESS_STATUS.json", prep_status)
    return wp

# -----------------------------------------------------------------------------
# Byte-identical Phase6 engine execution and split collection
# -----------------------------------------------------------------------------
def run_phase6_engine(ss, wp, files):
    status = wp["phase6"] / "P6_FINAL_STATUS.json"
    if status.exists():
        try:
            st = json.load(open(status))
            if st.get("status") == "PASS" and st.get("registry_sha256") == PHASE6_REGISTRY_SHA:
                log(f"{ss}: Phase6 retraining engine already PASS")
                return
        except Exception:
            pass
    cmd = [sys.executable, str(files["engine"]), "--root", str(wp["root"]), "--rf-jobs", str(ARGS.rf_jobs)]
    log(f"{ss}: launching byte-identical Phase6 engine")
    log("COMMAND: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"{ss}: Phase6 engine failed with code {r.returncode}. Rerun the same P9D command to resume checkpoints.")
    st = json.load(open(status))
    if st.get("status") != "PASS" or st.get("registry_sha256") != PHASE6_REGISTRY_SHA:
        raise RuntimeError(f"{ss}: Phase6 output did not retain the frozen computational registry")


def collect_split_outputs(ss, wp):
    d = SPLIT_OUT / ss
    d.mkdir(parents=True, exist_ok=True)
    p6 = wp["phase6"]
    pred = pd.read_csv(p6 / "P6_03_locked_parent_OOF_predictions.csv")
    pred.insert(0, "split_set", ss)
    pred.to_csv(d / f"P9D_{ss}_parent_OOF_predictions.csv", index=False)
    ctx = pd.read_csv(p6 / "P6_04_locked_context_OOF_raw_predictions.csv")
    ctx.insert(0, "split_set", ss)
    ctx.to_csv(d / f"P9D_{ss}_context_OOF_raw_predictions.csv", index=False)
    tun = pd.read_csv(p6 / "P6_05_tuning_calibration_threshold_registry.csv")
    tun.insert(0, "split_set", ss)
    tun.to_csv(d / f"P9D_{ss}_tuning_registry.csv", index=False)
    sup = pd.read_csv(wp["prep"] / "P9D_primary_factorial_assignment.csv")
    sup.to_csv(d / f"P9D_{ss}_factorial_assignment.csv", index=False)
    th = pd.read_csv(wp["prep"] / "P9D_biological_support_thresholds.csv")
    th.to_csv(d / f"P9D_{ss}_biological_support_thresholds.csv", index=False)
    status = {
        "split_set": ss, "status": "PASS",
        "phase6_registry_sha256": PHASE6_REGISTRY_SHA,
        "phase6_engine_sha256": PHASE6_ENGINE_SHA,
        "parent_prediction_rows": len(pred), "learned_parent_prediction_rows": int(pred.model_family.isin(FAMILIES).sum()),
        "context_prediction_rows": len(ctx), "tuning_rows": len(tun),
        "factorial_assignment_rows": len(sup),
    }
    write_json(d / f"P9D_{ss}_STATUS.json", status)
    return status


def split_complete(ss):
    d = SPLIT_OUT / ss
    p = d / f"P9D_{ss}_STATUS.json"
    if not p.exists():
        return False
    try:
        st = json.load(open(p))
        return st.get("status") == "PASS" and st.get("phase6_registry_sha256") == PHASE6_REGISTRY_SHA
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Robustness statistics after all four split predictions are locked
# -----------------------------------------------------------------------------
def endpoint_split_job(ss, ep, pred_file, support_file):
    use = ["endpoint", "compound_id", "y_true", "model_family", "modality", "probability", "predicted_class"]
    d = pd.read_csv(pred_file, usecols=use)
    d = d[(d.endpoint == ep) & d.model_family.isin(FAMILIES) & d.modality.isin(MODALITIES)].copy()
    sup = pd.read_csv(support_file, usecols=["compound_id", "factorial_cell"])
    meta = d[["compound_id", "y_true"]].drop_duplicates().sort_values("compound_id").reset_index(drop=True)
    meta = meta.merge(sup, on="compound_id", how="left", validate="one_to_one")
    if meta.factorial_cell.isna().any() or meta.compound_id.duplicated().any():
        raise RuntimeError(f"{ss}/{ep}: support alignment failed")
    ids = meta.compound_id.astype(str).to_numpy()
    y = meta.y_true.astype(int).to_numpy()
    cc = meta.factorial_cell.map(CI).astype(int).to_numpy()
    n = len(meta); J = len(MODEL_KEYS); K = len(METRICS)
    P = np.empty((n, J), np.float32); Q = np.empty((n, J), np.int8)
    for j, (fam, mod) in enumerate(MODEL_KEYS):
        g = d[(d.model_family == fam) & (d.modality == mod)][["compound_id", "probability", "predicted_class"]]
        g = g.set_index("compound_id").reindex(ids)
        if g.isna().any().any() or len(g) != n:
            raise RuntimeError(f"{ss}/{ep}/{fam}/{mod}: prediction alignment failure")
        P[:, j] = g.probability.to_numpy(np.float32)
        Q[:, j] = g.predicted_class.to_numpy(np.int8)

    rng = np.random.default_rng(seed_for(ss, ep))
    W = np.empty((BOOTSTRAP_N, n), np.uint16)
    for b in range(BOOTSTRAP_N):
        W[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    pool_obs = weighted_metrics(y, P, Q, np.ones((1, n), np.uint8))[0]
    pool_boot = weighted_metrics(y, P, Q, W)
    obs = np.full((J, 4, K), np.nan)
    boot = np.full((BOOTSTRAP_N, J, 4, K), np.nan, np.float32)
    cell_rows = []
    for c, cell in enumerate(CELLS):
        m = cc == c
        if np.any(m):
            obs[:, c, :] = weighted_metrics(y[m], P[m], Q[m], np.ones((1, int(m.sum())), np.uint8))[0]
            boot[:, :, c, :] = weighted_metrics(y[m], P[m], Q[m], W[:, m])
        active = int(y[m].sum()); inactive = int(m.sum() - active); prev = active / m.sum() if m.sum() else np.nan
        for j, (fam, mod) in enumerate(MODEL_KEYS):
            row = {"split_set": ss, "endpoint": ep, "model_family": fam, "modality": mod, "factorial_cell": cell,
                   "n": int(m.sum()), "active_n": active, "inactive_n": inactive, "prevalence": prev}
            for k, met in enumerate(METRICS):
                row[met] = obs[j, c, k]
            cell_rows.append(row)

    cons = []
    def add(af, met, contrast, fam, mod, est, bd, extra=None):
        r = {"split_set": ss, "endpoint": ep, "analysis_family": af, "metric": met, "contrast": contrast,
             "model_family": fam, "modality": mod, **boot_summary(est, bd)}
        if extra:
            r.update(extra)
        cons.append(r)

    for j, (fam, mod) in enumerate(MODEL_KEYS):
        for k, met in enumerate(METRICS):
            r, ch, bi, jo = [obs[j, CI[x], k] for x in CELLS]
            br, bch, bbi, bjo = [boot[:, j, CI[x], k] for x in CELLS]
            add("P9D_FACTORIAL", met, "joint_vs_reference", fam, mod, jo - r, bjo - br, {"reference_metric": r, "joint_metric": jo})
            add("P9D_FACTORIAL", met, "factorial_interaction", fam, mod, jo - ch - bi + r, bjo - bch - bbi + br)
            add("P9D_OPTIMISM", met, "pooled_OOF_minus_joint_OOD", fam, mod, pool_obs[j, k] - jo, pool_boot[:, j, k] - bjo, {"pooled_OOF_metric": pool_obs[j, k], "joint_metric": jo})
    midx = {k: i for i, k in enumerate(MODEL_KEYS)}
    for fam in FAMILIES:
        js = midx[(fam, "STRUCTURE")]
        for fm in ["EARLY_FUSION", "LATE_FUSION"]:
            jf = midx[(fam, fm)]
            for k, met in enumerate(METRICS):
                sref = obs[js, CI["C0/B0"], k]; sj = obs[js, CI["C1/B1"], k]
                fref = obs[jf, CI["C0/B0"], k]; fj = obs[jf, CI["C1/B1"], k]
                est = (fj - sj) - (fref - sref)
                bd = (boot[:, jf, CI["C1/B1"], k] - boot[:, js, CI["C1/B1"], k]) - (boot[:, jf, CI["C0/B0"], k] - boot[:, js, CI["C0/B0"], k])
                add("P9D_FUSION_ROBUSTNESS", met, "joint_minus_reference_fusion_advantage", fam, fm, est, bd,
                    {"reference_fusion_advantage": fref - sref, "joint_fusion_advantage": fj - sj})
    return cell_rows, cons


def random_effects(df):
    keys = ["split_set", "analysis_family", "metric", "contrast", "model_family", "modality"]
    z = df[df.bootstrap_supported & np.isfinite(df.estimate) & np.isfinite(df.bootstrap_se) & (df.bootstrap_se > 0)].copy()
    if not len(z):
        return pd.DataFrame()
    z["vi"] = z.bootstrap_se.to_numpy(float) ** 2
    z["wi"] = 1.0 / z.vi
    z["wiy"] = z.wi * z.estimate
    z["wiy2"] = z.wi * z.estimate * z.estimate
    z["wi2"] = z.wi * z.wi
    a = z.groupby(keys, sort=False, dropna=False).agg(endpoint_n=("estimate", "size"), sum_w=("wi", "sum"), sum_wy=("wiy", "sum"), sum_wy2=("wiy2", "sum"), sum_w2=("wi2", "sum")).reset_index()
    a = a[a.endpoint_n >= 3].copy()
    a["Q"] = a.sum_wy2 - (a.sum_wy * a.sum_wy / a.sum_w)
    a["den"] = a.sum_w - a.sum_w2 / a.sum_w
    a["tau2"] = np.where(a.den > 0, np.maximum(0, (a.Q - (a.endpoint_n - 1)) / a.den), 0.0)
    z = z.merge(a[keys + ["tau2"]], on=keys, how="inner", validate="many_to_one")
    z["wr"] = 1.0 / (z.vi + z.tau2); z["wry"] = z.wr * z.estimate
    b = z.groupby(keys, sort=False, dropna=False).agg(sum_wr=("wr", "sum"), sum_wry=("wry", "sum")).reset_index()
    out = a.merge(b, on=keys, validate="one_to_one")
    out["random_effect_mean"] = out.sum_wry / out.sum_wr
    out["random_effect_se"] = np.sqrt(1.0 / out.sum_wr)
    out["ci_low"] = out.random_effect_mean - 1.96 * out.random_effect_se
    out["ci_high"] = out.random_effect_mean + 1.96 * out.random_effect_se
    zz = out.random_effect_mean / out.random_effect_se
    out["p_value"] = 2 * norm.sf(np.abs(zz))
    out = out[keys + ["endpoint_n", "random_effect_mean", "random_effect_se", "ci_low", "ci_high", "tau2", "Q", "p_value"]]
    out["fdr_family"] = out.analysis_family.astype(str) + "|" + out.metric.astype(str) + "|" + out.contrast.astype(str)
    out["q_value"] = np.nan
    for (ss, fam), idx in out.groupby(["split_set", "fdr_family"]).groups.items():
        out.loc[idx, "q_value"] = bh(out.loc[idx, "p_value"].to_numpy(float))
    return out


def primary_reference_table(files):
    """Primary-split reference directions from the already-closed P9A primary definition."""
    d = pd.read_csv(files["p9a_re"])
    d = d[(d.chemical_definition == "CHEM_PRIMARY_HYBRID_050") & (d.biological_definition == "BIO_PRIMARY_Q75_MEDIAN")].copy()
    amap = {
        "P9A_FACTORIAL": "P9D_FACTORIAL",
        "P9A_OPTIMISM": "P9D_OPTIMISM",
        "P9A_FUSION_ROBUSTNESS": "P9D_FUSION_ROBUSTNESS",
    }
    d["analysis_family"] = d.analysis_family.map(amap)
    ref = d[["analysis_family", "metric", "contrast", "model_family", "modality", "random_effect_mean"]].rename(columns={"random_effect_mean": "primary_effect"})
    ref = ref.drop_duplicates(["analysis_family", "metric", "contrast", "model_family", "modality"])
    if len(ref) != 420:
        raise RuntimeError(f"Expected 420 P9A primary reference effects; found {len(ref)}")
    return ref


def gateD_split_stability(re, files):
    ref = primary_reference_table(files)
    keys = ["analysis_family", "metric", "contrast", "model_family", "modality"]
    rows = []
    for _, r0 in ref.iterrows():
        key = tuple(r0[k] for k in keys)
        g = re.copy()
        for k, v in zip(keys, key):
            g = g[g[k] == v]
        g = g[np.isfinite(g.random_effect_mean)].copy()
        ps = float(r0.primary_effect)
        sgn = int(np.sign(ps)) if np.isfinite(ps) and ps != 0 else 0
        if sgn == 0:
            match = pd.Series(False, index=g.index)
            reverse = pd.Series(False, index=g.index)
        else:
            gs = np.sign(g.random_effect_mean.to_numpy(float))
            match = pd.Series(gs == sgn, index=g.index)
            reverse = pd.Series(gs == -sgn, index=g.index)
        opp_ci = reverse & ((g.ci_low > 0) | (g.ci_high < 0))
        n = len(g); mn = int(match.sum()); rn = int(reverse.sum())
        if n < 4:
            verdict = "INSUFFICIENT_SPLITS"
        elif mn >= 3:
            verdict = "SPLIT_ROBUST"
        elif rn >= 2:
            verdict = "DOWNGRADE_DIRECTION_UNSTABLE"
        else:
            verdict = "MIXED"
        row = {k: v for k, v in zip(keys, key)}
        row.update({
            "primary_effect": ps, "primary_direction_sign": sgn, "evaluable_split_n": n,
            "direction_match_n": mn, "direction_reverse_n": rn,
            "opposite_ci_excludes_zero_n": int(opp_ci.sum()),
            "gateD_split_verdict": verdict,
        })
        for ss in SPLIT_SETS:
            x = g[g.split_set == ss]
            row[f"{ss}_effect"] = float(x.random_effect_mean.iloc[0]) if len(x) else np.nan
            row[f"{ss}_ci_low"] = float(x.ci_low.iloc[0]) if len(x) else np.nan
            row[f"{ss}_ci_high"] = float(x.ci_high.iloc[0]) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_all(files):
    if not all(split_complete(ss) for ss in SPLIT_SETS):
        missing = [ss for ss in SPLIT_SETS if not split_complete(ss)]
        raise RuntimeError(f"P9D analysis requires all four completed V3 splits. Missing: {missing}")
    tasks = []
    for ss in SPLIT_SETS:
        d = SPLIT_OUT / ss
        pred = d / f"P9D_{ss}_parent_OOF_predictions.csv"
        sup = d / f"P9D_{ss}_factorial_assignment.csv"
        for ep in PRIMARY_ENDPOINTS:
            tasks.append((ss, ep, pred, sup))
    log(f"P9D statistics: {BOOTSTRAP_N} paired parent bootstraps x 32 split-endpoint tasks using {ARGS.analysis_jobs} workers")
    res = Parallel(n_jobs=ARGS.analysis_jobs, backend="loky", verbose=10)(delayed(endpoint_split_job)(*x) for x in tasks)
    cells = []; cons = []
    for a, b in res:
        cells.extend(a); cons.extend(b)
    cells = pd.DataFrame(cells); cons = pd.DataFrame(cons)
    cons["fdr_family"] = cons.analysis_family.astype(str) + "|" + cons.metric.astype(str) + "|" + cons.contrast.astype(str)
    cons["q_value"] = np.nan
    for (ss, fam), idx in cons.groupby(["split_set", "fdr_family"]).groups.items():
        p = cons.loc[idx, "bootstrap_p_two_sided"].where(cons.loc[idx, "bootstrap_supported"], np.nan).to_numpy(float)
        cons.loc[idx, "q_value"] = bh(p)
    cons["ci_excludes_zero"] = (cons.ci_low > 0) | (cons.ci_high < 0)
    re = random_effects(cons)
    stab = gateD_split_stability(re, files)

    # Consolidated predictions/support/tuning before stats are interpreted.
    allpred = []; allctx = []; alltun = []; allsup = []; allth = []
    for ss in SPLIT_SETS:
        d = SPLIT_OUT / ss
        allpred.append(pd.read_csv(d / f"P9D_{ss}_parent_OOF_predictions.csv"))
        allctx.append(pd.read_csv(d / f"P9D_{ss}_context_OOF_raw_predictions.csv"))
        alltun.append(pd.read_csv(d / f"P9D_{ss}_tuning_registry.csv"))
        allsup.append(pd.read_csv(d / f"P9D_{ss}_factorial_assignment.csv"))
        allth.append(pd.read_csv(d / f"P9D_{ss}_biological_support_thresholds.csv"))
    ap = pd.concat(allpred, ignore_index=True); ac = pd.concat(allctx, ignore_index=True); at = pd.concat(alltun, ignore_index=True); ass = pd.concat(allsup, ignore_index=True); ath = pd.concat(allth, ignore_index=True)
    ap.to_csv(OUT / "P9D_01_all_split_parent_OOF_predictions.csv", index=False)
    ac.to_csv(OUT / "P9D_02_all_split_context_OOF_raw_predictions.csv", index=False)
    at.to_csv(OUT / "P9D_03_all_split_tuning_registry.csv", index=False)
    ass.to_csv(OUT / "P9D_04_all_split_factorial_assignments.csv", index=False)
    ath.to_csv(OUT / "P9D_05_all_split_biological_thresholds.csv", index=False)
    cells.to_csv(OUT / "P9D_06_cell_metrics.csv", index=False)
    cons.to_csv(OUT / "P9D_07_endpoint_contrasts.csv", index=False)
    re.to_csv(OUT / "P9D_08_split_random_effects.csv", index=False)
    stab.to_csv(OUT / "P9D_09_gateD_split_stability.csv", index=False)

    tests = []
    def T(tid, desc, ok, detail=""):
        tests.append({"test_id": tid, "description": desc, "status": "PASS" if bool(ok) else "FAIL", "detail": str(detail)})
    learned = ap[ap.model_family.isin(FAMILIES) & ap.modality.isin(MODALITIES)]
    T("P9D-T01", "All four V3 split sets consolidated", set(ap.split_set) == set(SPLIT_SETS), sorted(ap.split_set.unique()))
    T("P9D-T02", "Each V3 split has 150000 parent OOF rows", ap.groupby("split_set").size().eq(150000).all(), ap.groupby("split_set").size().to_dict())
    T("P9D-T03", "Each V3 split has 120000 learned parent OOF rows", learned.groupby("split_set").size().eq(120000).all(), learned.groupby("split_set").size().to_dict())
    T("P9D-T04", "No duplicate learned split/endpoint/model/modality/parent keys", not learned.duplicated(["split_set", "endpoint", "model_family", "modality", "compound_id"]).any(), int(learned.duplicated(["split_set", "endpoint", "model_family", "modality", "compound_id"]).sum()))
    T("P9D-T05", "All learned probabilities finite in [0,1]", np.isfinite(learned.probability.astype(float)).all() and learned.probability.astype(float).between(0, 1).all(), "finite/range")
    T("P9D-T06", "All V3 learned predictions use original Phase6 computational registry", (learned.registry_sha256 == PHASE6_REGISTRY_SHA).all(), learned.registry_sha256.value_counts().to_dict())
    T("P9D-T07", "Four V3 factorial assignments each contain 2017 unique parents", len(ass) == 8068 and ass.groupby("split_set").compound_id.nunique().eq(2017).all(), ass.groupby("split_set").compound_id.nunique().to_dict())
    T("P9D-T08", "Every V3 factorial assignment is one of four primary cells", set(ass.factorial_cell) == set(CELLS), ass.factorial_cell.value_counts().to_dict())
    T("P9D-T09", "Exactly 1536 split/endpoint/model/cell metric rows", len(cells) == 4 * 8 * 12 * 4, len(cells))
    T("P9D-T10", "Exactly 13440 endpoint contrast rows", len(cons) == 4 * 8 * 420, len(cons))
    T("P9D-T11", "Gate-D split stability table has 420 rows", len(stab) == 420, len(stab))
    T("P9D-T12", "All four split random-effect sets represented", set(re.split_set) == set(SPLIT_SETS), re.groupby("split_set").size().to_dict())
    T("P9D-T13", "Finite endpoint q-values are within [0,1]", cons.q_value.dropna().between(0, 1).all(), "range")
    T("P9D-T14", "No V3 split was selected or omitted", stab.evaluable_split_n.min() >= 4, int(stab.evaluable_split_n.min()))
    T("P9D-T15", "All split tuning rows retain 3 families x 4 modalities x 8 endpoints x 5 folds", at.groupby("split_set").size().eq(480).all(), at.groupby("split_set").size().to_dict())
    T("P9D-T16", "All split context prediction tables are non-empty", ac.groupby("split_set").size().gt(0).all(), ac.groupby("split_set").size().to_dict())
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "P9D_10_integrity_tests.csv", index=False)
    if (t.status != "PASS").any():
        print(t[t.status != "PASS"].to_string(index=False))
        raise RuntimeError("P9D consolidated integrity failure")

    status = {
        "phase": "9D", "status": "PASS", "phase9_lock_sha256": PHASE9_LOCK_SHA,
        "phase6_engine_sha256": PHASE6_ENGINE_SHA, "phase6_registry_sha256": PHASE6_REGISTRY_SHA,
        "v3_split_sets": SPLIT_SETS, "bootstrap_resamples_per_endpoint_split": BOOTSTRAP_N,
        "parent_oof_rows": len(ap), "learned_parent_oof_rows": len(learned),
        "context_oof_rows": len(ac), "cell_metric_rows": len(cells), "contrast_rows": len(cons),
        "split_random_effect_rows": len(re), "gateD_stability_rows": len(stab),
        "integrity": f"{len(t)}/{len(t)} PASS",
        "governance": "All four prospectively frozen V3 split sets were run. No favorable split selection; primary Phase6/7/7B/8 results unchanged.",
    }
    write_json(OUT / "P9D_FINAL_STATUS.json", status)

    package_files = [
        "P9D_00_execution_registry.json", "P9D_PREFLIGHT_TESTS.csv", "P9D_PREFLIGHT_STATUS.json",
        "P9D_01_all_split_parent_OOF_predictions.csv",
        "P9D_02_all_split_context_OOF_raw_predictions.csv",
        "P9D_03_all_split_tuning_registry.csv",
        "P9D_04_all_split_factorial_assignments.csv",
        "P9D_05_all_split_biological_thresholds.csv",
        "P9D_06_cell_metrics.csv", "P9D_07_endpoint_contrasts.csv",
        "P9D_08_split_random_effects.csv", "P9D_09_gateD_split_stability.csv",
        "P9D_10_integrity_tests.csv", "P9D_FINAL_STATUS.json",
    ]
    mf = OUT / "SHA256SUMS_P9D.txt"
    with open(mf, "w") as f:
        for n in package_files:
            f.write(f"{sha256(OUT / n)}  {n}\n")
        # Split-specific metadata files are included for auditable preprocessing.
        for ss in SPLIT_SETS:
            d = SPLIT_OUT / ss
            for p in sorted(d.glob("*.json")):
                f.write(f"{sha256(p)}  split_outputs/{ss}/{p.name}\n")
    zpath = OUT / "P9D_UPLOAD_TO_CHAT.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for n in package_files + ["SHA256SUMS_P9D.txt"]:
            zf.write(OUT / n, arcname=n)
        for ss in SPLIT_SETS:
            d = SPLIT_OUT / ss
            for p in sorted(d.glob("*.json")):
                zf.write(p, arcname=f"split_outputs/{ss}/{p.name}")
    status["upload_zip"] = str(zpath)
    status["upload_zip_sha256"] = sha256(zpath)
    status["upload_zip_size_bytes"] = zpath.stat().st_size
    write_json(OUT / "P9D_FINAL_STATUS.json", status)
    return status

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 92)
    print("FACTORIAL-OOD PHASE 9D — FOUR FROZEN V3 SPLIT-SET ROBUSTNESS RETRAINING")
    print("=" * 92)
    files = locate_inputs()
    pf = preflight()
    passn = int((pf.status == "PASS").sum())
    print(pf.to_string(index=False))
    print("-" * 92)
    print(f"P9D PREFLIGHT: {passn}/{len(pf)} PASS")
    print("NO V3 PERFORMANCE WAS COMPUTED BY PREFLIGHT.")
    if (pf.status != "PASS").any():
        raise RuntimeError("P9D preflight failed")
    if ARGS.preflight_only:
        return

    v3 = pd.read_csv(files["v3"])
    inner = pd.read_csv(files["inner"])
    cache = ensure_httr_cache(files)
    selected = SPLIT_SETS if ARGS.split_set == "ALL" else [ARGS.split_set]

    if not ARGS.analyze_only:
        for ss in selected:
            print("\n" + "=" * 92)
            print(f"P9D SPLIT START: {ss}")
            print("=" * 92)
            wp = build_support_and_preprocess(ss, files, v3, inner, cache)
            if ARGS.preprocess_only:
                print(f"{ss}: PREPROCESS PASS (model retraining not requested)")
                continue
            run_phase6_engine(ss, wp, files)
            st = collect_split_outputs(ss, wp)
            print(f"{ss}: RETRAINING PASS | parent rows={st['parent_prediction_rows']} | learned rows={st['learned_parent_prediction_rows']}")

    if ARGS.preprocess_only or ARGS.train_only:
        return

    if all(split_complete(ss) for ss in SPLIT_SETS):
        status = analyze_all(files)
        print("\n" + "=" * 92)
        print("PHASE 9D LOCAL COMPUTATION PASS")
        print("V3 split sets: 4/4")
        print("Parent OOF rows:", status["parent_oof_rows"])
        print("Learned parent OOF rows:", status["learned_parent_oof_rows"])
        print("Cell metric rows:", status["cell_metric_rows"])
        print("Endpoint contrast rows:", status["contrast_rows"])
        print("Gate-D split-stability rows:", status["gateD_stability_rows"])
        print("Integrity:", status["integrity"])
        print("Upload:", status["upload_zip"])
        print("Upload ZIP SHA256:", status["upload_zip_sha256"])
        print("All four frozen V3 split sets were retained; Phase6/7/7B/8 primary conclusions were not modified.")
        print("=" * 92)
    else:
        missing = [ss for ss in SPLIT_SETS if not split_complete(ss)]
        print("\nPartial P9D execution completed. Remaining frozen V3 split sets:", missing)
        print("Rerun the same command; completed preprocessing/model checkpoints will be skipped.")


if __name__ == "__main__":
    main()
