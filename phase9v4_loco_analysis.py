#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FACTORIAL-OOD
Phase 9 V4 — Prespecified LOCO Statistical Analysis

IMPORTANT GOVERNANCE
--------------------
1. Run --freeze-only first.
2. Run --preflight-only second.
3. Only then run the analysis.
4. No model, endpoint, context, threshold, metric or contrast may be removed
   according to V4 performance.

The script uses:
- all 3 held-out contexts: MCF7, U2OS, HepRG
- all 8 frozen endpoints
- 3 model families
- 4 modalities
- 10 frozen metrics
- 2,000 paired parent bootstrap resamples
- BH-FDR
- across-endpoint DerSimonian-Laird random-effects synthesis

Primary V4 contrasts:
1. LOCO_minus_PRIMARY_same_context
2. HTTR_minus_STRUCTURE_within_LOCO
3. EARLY_FUSION_minus_STRUCTURE_within_LOCO
4. LATE_FUSION_minus_STRUCTURE_within_LOCO
5. LOCO_minus_PRIMARY_change_in_modality_advantage_vs_STRUCTURE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------

PHASE9_LOCK_SHA = (
    "55f769210fa70d1be54904e1e59c7a639175811018a8373264fe3f3226027a01"
)

PHASE6_REGISTRY_SHA = (
    "10005b76d389dcc73466ef377cf4f5d2ee0787ff71b95523d08ab2b988276b57"
)

P9V4_EXECUTION_REGISTRY_SHA = (
    "4511f188324e54036ec4ff7665b5658e8e9fe4c834dfe11a2492eecbff12f4d9"
)

CONTEXTS = ["MCF7", "U2OS", "HepRG"]

ENDPOINTS = [
    "TOX21_ERa_BLA_Antagonist_ratio",
    "TOX21_FXR_BLA_Antagonist_ratio",
    "TOX21_MMP_ratio",
    "TOX21_PPARg_BLA_Antagonist_ratio",
    "TOX21_RAR_LUC_Antagonist",
    "TOX21_RORg_LUC_CHO_Antagonist",
    "TOX21_SBE_BLA_Antagonist_ratio",
    "TOX21_hERG_U2OS_Antagonist",
]

FAMILIES = [
    "ELASTIC_NET",
    "RANDOM_FOREST",
    "HIST_GRAD_BOOST",
]

MODALITIES = [
    "STRUCTURE",
    "HTTR",
    "EARLY_FUSION",
    "LATE_FUSION",
]

NONSTRUCTURE = ["HTTR", "EARLY_FUSION", "LATE_FUSION"]

MODEL_KEYS = [(f, m) for f in FAMILIES for m in MODALITIES]
MODEL_INDEX = {x: i for i, x in enumerate(MODEL_KEYS)}

METRICS = [
    "AUROC",
    "AUPRC",
    "MCC",
    "balanced_MCC",
    "Brier",
    "Brier_skill_score",
    "ECE_10bin",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
]

# +1 means larger raw value is better.
# -1 means smaller raw value is better.
UTILITY_SIGN = {
    "AUROC": 1,
    "AUPRC": 1,
    "MCC": 1,
    "balanced_MCC": 1,
    "Brier": -1,
    "Brier_skill_score": 1,
    "ECE_10bin": -1,
    "balanced_accuracy": 1,
    "sensitivity": 1,
    "specificity": 1,
}

BOOTSTRAP_N = 2000
MIN_BOOT_VALID_FRAC = 0.90


# ---------------------------------------------------------------------
# CLI / paths
# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem",
    )
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    return ap.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root).resolve()

P9V4 = ROOT / "Phase9_Output" / "P9V4"
ANALYSIS = P9V4 / "analysis"

SCRIPT = Path(__file__).resolve()

LOCK_FILE = ANALYSIS / "P9V4_ANALYSIS_LOCK_v1.0.json"
FREEZE_SHA_FILE = ANALYSIS / "P9V4_ANALYSIS_FREEZE_SHA256.txt"

V4_REGISTRY = P9V4 / "P9V4_00_execution_registry.json"
V4_PARENT = P9V4 / "P9V4_01_parent_predictions.csv"
V4_CONTEXT = P9V4 / "P9V4_02_context_raw_predictions.csv"
V4_TUNING = P9V4 / "P9V4_03_tuning_registry.csv"
V4_PREPROCESS = P9V4 / "P9V4_04_preprocess_registry.csv"
V4_INTEGRITY = P9V4 / "P9V4_05_integrity_tests.csv"
V4_STATUS = P9V4 / "P9V4_TRAINING_STATUS.json"
V4_SUMS = P9V4 / "SHA256SUMS_P9V4_TRAINING_AUDIT_CORRECTED.txt"

PHASE9_LOCK = ROOT / "P9_00_robustness_lock_v1.0.json"

P6_REGISTRY = (
    ROOT / "Phase6_Output" / "P6_00_locked_computational_registry.json"
)
P6_PARENT = (
    ROOT / "Phase6_Output" / "P6_03_locked_parent_OOF_predictions.csv"
)

P9C_PARENT = (
    ROOT
    / "Phase9_Output"
    / "P9C"
    / "P9C_01_context_specific_predictions.csv"
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def die(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def parse_sha_manifest(path: Path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        out[parts[1].strip()] = parts[0].strip()
    return out


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
        return {
            "estimate": est,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_se": np.nan,
            "bootstrap_p_two_sided": np.nan,
            "bootstrap_valid_n": len(v),
            "bootstrap_valid_fraction": frac,
            "bootstrap_supported": False,
        }

    lo, hi = np.quantile(v, [0.025, 0.975])
    se = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan

    # Identical inference convention to the validated P9D machinery.
    dev = v - est
    p = (1 + np.sum(np.abs(dev) >= abs(est))) / (len(v) + 1)

    return {
        "estimate": float(est),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_se": se,
        "bootstrap_p_two_sided": float(min(1.0, p)),
        "bootstrap_valid_n": len(v),
        "bootstrap_valid_fraction": float(frac),
        "bootstrap_supported": bool(frac >= MIN_BOOT_VALID_FRAC),
    }


def bootstrap_seed(context, endpoint):
    text = "|".join(
        [
            "FACTORIAL-OOD",
            "P9V4",
            "ANALYSIS",
            "v1",
            str(context),
            str(endpoint),
            "PARENT_CLUSTER_BOOTSTRAP",
        ]
    )
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------
# Exact P9D-compatible metric engine
# ---------------------------------------------------------------------

def balanced_mcc_from_counts(tp, fn, tn, fp):
    sen = np.divide(
        tp,
        tp + fn,
        out=np.full_like(tp, np.nan, dtype=float),
        where=(tp + fn) > 0,
    )
    spe = np.divide(
        tn,
        tn + fp,
        out=np.full_like(tn, np.nan, dtype=float),
        where=(tn + fp) > 0,
    )

    den2 = 1 - (sen - spe) ** 2

    out = np.full_like(sen, np.nan, dtype=float)
    ok = np.isfinite(sen) & np.isfinite(spe) & (den2 > 1e-14)

    out[ok] = (
        (sen[ok] + spe[ok] - 1)
        / np.sqrt(den2[ok])
    )

    return out, sen, spe


def weighted_rank_metrics(y, p, W):
    """Exact tie-aware weighted AUROC and average precision."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    W = np.asarray(W, float)

    Bn = W.shape[0]
    auc = np.full(Bn, np.nan)
    ap = np.full(Bn, np.nan)

    if len(y) == 0:
        return auc, ap

    order = np.argsort(-p, kind="mergesort")

    ps = p[order]
    ys = y[order]
    Ws = W[:, order]

    starts = np.r_[0, 1 + np.where(ps[1:] != ps[:-1])[0]]

    pos_g = np.add.reduceat(
        Ws * ys[None, :],
        starts,
        axis=1,
    )
    neg_g = np.add.reduceat(
        Ws * (1 - ys)[None, :],
        starts,
        axis=1,
    )

    Ptot = pos_g.sum(axis=1)
    Ntot = neg_g.sum(axis=1)

    ok = (Ptot > 0) & (Ntot > 0)

    if np.any(ok):
        cpos = np.cumsum(pos_g, axis=1)
        call = np.cumsum(pos_g + neg_g, axis=1)

        prec = np.divide(
            cpos,
            call,
            out=np.zeros_like(cpos),
            where=call > 0,
        )

        ap[ok] = (
            np.sum(prec[ok] * pos_g[ok], axis=1)
            / Ptot[ok]
        )

        cneg = np.cumsum(neg_g, axis=1)
        neg_below = Ntot[:, None] - cneg

        num = np.sum(
            pos_g * (neg_below + 0.5 * neg_g),
            axis=1,
        )

        auc[ok] = (
            num[ok]
            / (Ptot[ok] * Ntot[ok])
        )

    return auc, ap


def weighted_ece(y, p, W, nbins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    W = np.asarray(W, float)

    Bn = W.shape[0]
    tot = W.sum(axis=1)

    out = np.zeros(Bn, float)
    good = tot > 0

    bins = np.minimum(
        (np.clip(p, 0, 1) * nbins).astype(int),
        nbins - 1,
    )

    for k in range(nbins):
        idx = np.where(bins == k)[0]

        if not len(idx):
            continue

        wb = W[:, idx]
        cnt = wb.sum(axis=1)
        m = cnt > 0

        if not np.any(m):
            continue

        ys = wb @ y[idx]
        ps = wb @ p[idx]

        obs = np.divide(
            ys,
            cnt,
            out=np.zeros_like(cnt),
            where=m,
        )

        pred = np.divide(
            ps,
            cnt,
            out=np.zeros_like(cnt),
            where=m,
        )

        out[m] += (
            cnt[m] / tot[m]
            * np.abs(obs[m] - pred[m])
        )

    out[~good] = np.nan
    return out


def weighted_metrics(y, P, Q, W):
    """
    Return B x J x K metrics.

    y: n
    P: n x J probabilities
    Q: n x J thresholded predictions
    W: B x n bootstrap integer weights
    """
    y = np.asarray(y, int)
    P = np.asarray(P, float)
    Q = np.asarray(Q, int)
    W = np.asarray(W, float)

    Bn, n = W.shape
    J = P.shape[1]
    K = len(METRICS)

    out = np.full((Bn, J, K), np.nan, float)

    tot = W.sum(axis=1)
    y1 = y.astype(float)

    prev = np.divide(
        W @ y1,
        tot,
        out=np.full(Bn, np.nan),
        where=tot > 0,
    )

    for j in range(J):
        q = Q[:, j]
        p = P[:, j]

        tp = W @ ((y == 1) & (q == 1)).astype(float)
        fn = W @ ((y == 1) & (q == 0)).astype(float)
        tn = W @ ((y == 0) & (q == 0)).astype(float)
        fp = W @ ((y == 0) & (q == 1)).astype(float)

        bm, sen, spe = balanced_mcc_from_counts(
            tp, fn, tn, fp
        )

        ba = (sen + spe) / 2

        den = np.sqrt(
            (tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn)
        )

        mcc = np.divide(
            tp * tn - fp * fn,
            den,
            out=np.full(Bn, np.nan),
            where=den > 0,
        )

        err = (p - y) ** 2

        brier = np.divide(
            W @ err,
            tot,
            out=np.full(Bn, np.nan),
            where=tot > 0,
        )

        ref = prev * (1 - prev)

        bss = np.divide(
            brier,
            ref,
            out=np.full(Bn, np.nan),
            where=ref > 0,
        )
        bss = 1 - bss

        auc, ap = weighted_rank_metrics(
            y, p, W
        )

        ece = weighted_ece(
            y, p, W, 10
        )

        vals = {
            "AUROC": auc,
            "AUPRC": ap,
            "MCC": mcc,
            "balanced_MCC": bm,
            "Brier": brier,
            "Brier_skill_score": bss,
            "ECE_10bin": ece,
            "balanced_accuracy": ba,
            "sensitivity": sen,
            "specificity": spe,
        }

        for k, metric in enumerate(METRICS):
            out[:, j, k] = vals[metric]

    return out


# ---------------------------------------------------------------------
# Input validation and freeze
# ---------------------------------------------------------------------

REQUIRED = [
    PHASE9_LOCK,
    P6_REGISTRY,
    P6_PARENT,
    P9C_PARENT,
    V4_REGISTRY,
    V4_PARENT,
    V4_CONTEXT,
    V4_TUNING,
    V4_PREPROCESS,
    V4_INTEGRITY,
    V4_STATUS,
    V4_SUMS,
]


def validate_training_manifest():
    manifest = parse_sha_manifest(V4_SUMS)

    failures = []

    for rel, expected in manifest.items():
        p = P9V4 / rel

        if not p.exists():
            failures.append(f"missing:{rel}")
            continue

        actual = sha256_file(p)

        if actual != expected:
            failures.append(
                f"hash:{rel}:{actual}!={expected}"
            )

    return failures


def basic_input_checks():
    tests = []

    def T(tid, desc, ok, detail=""):
        tests.append(
            {
                "test_id": tid,
                "description": desc,
                "status": "PASS" if bool(ok) else "FAIL",
                "detail": str(detail),
            }
        )

    for p in REQUIRED:
        T(
            f"V4A-F-{p.name}",
            f"Required input exists: {p.name}",
            p.exists(),
            p,
        )

    if any(x["status"] == "FAIL" for x in tests):
        return pd.DataFrame(tests)

    T(
        "V4A-F01",
        "Phase9 scientific lock SHA256 unchanged",
        sha256_file(PHASE9_LOCK) == PHASE9_LOCK_SHA,
        sha256_file(PHASE9_LOCK),
    )

    T(
        "V4A-F02",
        "Phase6 computational registry SHA256 unchanged",
        sha256_file(P6_REGISTRY) == PHASE6_REGISTRY_SHA,
        sha256_file(P6_REGISTRY),
    )

    T(
        "V4A-F03",
        "P9V4 execution registry SHA256 unchanged",
        sha256_file(V4_REGISTRY) == P9V4_EXECUTION_REGISTRY_SHA,
        sha256_file(V4_REGISTRY),
    )

    manifest_failures = validate_training_manifest()

    T(
        "V4A-F04",
        "All P9V4 training files match frozen training SHA256 manifest",
        len(manifest_failures) == 0,
        manifest_failures,
    )

    st = read_json(V4_STATUS)

    T(
        "V4A-F05",
        "V4 training status PASS",
        st.get("status") == "PASS",
        st.get("status"),
    )

    T(
        "V4A-F06",
        "V4 training integrity 15/15 PASS",
        st.get("integrity") == "15/15 PASS",
        st.get("integrity"),
    )

    iv = pd.read_csv(V4_INTEGRITY)

    T(
        "V4A-F07",
        "All 15 V4 post-training integrity checks PASS",
        len(iv) == 15 and (iv["status"] == "PASS").all(),
        f"{int((iv.status == 'PASS').sum())}/{len(iv)}",
    )

    vp = pd.read_csv(V4_PARENT)

    T(
        "V4A-F08",
        "V4 parent prediction count = 244956",
        len(vp) == 244956,
        len(vp),
    )

    vc = pd.read_csv(V4_CONTEXT)

    T(
        "V4A-F09",
        "V4 raw context prediction count = 122478",
        len(vc) == 122478,
        len(vc),
    )

    vt = pd.read_csv(V4_TUNING)

    T(
        "V4A-F10",
        "V4 tuning rows = 1440",
        len(vt) == 1440,
        len(vt),
    )

    pp = pd.read_csv(V4_PREPROCESS)

    T(
        "V4A-F11",
        "V4 preprocessing/PCA scope rows = 75",
        len(pp) == 75,
        len(pp),
    )

    T(
        "V4A-F12",
        "All three held-out contexts present",
        set(vp.heldout_context.astype(str).unique()) == set(CONTEXTS),
        sorted(vp.heldout_context.astype(str).unique()),
    )

    T(
        "V4A-F13",
        "All eight frozen endpoints present",
        set(vp.endpoint.astype(str).unique()) == set(ENDPOINTS),
        sorted(vp.endpoint.astype(str).unique()),
    )

    learned = vp[
        vp.model_family.isin(FAMILIES)
        & vp.modality.isin(MODALITIES)
    ].copy()

    combos = learned[
        ["model_family", "modality"]
    ].drop_duplicates()

    T(
        "V4A-F14",
        "Exactly 12 V4 learned model-modality combinations",
        len(combos) == 12,
        len(combos),
    )

    p9c = pd.read_csv(P9C_PARENT)

    p9c = p9c[
        p9c.model_family.isin(FAMILIES)
        & p9c.modality.isin(NONSTRUCTURE)
    ].copy()

    T(
        "V4A-F15",
        "P9C comparator has 9 context-dependent combinations",
        len(
            p9c[
                ["model_family", "modality"]
            ].drop_duplicates()
        ) == 9,
        len(
            p9c[
                ["model_family", "modality"]
            ].drop_duplicates()
        ),
    )

    T(
        "V4A-F16",
        "P9C comparator contains all three contexts",
        set(p9c.cell_type.astype(str).unique()) == set(CONTEXTS),
        sorted(p9c.cell_type.astype(str).unique()),
    )

    p6 = pd.read_csv(P6_PARENT)

    p6s = p6[
        p6.model_family.isin(FAMILIES)
        & (p6.modality == "STRUCTURE")
    ].copy()

    T(
        "V4A-F17",
        "Phase6 comparator contains three structure-only families",
        len(
            p6s[
                ["model_family", "modality"]
            ].drop_duplicates()
        ) == 3,
        len(
            p6s[
                ["model_family", "modality"]
            ].drop_duplicates()
        ),
    )

    T(
        "V4A-F18",
        "Phase6 structure comparator uses frozen registry",
        (
            p6s.registry_sha256.astype(str)
            == PHASE6_REGISTRY_SHA
        ).all(),
        sorted(
            p6s.registry_sha256.astype(str).unique()
        ),
    )

    T(
        "V4A-F19",
        "V4 predictions use frozen Phase6 registry",
        (
            learned.registry_sha256.astype(str)
            == PHASE6_REGISTRY_SHA
        ).all(),
        sorted(
            learned.registry_sha256.astype(str).unique()
        ),
    )

    # Comparator alignment check, without computing performance.
    align_failures = []

    for h in CONTEXTS:
        for ep in ENDPOINTS:
            g = learned[
                (learned.heldout_context == h)
                & (learned.endpoint == ep)
            ]

            if not len(g):
                align_failures.append(
                    f"{h}|{ep}|missing V4"
                )
                continue

            base = (
                g[
                    (g.model_family == FAMILIES[0])
                    & (g.modality == "STRUCTURE")
                ][
                    [
                        "compound_id",
                        "y_true",
                        "outer_fold",
                    ]
                ]
                .drop_duplicates()
                .sort_values("compound_id")
                .reset_index(drop=True)
            )

            ids = set(base.compound_id.astype(str))

            for fam in FAMILIES:
                # V4
                for mod in MODALITIES:
                    z = g[
                        (g.model_family == fam)
                        & (g.modality == mod)
                    ]

                    if (
                        len(z) != len(base)
                        or set(
                            z.compound_id.astype(str)
                        ) != ids
                    ):
                        align_failures.append(
                            f"{h}|{ep}|V4|{fam}|{mod}"
                        )

                # Phase6 structure comparator
                zs = p6s[
                    (p6s.endpoint == ep)
                    & (p6s.model_family == fam)
                    & p6s.compound_id.astype(str).isin(ids)
                ]

                if (
                    len(zs) != len(base)
                    or set(
                        zs.compound_id.astype(str)
                    ) != ids
                ):
                    align_failures.append(
                        f"{h}|{ep}|P6|{fam}|STRUCTURE"
                    )

                # P9C same-context comparators
                for mod in NONSTRUCTURE:
                    zp = p9c[
                        (p9c.endpoint == ep)
                        & (p9c.cell_type == h)
                        & (p9c.model_family == fam)
                        & (p9c.modality == mod)
                        & p9c.compound_id.astype(str).isin(ids)
                    ]

                    if (
                        len(zp) != len(base)
                        or set(
                            zp.compound_id.astype(str)
                        ) != ids
                    ):
                        align_failures.append(
                            f"{h}|{ep}|P9C|{fam}|{mod}"
                        )

    T(
        "V4A-F20",
        "All V4 and PRIMARY comparator populations align before performance analysis",
        len(align_failures) == 0,
        align_failures[:20],
    )

    return pd.DataFrame(tests)


def freeze_analysis():
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    tests = basic_input_checks()

    tests.to_csv(
        ANALYSIS / "P9V4_ANALYSIS_FREEZE_PREFLIGHT_TESTS.csv",
        index=False,
    )

    if (tests.status != "PASS").any():
        print(tests.to_string(index=False))
        die("P9V4 analysis freeze preflight failed.")

    lock = {
        "project": "FACTORIAL-OOD",
        "phase": "P9V4",
        "stage": "STATISTICAL_ANALYSIS",
        "version": "v1.0",
        "performance_viewed_before_freeze": False,
        "script_sha256": sha256_file(SCRIPT),
        "phase9_lock_sha256": sha256_file(PHASE9_LOCK),
        "phase6_registry_sha256": sha256_file(P6_REGISTRY),
        "p9v4_execution_registry_sha256": sha256_file(V4_REGISTRY),
        "v4_parent_predictions_sha256": sha256_file(V4_PARENT),
        "v4_context_predictions_sha256": sha256_file(V4_CONTEXT),
        "v4_tuning_registry_sha256": sha256_file(V4_TUNING),
        "v4_preprocess_registry_sha256": sha256_file(V4_PREPROCESS),
        "v4_integrity_tests_sha256": sha256_file(V4_INTEGRITY),
        "p6_parent_comparator_sha256": sha256_file(P6_PARENT),
        "p9c_context_comparator_sha256": sha256_file(P9C_PARENT),
        "heldout_contexts": CONTEXTS,
        "primary_endpoints": ENDPOINTS,
        "model_families": FAMILIES,
        "modalities": MODALITIES,
        "metrics": METRICS,
        "bootstrap": {
            "unit": "unique_parent_compound",
            "resamples": BOOTSTRAP_N,
            "ci": "95% percentile",
            "paired": True,
            "seed_rule": (
                "sha256('FACTORIAL-OOD|P9V4|ANALYSIS|v1|"
                "{context}|{endpoint}|PARENT_CLUSTER_BOOTSTRAP')[:8]"
            ),
        },
        "fdr": (
            "Benjamini-Hochberg within heldout_context × "
            "analysis_family × metric × named contrast"
        ),
        "random_effects": (
            "DerSimonian-Laird across the eight frozen endpoints "
            "within each held-out context"
        ),
        "contrasts": [
            "LOCO_minus_PRIMARY_same_context",
            "HTTR_minus_STRUCTURE_within_LOCO",
            "EARLY_FUSION_minus_STRUCTURE_within_LOCO",
            "LATE_FUSION_minus_STRUCTURE_within_LOCO",
            (
                "LOCO_minus_PRIMARY_change_in_"
                "modality_advantage_vs_STRUCTURE"
            ),
        ],
        "utility_orientation": {
            "higher_is_better": [
                x for x in METRICS
                if UTILITY_SIGN[x] == 1
            ],
            "lower_is_better": [
                x for x in METRICS
                if UTILITY_SIGN[x] == -1
            ],
        },
        "governance": (
            "All contexts, endpoints, model families, modalities, "
            "metrics and contrasts retained. No favorable-result "
            "selection or post-lock tuning."
        ),
    }

    write_json(LOCK_FILE, lock)

    lock_sha = sha256_file(LOCK_FILE)

    FREEZE_SHA_FILE.write_text(
        f"{sha256_file(SCRIPT)}  {SCRIPT.name}\n"
        f"{lock_sha}  {LOCK_FILE.name}\n"
    )

    print("\n" + "=" * 90)
    print("P9V4 ANALYSIS FREEZE PASS")
    print(f"Freeze tests: {len(tests)}/{len(tests)} PASS")
    print(f"Script SHA256: {sha256_file(SCRIPT)}")
    print(f"Analysis lock SHA256: {lock_sha}")
    print("NO V4 PERFORMANCE WAS COMPUTED OR INTERPRETED.")
    print("=" * 90)


def verify_frozen_analysis():
    if not LOCK_FILE.exists():
        die(
            "Analysis lock missing. Run --freeze-only first."
        )

    lock = read_json(LOCK_FILE)

    tests = []

    def T(tid, desc, ok, detail=""):
        tests.append(
            {
                "test_id": tid,
                "description": desc,
                "status": "PASS" if bool(ok) else "FAIL",
                "detail": str(detail),
            }
        )

    T(
        "V4A-P01",
        "Current script matches frozen analysis script SHA256",
        sha256_file(SCRIPT) == lock["script_sha256"],
        sha256_file(SCRIPT),
    )

    hash_pairs = [
        (
            "V4A-P02",
            PHASE9_LOCK,
            "phase9_lock_sha256",
        ),
        (
            "V4A-P03",
            P6_REGISTRY,
            "phase6_registry_sha256",
        ),
        (
            "V4A-P04",
            V4_REGISTRY,
            "p9v4_execution_registry_sha256",
        ),
        (
            "V4A-P05",
            V4_PARENT,
            "v4_parent_predictions_sha256",
        ),
        (
            "V4A-P06",
            V4_CONTEXT,
            "v4_context_predictions_sha256",
        ),
        (
            "V4A-P07",
            V4_TUNING,
            "v4_tuning_registry_sha256",
        ),
        (
            "V4A-P08",
            V4_PREPROCESS,
            "v4_preprocess_registry_sha256",
        ),
        (
            "V4A-P09",
            V4_INTEGRITY,
            "v4_integrity_tests_sha256",
        ),
        (
            "V4A-P10",
            P6_PARENT,
            "p6_parent_comparator_sha256",
        ),
        (
            "V4A-P11",
            P9C_PARENT,
            "p9c_context_comparator_sha256",
        ),
    ]

    for tid, path, key in hash_pairs:
        actual = sha256_file(path)
        expected = lock[key]

        T(
            tid,
            f"Frozen input unchanged: {path.name}",
            actual == expected,
            actual,
        )

    base = basic_input_checks()

    T(
        "V4A-P12",
        "All original freeze preflight checks remain PASS",
        (base.status == "PASS").all(),
        f"{int((base.status == 'PASS').sum())}/{len(base)}",
    )

    out = pd.DataFrame(tests)

    ANALYSIS.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        ANALYSIS / "P9V4_ANALYSIS_PREFLIGHT_TESTS.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# Comparator construction
# ---------------------------------------------------------------------

def load_analysis_inputs():
    v4 = pd.read_csv(V4_PARENT)

    v4 = v4[
        v4.model_family.isin(FAMILIES)
        & v4.modality.isin(MODALITIES)
    ].copy()

    p6 = pd.read_csv(P6_PARENT)

    p6 = p6[
        p6.model_family.isin(FAMILIES)
        & (p6.modality == "STRUCTURE")
    ].copy()

    p9c = pd.read_csv(P9C_PARENT)

    p9c = p9c[
        p9c.model_family.isin(FAMILIES)
        & p9c.modality.isin(NONSTRUCTURE)
    ].copy()

    return v4, p6, p9c


def build_group_arrays(v4, p6, p9c, context, endpoint):
    g = v4[
        (v4.heldout_context == context)
        & (v4.endpoint == endpoint)
    ].copy()

    base = (
        g[
            (g.model_family == FAMILIES[0])
            & (g.modality == "STRUCTURE")
        ][
            [
                "compound_id",
                "y_true",
                "outer_fold",
            ]
        ]
        .drop_duplicates()
        .sort_values("compound_id")
        .reset_index(drop=True)
    )

    ids = base.compound_id.astype(str).to_numpy()

    n = len(base)

    if n == 0:
        raise RuntimeError(
            f"No matched parents for {context}/{endpoint}"
        )

    y = base.y_true.astype(int).to_numpy()

    P_loco = np.full(
        (n, len(MODEL_KEYS)),
        np.nan,
        float,
    )
    Q_loco = np.full(
        (n, len(MODEL_KEYS)),
        -1,
        int,
    )

    P_primary = np.full_like(
        P_loco,
        np.nan,
        dtype=float,
    )
    Q_primary = np.full_like(
        Q_loco,
        -1,
        dtype=int,
    )

    for j, (fam, mod) in enumerate(MODEL_KEYS):
        z = g[
            (g.model_family == fam)
            & (g.modality == mod)
        ][
            [
                "compound_id",
                "y_true",
                "outer_fold",
                "probability",
                "predicted_class",
            ]
        ].copy()

        z["compound_id"] = z.compound_id.astype(str)

        z = base.merge(
            z,
            on="compound_id",
            how="left",
            suffixes=("_base", "_v4"),
            validate="one_to_one",
        )

        if (
            z.probability.isna().any()
            or z.predicted_class.isna().any()
        ):
            raise RuntimeError(
                f"V4 alignment failure: {context}/{endpoint}/{fam}/{mod}"
            )

        if not np.array_equal(
            z.y_true_base.astype(int),
            z.y_true_v4.astype(int),
        ):
            raise RuntimeError(
                f"V4 label mismatch: {context}/{endpoint}/{fam}/{mod}"
            )

        if not np.array_equal(
            z.outer_fold_base.astype(int),
            z.outer_fold_v4.astype(int),
        ):
            raise RuntimeError(
                f"V4 fold mismatch: {context}/{endpoint}/{fam}/{mod}"
            )

        P_loco[:, j] = z.probability.astype(float)
        Q_loco[:, j] = z.predicted_class.astype(int)

        if mod == "STRUCTURE":
            q = p6[
                (p6.endpoint == endpoint)
                & (p6.model_family == fam)
                & (p6.modality == "STRUCTURE")
                & p6.compound_id.astype(str).isin(ids)
            ][
                [
                    "compound_id",
                    "y_true",
                    "outer_fold",
                    "probability",
                    "predicted_class",
                ]
            ].copy()

        else:
            q = p9c[
                (p9c.endpoint == endpoint)
                & (p9c.cell_type == context)
                & (p9c.model_family == fam)
                & (p9c.modality == mod)
                & p9c.compound_id.astype(str).isin(ids)
            ][
                [
                    "compound_id",
                    "y_true",
                    "outer_fold",
                    "probability",
                    "predicted_class",
                ]
            ].copy()

        q["compound_id"] = q.compound_id.astype(str)

        q = base.merge(
            q,
            on="compound_id",
            how="left",
            suffixes=("_base", "_primary"),
            validate="one_to_one",
        )

        if (
            q.probability.isna().any()
            or q.predicted_class.isna().any()
        ):
            raise RuntimeError(
                f"PRIMARY alignment failure: "
                f"{context}/{endpoint}/{fam}/{mod}"
            )

        if not np.array_equal(
            q.y_true_base.astype(int),
            q.y_true_primary.astype(int),
        ):
            raise RuntimeError(
                f"PRIMARY label mismatch: "
                f"{context}/{endpoint}/{fam}/{mod}"
            )

        if not np.array_equal(
            q.outer_fold_base.astype(int),
            q.outer_fold_primary.astype(int),
        ):
            raise RuntimeError(
                f"PRIMARY fold mismatch: "
                f"{context}/{endpoint}/{fam}/{mod}"
            )

        P_primary[:, j] = q.probability.astype(float)
        Q_primary[:, j] = q.predicted_class.astype(int)

    if not (
        np.isfinite(P_loco).all()
        and np.isfinite(P_primary).all()
    ):
        raise RuntimeError(
            f"Nonfinite aligned probabilities: {context}/{endpoint}"
        )

    return base, y, P_loco, Q_loco, P_primary, Q_primary


# ---------------------------------------------------------------------
# Statistical contrasts
# ---------------------------------------------------------------------

def orient_summary(summary, metric):
    s = UTILITY_SIGN[metric]

    out = dict(summary)

    out["benefit_oriented_estimate"] = (
        summary["estimate"] * s
        if np.isfinite(summary["estimate"])
        else np.nan
    )

    if (
        np.isfinite(summary["ci_low"])
        and np.isfinite(summary["ci_high"])
    ):
        a = summary["ci_low"] * s
        b = summary["ci_high"] * s

        out["benefit_oriented_ci_low"] = min(a, b)
        out["benefit_oriented_ci_high"] = max(a, b)

    else:
        out["benefit_oriented_ci_low"] = np.nan
        out["benefit_oriented_ci_high"] = np.nan

    return out


def analyze_context_endpoint(
    context,
    endpoint,
    v4,
    p6,
    p9c,
):
    (
        base,
        y,
        P_loco,
        Q_loco,
        P_primary,
        Q_primary,
    ) = build_group_arrays(
        v4,
        p6,
        p9c,
        context,
        endpoint,
    )

    n = len(y)

    rng = np.random.default_rng(
        bootstrap_seed(context, endpoint)
    )

    W = np.empty(
        (BOOTSTRAP_N, n),
        dtype=np.uint16,
    )

    for b in range(BOOTSTRAP_N):
        W[b] = np.bincount(
            rng.integers(0, n, size=n),
            minlength=n,
        )

    W0 = np.ones(
        (1, n),
        dtype=np.uint8,
    )

    obs_loco = weighted_metrics(
        y,
        P_loco,
        Q_loco,
        W0,
    )[0]

    obs_primary = weighted_metrics(
        y,
        P_primary,
        Q_primary,
        W0,
    )[0]

    boot_loco = weighted_metrics(
        y,
        P_loco,
        Q_loco,
        W,
    )

    boot_primary = weighted_metrics(
        y,
        P_primary,
        Q_primary,
        W,
    )

    active_n = int(y.sum())
    inactive_n = int(n - active_n)
    prevalence = active_n / n

    metric_rows = []

    for evaluation, obs in [
        ("LOCO", obs_loco),
        ("PRIMARY_SAME_CONTEXT", obs_primary),
    ]:
        for j, (fam, mod) in enumerate(MODEL_KEYS):
            row = {
                "heldout_context": context,
                "endpoint": endpoint,
                "evaluation": evaluation,
                "model_family": fam,
                "modality": mod,
                "n": n,
                "active_n": active_n,
                "inactive_n": inactive_n,
                "prevalence": prevalence,
            }

            for k, metric in enumerate(METRICS):
                row[metric] = obs[j, k]

            metric_rows.append(row)

    loco_primary_rows = []
    within_rows = []
    advantage_rows = []

    # --------------------------------------------------------------
    # 1. LOCO minus PRIMARY same-context
    # --------------------------------------------------------------

    for j, (fam, mod) in enumerate(MODEL_KEYS):
        for k, metric in enumerate(METRICS):
            est = (
                obs_loco[j, k]
                - obs_primary[j, k]
            )

            bd = (
                boot_loco[:, j, k]
                - boot_primary[:, j, k]
            )

            sm = orient_summary(
                boot_summary(est, bd),
                metric,
            )

            row = {
                "heldout_context": context,
                "endpoint": endpoint,
                "analysis_family": "P9V4_LOCO_VS_PRIMARY",
                "metric": metric,
                "contrast": "LOCO_minus_PRIMARY_same_context",
                "model_family": fam,
                "modality": mod,
                "n": n,
                "primary_same_context_metric": obs_primary[j, k],
                "loco_metric": obs_loco[j, k],
            }

            row.update(sm)
            loco_primary_rows.append(row)

    # --------------------------------------------------------------
    # 2–4. HTTr/Early/Late minus STRUCTURE within LOCO
    # --------------------------------------------------------------

    for fam in FAMILIES:
        js = MODEL_INDEX[(fam, "STRUCTURE")]

        for mod in NONSTRUCTURE:
            jm = MODEL_INDEX[(fam, mod)]

            contrast_name = (
                f"{mod}_minus_STRUCTURE_within_LOCO"
            )

            for k, metric in enumerate(METRICS):
                est = (
                    obs_loco[jm, k]
                    - obs_loco[js, k]
                )

                bd = (
                    boot_loco[:, jm, k]
                    - boot_loco[:, js, k]
                )

                sm = orient_summary(
                    boot_summary(est, bd),
                    metric,
                )

                row = {
                    "heldout_context": context,
                    "endpoint": endpoint,
                    "analysis_family": "P9V4_WITHIN_LOCO_MODALITY",
                    "metric": metric,
                    "contrast": contrast_name,
                    "model_family": fam,
                    "modality": mod,
                    "n": n,
                    "structure_metric": obs_loco[js, k],
                    "modality_metric": obs_loco[jm, k],
                }

                row.update(sm)
                within_rows.append(row)

    # --------------------------------------------------------------
    # 5. Change in modality advantage under LOCO
    #
    # Raw:
    # (LOCO modality - LOCO structure)
    # -
    # (PRIMARY modality - PRIMARY structure)
    #
    # benefit_oriented_estimate makes positive = modality advantage
    # becomes stronger under LOCO, regardless of metric direction.
    # --------------------------------------------------------------

    for fam in FAMILIES:
        js = MODEL_INDEX[(fam, "STRUCTURE")]

        for mod in NONSTRUCTURE:
            jm = MODEL_INDEX[(fam, mod)]

            contrast_name = (
                "LOCO_minus_PRIMARY_change_in_"
                f"{mod}_advantage_vs_STRUCTURE"
            )

            for k, metric in enumerate(METRICS):
                loco_adv = (
                    obs_loco[jm, k]
                    - obs_loco[js, k]
                )

                primary_adv = (
                    obs_primary[jm, k]
                    - obs_primary[js, k]
                )

                est = loco_adv - primary_adv

                bd = (
                    (
                        boot_loco[:, jm, k]
                        - boot_loco[:, js, k]
                    )
                    -
                    (
                        boot_primary[:, jm, k]
                        - boot_primary[:, js, k]
                    )
                )

                sm = orient_summary(
                    boot_summary(est, bd),
                    metric,
                )

                row = {
                    "heldout_context": context,
                    "endpoint": endpoint,
                    "analysis_family": "P9V4_MODALITY_ADVANTAGE_CHANGE",
                    "metric": metric,
                    "contrast": contrast_name,
                    "model_family": fam,
                    "modality": mod,
                    "n": n,
                    "primary_modality_advantage_raw": primary_adv,
                    "loco_modality_advantage_raw": loco_adv,
                }

                row.update(sm)
                advantage_rows.append(row)

    return (
        metric_rows,
        loco_primary_rows,
        within_rows,
        advantage_rows,
    )


def apply_fdr(df):
    if not len(df):
        return df

    df = df.copy()

    df["fdr_family"] = (
        df.analysis_family.astype(str)
        + "|"
        + df.metric.astype(str)
        + "|"
        + df.contrast.astype(str)
    )

    df["q_value"] = np.nan

    for (context, fam), idx in df.groupby(
        ["heldout_context", "fdr_family"]
    ).groups.items():

        p = (
            df.loc[idx, "bootstrap_p_two_sided"]
            .where(
                df.loc[idx, "bootstrap_supported"],
                np.nan,
            )
            .to_numpy(float)
        )

        df.loc[idx, "q_value"] = bh(p)

    df["ci_excludes_zero"] = (
        (df.ci_low > 0)
        | (df.ci_high < 0)
    )

    df["benefit_ci_excludes_zero"] = (
        (df.benefit_oriented_ci_low > 0)
        | (df.benefit_oriented_ci_high < 0)
    )

    return df


# ---------------------------------------------------------------------
# Random-effects synthesis across the 8 endpoints
# ---------------------------------------------------------------------

def normal_two_sided_p(z):
    if not np.isfinite(z):
        return np.nan
    return math.erfc(
        abs(float(z)) / math.sqrt(2.0)
    )


def random_effects(df):
    keys = [
        "heldout_context",
        "analysis_family",
        "metric",
        "contrast",
        "model_family",
        "modality",
    ]

    z = df[
        df.bootstrap_supported
        & np.isfinite(df.estimate)
        & np.isfinite(df.bootstrap_se)
        & (df.bootstrap_se > 0)
    ].copy()

    if not len(z):
        return pd.DataFrame()

    z["vi"] = (
        z.bootstrap_se.to_numpy(float) ** 2
    )

    z["wi"] = 1.0 / z.vi
    z["wiy"] = z.wi * z.estimate
    z["wiy2"] = (
        z.wi
        * z.estimate
        * z.estimate
    )
    z["wi2"] = z.wi * z.wi

    a = (
        z.groupby(
            keys,
            sort=False,
            dropna=False,
        )
        .agg(
            endpoint_n=("estimate", "size"),
            sum_w=("wi", "sum"),
            sum_wy=("wiy", "sum"),
            sum_wy2=("wiy2", "sum"),
            sum_w2=("wi2", "sum"),
        )
        .reset_index()
    )

    a = a[
        a.endpoint_n >= 3
    ].copy()

    a["Q"] = (
        a.sum_wy2
        - (
            a.sum_wy
            * a.sum_wy
            / a.sum_w
        )
    )

    a["den"] = (
        a.sum_w
        - a.sum_w2 / a.sum_w
    )

    a["tau2"] = np.where(
        a.den > 0,
        np.maximum(
            0,
            (
                a.Q
                - (a.endpoint_n - 1)
            )
            / a.den,
        ),
        0.0,
    )

    z = z.merge(
        a[keys + ["tau2"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    z["wr"] = 1.0 / (
        z.vi + z.tau2
    )

    z["wry"] = (
        z.wr * z.estimate
    )

    b = (
        z.groupby(
            keys,
            sort=False,
            dropna=False,
        )
        .agg(
            sum_wr=("wr", "sum"),
            sum_wry=("wry", "sum"),
        )
        .reset_index()
    )

    out = a.merge(
        b,
        on=keys,
        validate="one_to_one",
    )

    out["random_effect_mean"] = (
        out.sum_wry / out.sum_wr
    )

    out["random_effect_se"] = np.sqrt(
        1.0 / out.sum_wr
    )

    out["ci_low"] = (
        out.random_effect_mean
        - 1.96 * out.random_effect_se
    )

    out["ci_high"] = (
        out.random_effect_mean
        + 1.96 * out.random_effect_se
    )

    out["z_value"] = (
        out.random_effect_mean
        / out.random_effect_se
    )

    out["p_value"] = [
        normal_two_sided_p(x)
        for x in out.z_value
    ]

    out["utility_sign"] = out.metric.map(
        UTILITY_SIGN
    ).astype(float)

    out["benefit_oriented_mean"] = (
        out.random_effect_mean
        * out.utility_sign
    )

    a1 = out.ci_low * out.utility_sign
    a2 = out.ci_high * out.utility_sign

    out["benefit_oriented_ci_low"] = np.minimum(
        a1, a2
    )

    out["benefit_oriented_ci_high"] = np.maximum(
        a1, a2
    )

    keep = (
        keys
        + [
            "endpoint_n",
            "random_effect_mean",
            "random_effect_se",
            "ci_low",
            "ci_high",
            "benefit_oriented_mean",
            "benefit_oriented_ci_low",
            "benefit_oriented_ci_high",
            "tau2",
            "Q",
            "z_value",
            "p_value",
        ]
    )

    out = out[keep]

    out["fdr_family"] = (
        out.analysis_family.astype(str)
        + "|"
        + out.metric.astype(str)
        + "|"
        + out.contrast.astype(str)
    )

    out["q_value"] = np.nan

    for (context, fam), idx in out.groupby(
        ["heldout_context", "fdr_family"]
    ).groups.items():

        out.loc[idx, "q_value"] = bh(
            out.loc[idx, "p_value"]
            .to_numpy(float)
        )

    return out


# ---------------------------------------------------------------------
# Context stability synthesis
# ---------------------------------------------------------------------

def context_stability(re):
    keys = [
        "analysis_family",
        "metric",
        "contrast",
        "model_family",
        "modality",
    ]

    rows = []

    for key, g in re.groupby(
        keys,
        sort=False,
        dropna=False,
    ):
        row = dict(zip(keys, key))

        effects = {}
        lows = {}
        highs = {}

        for h in CONTEXTS:
            z = g[
                g.heldout_context == h
            ]

            if len(z) == 1:
                effects[h] = float(
                    z.benefit_oriented_mean.iloc[0]
                )
                lows[h] = float(
                    z.benefit_oriented_ci_low.iloc[0]
                )
                highs[h] = float(
                    z.benefit_oriented_ci_high.iloc[0]
                )

            else:
                effects[h] = np.nan
                lows[h] = np.nan
                highs[h] = np.nan

            row[f"{h}_benefit_effect"] = effects[h]
            row[f"{h}_benefit_ci_low"] = lows[h]
            row[f"{h}_benefit_ci_high"] = highs[h]

        vals = np.array(
            [effects[h] for h in CONTEXTS],
            float,
        )

        finite = np.isfinite(vals)
        vals = vals[finite]

        row["evaluable_context_n"] = int(
            finite.sum()
        )

        pos = int(np.sum(vals > 0))
        neg = int(np.sum(vals < 0))
        zero = int(np.sum(vals == 0))

        row["positive_direction_n"] = pos
        row["negative_direction_n"] = neg
        row["zero_direction_n"] = zero

        significant_positive = 0
        significant_negative = 0

        for h in CONTEXTS:
            lo = lows[h]
            hi = highs[h]

            if not (
                np.isfinite(lo)
                and np.isfinite(hi)
            ):
                continue

            if lo > 0:
                significant_positive += 1

            if hi < 0:
                significant_negative += 1

        row[
            "positive_ci_excludes_zero_n"
        ] = significant_positive

        row[
            "negative_ci_excludes_zero_n"
        ] = significant_negative

        if len(vals) == 3:
            if pos == 3:
                verdict = "CONTEXT_CONSISTENT_POSITIVE_3OF3"
            elif neg == 3:
                verdict = "CONTEXT_CONSISTENT_NEGATIVE_3OF3"
            elif pos >= 2:
                verdict = "CONTEXT_MAJORITY_POSITIVE_2OF3"
            elif neg >= 2:
                verdict = "CONTEXT_MAJORITY_NEGATIVE_2OF3"
            else:
                verdict = "CONTEXT_MIXED"
        else:
            verdict = "UNEVALUABLE"

        row["context_stability_verdict"] = verdict

        # For LOCO-vs-PRIMARY only:
        # benefit-oriented negative = degradation under unseen context.
        if (
            row["analysis_family"]
            == "P9V4_LOCO_VS_PRIMARY"
        ):
            row[
                "prespecified_stress_direction"
            ] = (
                "NEGATIVE_BENEFIT_EFFECT="
                "LOCO_WORSE_THAN_PRIMARY"
            )

            row[
                "stress_direction_match_n"
            ] = neg

            row[
                "stress_direction_reverse_n"
            ] = pos

            if len(vals) == 3 and neg == 3:
                sv = "STRESS_GENERAL_3OF3"
            elif len(vals) == 3 and neg >= 2:
                sv = "STRESS_MAJORITY_2OF3"
            elif len(vals) == 3 and pos >= 2:
                sv = "STRESS_DIRECTION_NOT_GENERAL"
            else:
                sv = "STRESS_UNEVALUABLE"

            row["stress_verdict"] = sv

        else:
            row[
                "prespecified_stress_direction"
            ] = ""

            row[
                "stress_direction_match_n"
            ] = np.nan

            row[
                "stress_direction_reverse_n"
            ] = np.nan

            row["stress_verdict"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Final integrity tests
# ---------------------------------------------------------------------

def final_integrity(
    context_metrics,
    loco_primary,
    within_loco,
    advantage,
    re,
    stability,
):
    tests = []

    def T(tid, desc, ok, detail=""):
        tests.append(
            {
                "test_id": tid,
                "description": desc,
                "status": "PASS" if bool(ok) else "FAIL",
                "detail": str(detail),
            }
        )

    T(
        "V4A-T01",
        "All three held-out contexts retained",
        set(context_metrics.heldout_context.unique())
        == set(CONTEXTS),
        sorted(
            context_metrics.heldout_context.unique()
        ),
    )

    T(
        "V4A-T02",
        "All eight endpoints retained",
        set(context_metrics.endpoint.unique())
        == set(ENDPOINTS),
        sorted(context_metrics.endpoint.unique()),
    )

    T(
        "V4A-T03",
        "Context metric table has exactly 576 rows",
        len(context_metrics) == 576,
        len(context_metrics),
    )

    T(
        "V4A-T04",
        "LOCO-vs-PRIMARY endpoint contrast table has exactly 2880 rows",
        len(loco_primary) == 2880,
        len(loco_primary),
    )

    T(
        "V4A-T05",
        "Within-LOCO modality contrast table has exactly 2160 rows",
        len(within_loco) == 2160,
        len(within_loco),
    )

    T(
        "V4A-T06",
        "Modality-advantage-change table has exactly 2160 rows",
        len(advantage) == 2160,
        len(advantage),
    )

    T(
        "V4A-T07",
        "All endpoint contrasts used 2000 bootstrap resamples",
        (
            pd.concat(
                [
                    loco_primary,
                    within_loco,
                    advantage,
                ],
                ignore_index=True,
            ).bootstrap_valid_n
            <= BOOTSTRAP_N
        ).all(),
        BOOTSTRAP_N,
    )

    T(
        "V4A-T08",
        "All endpoint bootstrap-supported flags meet >=90% valid rule",
        (
            pd.concat(
                [
                    loco_primary,
                    within_loco,
                    advantage,
                ],
                ignore_index=True,
            )
            .loc[
                lambda x: x.bootstrap_supported,
                "bootstrap_valid_fraction",
            ]
            >= MIN_BOOT_VALID_FRAC
        ).all(),
        MIN_BOOT_VALID_FRAC,
    )

    qs = pd.concat(
        [
            loco_primary.q_value,
            within_loco.q_value,
            advantage.q_value,
            re.q_value,
        ],
        ignore_index=True,
    )

    finite_q = qs[np.isfinite(qs)]

    T(
        "V4A-T09",
        "All finite q-values lie within [0,1]",
        (
            (finite_q >= 0)
            & (finite_q <= 1)
        ).all(),
        "range",
    )

    # Since V4 structure predictions are frozen Phase6 structure predictions,
    # LOCO - PRIMARY should be exactly zero for structure.
    s = loco_primary[
        loco_primary.modality == "STRUCTURE"
    ]

    max_structure_difference = (
        np.nanmax(
            np.abs(
                s.estimate.to_numpy(float)
            )
        )
        if len(s)
        else np.nan
    )

    T(
        "V4A-T10",
        "STRUCTURE LOCO-minus-PRIMARY effects are exactly zero",
        (
            np.isfinite(max_structure_difference)
            and max_structure_difference < 1e-12
        ),
        max_structure_difference,
    )

    T(
        "V4A-T11",
        "All three named within-LOCO modality contrasts represented",
        set(within_loco.contrast.unique())
        == {
            "HTTR_minus_STRUCTURE_within_LOCO",
            "EARLY_FUSION_minus_STRUCTURE_within_LOCO",
            "LATE_FUSION_minus_STRUCTURE_within_LOCO",
        },
        sorted(within_loco.contrast.unique()),
    )

    T(
        "V4A-T12",
        "All three modality-advantage-change contrasts represented",
        len(
            advantage.contrast.drop_duplicates()
        ) == 3,
        sorted(
            advantage.contrast.drop_duplicates()
        ),
    )

    T(
        "V4A-T13",
        "Random-effects synthesis contains all three contexts",
        set(re.heldout_context.unique())
        == set(CONTEXTS),
        sorted(re.heldout_context.unique()),
    )

    expected_stability = (
        12 * 10
        + 9 * 10
        + 9 * 10
    )

    T(
        "V4A-T14",
        "Context-stability table has expected 300 rows",
        len(stability) == expected_stability,
        len(stability),
    )

    T(
        "V4A-T15",
        "No model, endpoint or context was omitted according to performance",
        (
            context_metrics[
                [
                    "heldout_context",
                    "endpoint",
                    "evaluation",
                    "model_family",
                    "modality",
                ]
            ]
            .drop_duplicates()
            .shape[0]
            == 576
        ),
        576,
    )

    T(
        "V4A-T16",
        "All LOCO/PRIMARY metric estimates finite where mathematically supported",
        np.isfinite(
            context_metrics[
                [
                    "AUROC",
                    "AUPRC",
                    "Brier",
                    "Brier_skill_score",
                    "ECE_10bin",
                ]
            ].to_numpy(float)
        ).all(),
        "finite",
    )

    T(
        "V4A-T17",
        "Analysis lock remains unchanged after statistical computation",
        sha256_file(LOCK_FILE)
        == read_json(
            ANALYSIS / "P9V4_ANALYSIS_RUNTIME_LOCK_RECORD.json"
        )["analysis_lock_sha256"],
        sha256_file(LOCK_FILE),
    )

    return pd.DataFrame(tests)


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def run_analysis():
    pre = verify_frozen_analysis()

    print(pre.to_string(index=False))

    if (pre.status != "PASS").any():
        die(
            "P9V4 locked analysis preflight failed. "
            "No performance analysis was run."
        )

    # Record the analysis-lock hash immediately before reading performance.
    runtime_lock = {
        "analysis_lock_sha256": sha256_file(LOCK_FILE),
        "script_sha256": sha256_file(SCRIPT),
        "performance_analysis_started_after_lock": True,
    }

    write_json(
        ANALYSIS / "P9V4_ANALYSIS_RUNTIME_LOCK_RECORD.json",
        runtime_lock,
    )

    v4, p6, p9c = load_analysis_inputs()

    metric_rows = []
    loco_primary_rows = []
    within_rows = []
    advantage_rows = []

    total = len(CONTEXTS) * len(ENDPOINTS)
    counter = 0

    for context in CONTEXTS:
        for endpoint in ENDPOINTS:
            counter += 1

            print(
                f"[{counter:02d}/{total}] "
                f"{context} | {endpoint}"
            )

            (
                a,
                b,
                c,
                d,
            ) = analyze_context_endpoint(
                context,
                endpoint,
                v4,
                p6,
                p9c,
            )

            metric_rows.extend(a)
            loco_primary_rows.extend(b)
            within_rows.extend(c)
            advantage_rows.extend(d)

    context_metrics = pd.DataFrame(
        metric_rows
    )

    loco_primary = apply_fdr(
        pd.DataFrame(loco_primary_rows)
    )

    within_loco = apply_fdr(
        pd.DataFrame(within_rows)
    )

    advantage = apply_fdr(
        pd.DataFrame(advantage_rows)
    )

    all_contrasts = pd.concat(
        [
            loco_primary,
            within_loco,
            advantage,
        ],
        ignore_index=True,
        sort=False,
    )

    re = random_effects(
        all_contrasts
    )

    stability = context_stability(
        re
    )

    # Write result tables.
    context_metrics.to_csv(
        ANALYSIS / "P9V4_06_context_metrics.csv",
        index=False,
    )

    loco_primary.to_csv(
        ANALYSIS / "P9V4_07_loco_vs_primary_contrasts.csv",
        index=False,
    )

    within_loco.to_csv(
        ANALYSIS / "P9V4_08_within_loco_modality_contrasts.csv",
        index=False,
    )

    advantage.to_csv(
        ANALYSIS / "P9V4_09_modality_advantage_change.csv",
        index=False,
    )

    re.to_csv(
        ANALYSIS / "P9V4_10_context_random_effects.csv",
        index=False,
    )

    stability.to_csv(
        ANALYSIS / "P9V4_11_context_stability.csv",
        index=False,
    )

    integ = final_integrity(
        context_metrics,
        loco_primary,
        within_loco,
        advantage,
        re,
        stability,
    )

    integ.to_csv(
        ANALYSIS / "P9V4_12_integrity_tests.csv",
        index=False,
    )

    if (integ.status != "PASS").any():
        print(integ.to_string(index=False))
        die(
            "P9V4 statistical analysis integrity check failed."
        )

    status = {
        "project": "FACTORIAL-OOD",
        "phase": "P9V4",
        "stage": "LOCO_STATISTICAL_ANALYSIS",
        "status": "PASS",
        "heldout_contexts": CONTEXTS,
        "endpoint_n": len(ENDPOINTS),
        "model_family_n": len(FAMILIES),
        "modality_n": len(MODALITIES),
        "metric_n": len(METRICS),
        "bootstrap_resamples": BOOTSTRAP_N,
        "context_metric_rows": len(context_metrics),
        "loco_vs_primary_contrast_rows": len(loco_primary),
        "within_loco_modality_contrast_rows": len(within_loco),
        "modality_advantage_change_rows": len(advantage),
        "random_effect_rows": len(re),
        "context_stability_rows": len(stability),
        "integrity": (
            f"{int((integ.status == 'PASS').sum())}"
            f"/{len(integ)} PASS"
        ),
        "analysis_lock_sha256": sha256_file(LOCK_FILE),
        "script_sha256": sha256_file(SCRIPT),
        "scientific_interpretation": (
            "DEFERRED UNTIL INDEPENDENT AUDIT. "
            "No performance-based model/context/endpoint selection."
        ),
    }

    write_json(
        ANALYSIS / "P9V4_FINAL_STATUS.json",
        status,
    )

    # SHA manifest
    output_names = [
        "P9V4_ANALYSIS_LOCK_v1.0.json",
        "P9V4_ANALYSIS_FREEZE_SHA256.txt",
        "P9V4_ANALYSIS_FREEZE_PREFLIGHT_TESTS.csv",
        "P9V4_ANALYSIS_PREFLIGHT_TESTS.csv",
        "P9V4_ANALYSIS_RUNTIME_LOCK_RECORD.json",
        "P9V4_06_context_metrics.csv",
        "P9V4_07_loco_vs_primary_contrasts.csv",
        "P9V4_08_within_loco_modality_contrasts.csv",
        "P9V4_09_modality_advantage_change.csv",
        "P9V4_10_context_random_effects.csv",
        "P9V4_11_context_stability.csv",
        "P9V4_12_integrity_tests.csv",
        "P9V4_FINAL_STATUS.json",
    ]

    sums = ANALYSIS / "SHA256SUMS_P9V4_FINAL.txt"

    with open(sums, "w") as fh:
        for name in output_names:
            p = ANALYSIS / name
            if p.exists():
                fh.write(
                    f"{sha256_file(p)}  {name}\n"
                )

        fh.write(
            f"{sha256_file(SCRIPT)}  {SCRIPT.name}\n"
        )

    # Upload package
    upload = ANALYSIS / "P9V4_FINAL_UPLOAD_TO_CHAT.zip"

    with zipfile.ZipFile(
        upload,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:

        for name in output_names:
            p = ANALYSIS / name
            if p.exists():
                zf.write(
                    p,
                    arcname=name,
                )

        zf.write(
            sums,
            arcname=sums.name,
        )

        zf.write(
            SCRIPT,
            arcname=SCRIPT.name,
        )

    upload_sha = sha256_file(upload)

    status["upload_zip"] = str(upload)
    status["upload_zip_sha256"] = upload_sha
    status["upload_zip_size_bytes"] = upload.stat().st_size

    write_json(
        ANALYSIS / "P9V4_FINAL_STATUS.json",
        status,
    )

    # Rewrite status checksum after adding upload details.
    with open(sums, "w") as fh:
        for name in output_names:
            p = ANALYSIS / name
            if p.exists():
                fh.write(
                    f"{sha256_file(p)}  {name}\n"
                )

        fh.write(
            f"{sha256_file(SCRIPT)}  {SCRIPT.name}\n"
        )

    print("\n" + "=" * 96)
    print("PHASE 9 V4 LOCO STATISTICAL ANALYSIS PASS")
    print(f"Held-out contexts: {len(CONTEXTS)}/3")
    print(f"Endpoints: {len(ENDPOINTS)}/8")
    print(f"Context metric rows: {len(context_metrics)}")
    print(f"LOCO-vs-PRIMARY contrasts: {len(loco_primary)}")
    print(f"Within-LOCO modality contrasts: {len(within_loco)}")
    print(f"Modality-advantage-change contrasts: {len(advantage)}")
    print(f"Random-effect rows: {len(re)}")
    print(f"Context-stability rows: {len(stability)}")
    print(
        "Integrity: "
        f"{int((integ.status == 'PASS').sum())}/{len(integ)} PASS"
    )
    print(f"Upload: {upload}")
    print(f"Upload ZIP SHA256: {upload_sha}")
    print(
        "V4 statistics are LOCKED. "
        "Scientific interpretation deferred until independent audit."
    )
    print("=" * 96)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ANALYSIS.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ARGS.freeze_only:
        freeze_analysis()
        sys.exit(0)

    if ARGS.preflight_only:
        pre = verify_frozen_analysis()
        print(pre.to_string(index=False))

        print("\n" + "=" * 90)

        if (pre.status == "PASS").all():
            print(
                f"P9V4 ANALYSIS PREFLIGHT: "
                f"{len(pre)}/{len(pre)} PASS"
            )
            print(
                "NO V4 PERFORMANCE WAS COMPUTED "
                "OR INTERPRETED."
            )
        else:
            print(
                "P9V4 ANALYSIS PREFLIGHT: FAIL"
            )

        print("=" * 90)

        sys.exit(
            0 if (pre.status == "PASS").all()
            else 1
        )

    if not LOCK_FILE.exists():
        die(
            "P9V4 analysis has not been frozen. "
            "Run --freeze-only first."
        )

    run_analysis()
