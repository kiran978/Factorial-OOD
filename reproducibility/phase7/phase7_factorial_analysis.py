#!/usr/bin/env python3

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import hashlib
import json
import math
import re
import warnings
import zipfile

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from scipy.special import logit
from scipy.stats import norm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

ROOT = Path(__file__).resolve().parent
P4 = ROOT / "Phase4"
P6 = ROOT / "Phase6_Output"
P7 = ROOT / "Phase7_Output"
BOOTDIR = P7 / "bootstrap"
P7.mkdir(exist_ok=True)
BOOTDIR.mkdir(exist_ok=True)

BOOTSTRAPS = 2000
JOBS = int(os.environ.get("P7_JOBS", "8"))

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

CELLS = ["C0/B0", "C1/B0", "C0/B1", "C1/B1"]
CELL_INDEX = {x:i for i,x in enumerate(CELLS)}

EXPECTED_CELL_COUNTS = {
    "C0/B0": 1219,
    "C1/B0": 574,
    "C0/B1": 135,
    "C1/B1": 89,
}

PRIMARY_METRICS = ["AUPRC", "MCC"]
BOOT_CELL_METRICS = ["AUPRC", "MCC", "Brier", "ECE_10bin"]

tests = []


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def sha256_file(path, chunk=8*1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_for(*parts):
    txt = "|".join(["FACTORIAL-OOD", "P7", "v1"] + list(map(str, parts)))
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8], 16)


def add_test(tid, desc, ok, detail=""):
    tests.append({
        "test_id": tid,
        "description": desc,
        "status": "PASS" if bool(ok) else "FAIL",
        "detail": str(detail),
    })


def parse_factorial_cell(v):
    if pd.isna(v):
        return None
    x = str(v).upper().strip()
    x = x.replace(" ", "")
    x = x.replace("_", "/")
    m = re.search(r"C([01]).*B([01])", x)
    if not m:
        return None
    return f"C{m.group(1)}/B{m.group(2)}"


def parse_c_state(v):
    if pd.isna(v):
        return None
    x = str(v).upper().strip().replace(" ", "")
    if re.fullmatch(r"C0", x):
        return "C0"
    if re.fullmatch(r"C1", x):
        return "C1"
    return None


def parse_b_state(v):
    if pd.isna(v):
        return None
    x = str(v).upper().strip().replace(" ", "")
    if re.fullmatch(r"B0", x):
        return "B0"
    if re.fullmatch(r"B1", x):
        return "B1"
    return None


def collapse_mapping(df, state_col, parser, outcol):
    z = pd.DataFrame({
        "compound_id": df["compound_id"].astype(str),
        outcol: df[state_col].map(parser)
    }).dropna()

    if len(z) == 0:
        return None

    nunique = z.groupby("compound_id")[outcol].nunique()
    if (nunique > 1).any():
        return None

    z = z.drop_duplicates("compound_id")
    return z


def discover_factorial_assignment():
    csvs = sorted(P4.rglob("*.csv")) + sorted((ROOT / "Phase3").rglob("*.csv"))
    discovery = []
    exact_candidates = []
    c_candidates = []
    b_candidates = []

    for path in csvs:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue

        if "compound_id" not in df.columns:
            continue

        # Direct factorial-cell candidates.
        for col in df.columns:
            parsed = df[col].map(parse_factorial_cell)
            nparsed = int(parsed.notna().sum())

            if nparsed < 100:
                continue

            z = pd.DataFrame({
                "compound_id": df["compound_id"].astype(str),
                "factorial_cell": parsed
            }).dropna()

            if len(z) == 0:
                continue

            conflict = z.groupby("compound_id")["factorial_cell"].nunique()
            if (conflict > 1).any():
                continue

            z = z.drop_duplicates("compound_id")
            counts = z["factorial_cell"].value_counts().to_dict()

            exact = (
                len(z) == 2017
                and all(int(counts.get(k, 0)) == v
                        for k,v in EXPECTED_CELL_COUNTS.items())
            )

            discovery.append({
                "candidate_type": "DIRECT",
                "file": str(path),
                "column_1": col,
                "column_2": "",
                "parent_n": len(z),
                "cell_counts": json.dumps(counts, sort_keys=True),
                "exact_expected_counts": exact,
            })

            if exact:
                exact_candidates.append((
                    f"DIRECT:{path.name}:{col}",
                    z[["compound_id","factorial_cell"]].copy()
                ))

        # Separate C/B state candidates.
        for col in df.columns:
            cm = collapse_mapping(df, col, parse_c_state, "C_state")
            if cm is not None and len(cm) >= 1500:
                if set(cm["C_state"].unique()) == {"C0","C1"}:
                    c_candidates.append((str(path), col, cm))

            bm = collapse_mapping(df, col, parse_b_state, "B_state")
            if bm is not None and len(bm) >= 1500:
                if set(bm["B_state"].unique()) == {"B0","B1"}:
                    b_candidates.append((str(path), col, bm))

    # Try separate chemical and biological mapping tables.
    for cp, cc, cm in c_candidates:
        for bp, bc, bm in b_candidates:
            try:
                z = cm.merge(bm, on="compound_id", how="inner",
                             validate="one_to_one")
            except Exception:
                continue

            z["factorial_cell"] = z["C_state"] + "/" + z["B_state"]
            counts = z["factorial_cell"].value_counts().to_dict()

            exact = (
                len(z) == 2017
                and all(int(counts.get(k, 0)) == v
                        for k,v in EXPECTED_CELL_COUNTS.items())
            )

            discovery.append({
                "candidate_type": "MERGED_CB",
                "file": cp,
                "column_1": cc,
                "column_2": f"{bp}::{bc}",
                "parent_n": len(z),
                "cell_counts": json.dumps(counts, sort_keys=True),
                "exact_expected_counts": exact,
            })

            if exact:
                exact_candidates.append((
                    f"MERGED:{Path(cp).name}:{cc}+{Path(bp).name}:{bc}",
                    z[["compound_id","factorial_cell"]].copy()
                ))

    pd.DataFrame(discovery).to_csv(
        P7 / "P7_12_assignment_discovery_log.csv", index=False
    )

    if not exact_candidates:
        raise RuntimeError(
            "No Phase-4 parent-level factorial assignment reproduced "
            "the frozen 1219/574/135/89 counts. "
            "See P7_12_assignment_discovery_log.csv."
        )

    # All exact candidates must encode the same mapping.
    canonical = None
    descriptions = []

    for desc, z in exact_candidates:
        z = z.sort_values("compound_id").reset_index(drop=True)
        descriptions.append(desc)

        if canonical is None:
            canonical = z
        else:
            if not canonical.equals(z):
                raise RuntimeError(
                    "Multiple exact-count Phase-4 mappings disagree at "
                    "compound level. Analysis stopped."
                )

    outer_path = (
        P4 / "outputs" /
        "P4_01_primary_outer_split_manifest_FROZEN.csv"
    )
    outer = pd.read_csv(outer_path)
    outer = outer[["compound_id","outer_fold"]].copy()
    outer["compound_id"] = outer["compound_id"].astype(str)

    canonical = canonical.merge(
        outer, on="compound_id", how="left", validate="one_to_one"
    )

    if canonical["outer_fold"].isna().any():
        raise RuntimeError("Factorial mapping does not fully align to P4 outer split.")

    canonical["outer_fold"] = canonical["outer_fold"].astype(int)
    canonical["assignment_source"] = " | ".join(sorted(descriptions))
    canonical = canonical[
        ["compound_id","outer_fold","factorial_cell","assignment_source"]
    ]

    return canonical


def ece_equal_width(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0, 1, bins+1)
    n = len(y)
    if n == 0:
        return np.nan

    ans = 0.0
    for i in range(bins):
        if i == bins-1:
            m = (p >= edges[i]) & (p <= edges[i+1])
        else:
            m = (p >= edges[i]) & (p < edges[i+1])

        if m.any():
            ans += (
                m.sum()/n
                * abs(float(p[m].mean()) - float(y[m].mean()))
            )
    return float(ans)


def calibration_intercept_slope(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    if np.std(p) < 1e-12:
        return np.nan, np.nan

    try:
        x = logit(p).reshape(-1,1)
        m = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000
        )
        m.fit(x, y)
        return float(m.intercept_[0]), float(m.coef_[0,0])
    except Exception:
        return np.nan, np.nan


def metric_value(y, p, pred, metric):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=int)

    # Locked support rule: unstable one-class bootstrap cells are unsupported.
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan

    if metric == "AUPRC":
        return float(average_precision_score(y, p))

    if metric == "MCC":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(matthews_corrcoef(y, pred))

    if metric == "Brier":
        return float(brier_score_loss(y, p))

    if metric == "ECE_10bin":
        return ece_equal_width(y, p, 10)

    raise ValueError(metric)


def full_metric_bundle(g):
    y = g["y_true"].astype(int).to_numpy()
    p = g["probability"].astype(float).to_numpy()
    pred = g["predicted_class"].astype(int).to_numpy()

    n = len(y)
    active = int(y.sum())
    inactive = int((y == 0).sum())

    out = {
        "n": n,
        "active_n": active,
        "inactive_n": inactive,
        "prevalence": float(y.mean()) if n else np.nan,
    }

    if len(np.unique(y)) < 2:
        for m in [
            "AUPRC","MCC","AUROC","balanced_accuracy",
            "sensitivity","specificity","Brier","ECE_10bin",
            "calibration_intercept","calibration_slope"
        ]:
            out[m] = np.nan
        return out

    out["AUPRC"] = float(average_precision_score(y,p))
    out["MCC"] = float(matthews_corrcoef(y,pred))
    out["AUROC"] = float(roc_auc_score(y,p))
    out["balanced_accuracy"] = float(
        balanced_accuracy_score(y,pred)
    )

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0,1]
    ).ravel()

    out["sensitivity"] = (
        tp/(tp+fn) if (tp+fn) else np.nan
    )
    out["specificity"] = (
        tn/(tn+fp) if (tn+fp) else np.nan
    )
    out["Brier"] = float(brier_score_loss(y,p))
    out["ECE_10bin"] = ece_equal_width(y,p)

    cint, cslope = calibration_intercept_slope(y,p)
    out["calibration_intercept"] = cint
    out["calibration_slope"] = cslope

    return out


def boot_summary(estimate, boot):
    boot = np.asarray(boot, dtype=float)
    valid = boot[np.isfinite(boot)]

    valid_n = len(valid)
    valid_fraction = valid_n / len(boot)

    if not np.isfinite(estimate) or valid_n == 0:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_se": np.nan,
            "bootstrap_p_two_sided": np.nan,
            "bootstrap_valid_n": valid_n,
            "bootstrap_valid_fraction": valid_fraction,
            "bootstrap_supported": False,
        }

    ci_low, ci_high = np.quantile(valid, [0.025,0.975])

    se = float(np.std(valid, ddof=1)) if valid_n > 1 else np.nan

    # Centered-bootstrap two-sided p-value.
    deviation = valid - estimate
    pval = (
        1 + np.sum(np.abs(deviation) >= abs(estimate))
    ) / (valid_n + 1)

    return {
        "estimate": float(estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_se": se,
        "bootstrap_p_two_sided": float(min(1.0,pval)),
        "bootstrap_valid_n": valid_n,
        "bootstrap_valid_fraction": float(valid_fraction),
        "bootstrap_supported": bool(valid_fraction >= 0.90),
    }


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)

    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return q

    pv = p[valid_idx]
    order = np.argsort(pv)
    ranked = pv[order]

    m = len(ranked)
    adj = ranked * m / np.arange(1,m+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)

    q[valid_idx[order]] = adj
    return q


def bootstrap_endpoint(endpoint, joined_path, model_keys):
    df = pd.read_csv(
        joined_path,
        usecols=[
            "endpoint","compound_id","y_true",
            "model_family","modality",
            "probability","predicted_class",
            "factorial_cell"
        ]
    )

    df = df[
        (df["endpoint"] == endpoint)
        & df["model_family"].isin(FAMILIES)
        & df["modality"].isin(MODALITIES)
    ].copy()

    meta = (
        df[["compound_id","y_true","factorial_cell"]]
        .drop_duplicates()
        .sort_values("compound_id")
        .reset_index(drop=True)
    )

    if meta["compound_id"].duplicated().any():
        raise RuntimeError(f"{endpoint}: duplicate endpoint compound unit")

    compounds = meta["compound_id"].astype(str).to_numpy()
    y = meta["y_true"].astype(int).to_numpy()
    cell_code = (
        meta["factorial_cell"].map(CELL_INDEX).astype(int).to_numpy()
    )

    n = len(meta)
    jn = len(model_keys)

    probs = np.empty((n,jn), dtype=np.float32)
    classes = np.empty((n,jn), dtype=np.int8)

    for j,(fam,mod) in enumerate(model_keys):
        g = df[
            (df["model_family"] == fam)
            & (df["modality"] == mod)
        ][["compound_id","probability","predicted_class"]]

        if g["compound_id"].duplicated().any():
            raise RuntimeError(
                f"{endpoint} {fam} {mod}: duplicate prediction"
            )

        g = g.set_index("compound_id").reindex(compounds)

        if g.isna().any().any():
            raise RuntimeError(
                f"{endpoint} {fam} {mod}: incomplete model alignment"
            )

        probs[:,j] = g["probability"].to_numpy(np.float32)
        classes[:,j] = g["predicted_class"].to_numpy(np.int8)

    # Observed cell and pooled metrics.
    obs_cell = np.full(
        (jn,4,len(BOOT_CELL_METRICS)),
        np.nan,
        dtype=float
    )

    obs_pool = np.full((jn,2), np.nan, dtype=float)

    for j in range(jn):
        for c in range(4):
            m = cell_code == c
            for mi,metric in enumerate(BOOT_CELL_METRICS):
                obs_cell[j,c,mi] = metric_value(
                    y[m], probs[m,j], classes[m,j], metric
                )

        for mi,metric in enumerate(PRIMARY_METRICS):
            obs_pool[j,mi] = metric_value(
                y, probs[:,j], classes[:,j], metric
            )

    B = BOOTSTRAPS

    cell_boot = np.full(
        (B,jn,4,len(BOOT_CELL_METRICS)),
        np.nan,
        dtype=np.float32
    )

    pool_boot = np.full(
        (B,jn,2),
        np.nan,
        dtype=np.float32
    )

    rng = np.random.default_rng(
        seed_for(endpoint, "PARENT_CLUSTER_BOOTSTRAP")
    )

    for b in range(B):
        idx = rng.integers(0,n,size=n)

        yb = y[idx]
        cb = cell_code[idx]

        for j in range(jn):
            pb = probs[idx,j]
            prb = classes[idx,j]

            for mi,metric in enumerate(PRIMARY_METRICS):
                pool_boot[b,j,mi] = metric_value(
                    yb,pb,prb,metric
                )

            for c in range(4):
                m = cb == c
                if not m.any():
                    continue

                yc = yb[m]
                pc = pb[m]
                qc = prb[m]

                for mi,metric in enumerate(BOOT_CELL_METRICS):
                    cell_boot[b,j,c,mi] = metric_value(
                        yc,pc,qc,metric
                    )

    np.savez_compressed(
        BOOTDIR / f"P7_BOOT_{safe(endpoint)}.npz",
        cell_boot=cell_boot,
        pool_boot=pool_boot,
        model_family=np.asarray(
            [x[0] for x in model_keys], dtype=str
        ),
        modality=np.asarray(
            [x[1] for x in model_keys], dtype=str
        ),
        cells=np.asarray(CELLS,dtype=str),
        cell_metrics=np.asarray(BOOT_CELL_METRICS,dtype=str),
        primary_metrics=np.asarray(PRIMARY_METRICS,dtype=str),
        bootstrap_n=np.asarray([B],dtype=int),
        compound_n=np.asarray([n],dtype=int),
    )

    # Cell summaries.
    cell_rows = []

    for j,(fam,mod) in enumerate(model_keys):
        for c,cell in enumerate(CELLS):
            m0 = cell_code == c
            ncell = int(m0.sum())
            active = int(y[m0].sum())
            inactive = int((y[m0]==0).sum())

            row = {
                "endpoint": endpoint,
                "model_family": fam,
                "modality": mod,
                "factorial_cell": cell,
                "n": ncell,
                "active_n": active,
                "inactive_n": inactive,
                "prevalence": float(y[m0].mean()),
            }

            for mi,metric in enumerate(BOOT_CELL_METRICS):
                s = boot_summary(
                    obs_cell[j,c,mi],
                    cell_boot[:,j,c,mi]
                )
                row[metric] = s["estimate"]
                row[f"{metric}_ci_low"] = s["ci_low"]
                row[f"{metric}_ci_high"] = s["ci_high"]
                row[f"{metric}_bootstrap_valid_n"] = (
                    s["bootstrap_valid_n"]
                )
                row[f"{metric}_bootstrap_supported"] = (
                    s["bootstrap_supported"]
                )

            cell_rows.append(row)

    contrast_rows = []

    def add_contrast(
        hypothesis, metric, contrast,
        fam, mod, estimate, boot,
        extra=None
    ):
        s = boot_summary(estimate, boot)
        row = {
            "endpoint": endpoint,
            "hypothesis": hypothesis,
            "metric": metric,
            "contrast": contrast,
            "model_family": fam,
            "modality": mod,
            **s,
        }
        if extra:
            row.update(extra)
        contrast_rows.append(row)

    # H1/H2 + pooled optimism.
    for j,(fam,mod) in enumerate(model_keys):
        for metric in PRIMARY_METRICS:
            mi = BOOT_CELL_METRICS.index(metric)
            pi = PRIMARY_METRICS.index(metric)

            r = obs_cell[j,CELL_INDEX["C0/B0"],mi]
            ch = obs_cell[j,CELL_INDEX["C1/B0"],mi]
            bi = obs_cell[j,CELL_INDEX["C0/B1"],mi]
            jo = obs_cell[j,CELL_INDEX["C1/B1"],mi]

            br = cell_boot[:,j,CELL_INDEX["C0/B0"],mi]
            bch = cell_boot[:,j,CELL_INDEX["C1/B0"],mi]
            bbi = cell_boot[:,j,CELL_INDEX["C0/B1"],mi]
            bjo = cell_boot[:,j,CELL_INDEX["C1/B1"],mi]

            contrasts = [
                (
                    "chemical_shift",
                    ch-r,
                    bch-br
                ),
                (
                    "biological_shift",
                    bi-r,
                    bbi-br
                ),
                (
                    "joint_vs_reference",
                    jo-r,
                    bjo-br
                ),
                (
                    "joint_vs_chemical",
                    jo-ch,
                    bjo-bch
                ),
                (
                    "joint_vs_biological",
                    jo-bi,
                    bjo-bbi
                ),
            ]

            for cname,est,bd in contrasts:
                add_contrast(
                    "H1_PERFORMANCE_SHIFT",
                    metric,cname,fam,mod,est,bd
                )

            interaction = jo-ch-bi+r
            binteraction = bjo-bch-bbi+br

            add_contrast(
                "H2_FACTORIAL_INTERACTION",
                metric,
                "factorial_interaction",
                fam,mod,
                interaction,
                binteraction
            )

            pooled = obs_pool[j,pi]
            bpooled = pool_boot[:,j,pi]

            add_contrast(
                "OPTIMISM",
                metric,
                "pooled_OOF_minus_joint_OOD",
                fam,mod,
                pooled-jo,
                bpooled-bjo,
                extra={
                    "pooled_OOF_metric": pooled,
                    "joint_OOD_metric": jo,
                }
            )

        # H4 reliability: raw positive difference means larger loss.
        for metric in ["Brier","ECE_10bin"]:
            mi = BOOT_CELL_METRICS.index(metric)

            r = obs_cell[j,CELL_INDEX["C0/B0"],mi]
            ch = obs_cell[j,CELL_INDEX["C1/B0"],mi]
            bi = obs_cell[j,CELL_INDEX["C0/B1"],mi]
            jo = obs_cell[j,CELL_INDEX["C1/B1"],mi]

            br = cell_boot[:,j,CELL_INDEX["C0/B0"],mi]
            bch = cell_boot[:,j,CELL_INDEX["C1/B0"],mi]
            bbi = cell_boot[:,j,CELL_INDEX["C0/B1"],mi]
            bjo = cell_boot[:,j,CELL_INDEX["C1/B1"],mi]

            add_contrast(
                "H4_RELIABILITY_SHIFT",
                metric,
                "joint_vs_reference",
                fam,mod,
                jo-r,
                bjo-br
            )

            add_contrast(
                "H4_RELIABILITY_SHIFT",
                metric,
                "factorial_interaction",
                fam,mod,
                jo-ch-bi+r,
                bjo-bch-bbi+br
            )

    # H3 fusion robustness.
    model_index = {k:i for i,k in enumerate(model_keys)}

    for fam in FAMILIES:
        js = model_index[(fam,"STRUCTURE")]

        for fusion_mod in ["EARLY_FUSION","LATE_FUSION"]:
            jf = model_index[(fam,fusion_mod)]

            for metric in PRIMARY_METRICS:
                mi = BOOT_CELL_METRICS.index(metric)

                sref = obs_cell[js,CELL_INDEX["C0/B0"],mi]
                sjoint = obs_cell[js,CELL_INDEX["C1/B1"],mi]

                fref = obs_cell[jf,CELL_INDEX["C0/B0"],mi]
                fjoint = obs_cell[jf,CELL_INDEX["C1/B1"],mi]

                ref_adv = fref-sref
                joint_adv = fjoint-sjoint
                estimate = joint_adv-ref_adv

                bsref = cell_boot[:,js,CELL_INDEX["C0/B0"],mi]
                bsjoint = cell_boot[:,js,CELL_INDEX["C1/B1"],mi]

                bfref = cell_boot[:,jf,CELL_INDEX["C0/B0"],mi]
                bfjoint = cell_boot[:,jf,CELL_INDEX["C1/B1"],mi]

                bdist = (
                    (bfjoint-bsjoint)
                    -
                    (bfref-bsref)
                )

                add_contrast(
                    "H3_FUSION_ROBUSTNESS",
                    metric,
                    "joint_minus_reference_fusion_advantage",
                    fam,
                    fusion_mod,
                    estimate,
                    bdist,
                    extra={
                        "reference_fusion_advantage": ref_adv,
                        "joint_fusion_advantage": joint_adv,
                    }
                )

    return {
        "endpoint": endpoint,
        "cell_rows": cell_rows,
        "contrast_rows": contrast_rows,
    }


def random_effects_synthesis(df):
    rows = []

    group_cols = [
        "hypothesis","metric","contrast",
        "model_family","modality"
    ]

    for keys,g in df.groupby(group_cols,dropna=False):
        z = g[
            g["bootstrap_supported"]
            & np.isfinite(g["estimate"])
            & np.isfinite(g["bootstrap_se"])
            & (g["bootstrap_se"] > 0)
        ].copy()

        k = len(z)
        if k < 3:
            continue

        yi = z["estimate"].to_numpy(float)
        sei = z["bootstrap_se"].to_numpy(float)

        vi = sei**2
        wi = 1.0/vi

        fixed = np.sum(wi*yi)/np.sum(wi)
        Q = np.sum(wi*(yi-fixed)**2)

        denom = np.sum(wi) - np.sum(wi**2)/np.sum(wi)
        tau2 = max(
            0.0,
            (Q-(k-1))/denom
        ) if denom > 0 else 0.0

        wr = 1.0/(vi+tau2)
        mu = np.sum(wr*yi)/np.sum(wr)
        se_mu = math.sqrt(1.0/np.sum(wr))

        zl = mu/se_mu if se_mu > 0 else np.nan
        p = (
            2*norm.sf(abs(zl))
            if np.isfinite(zl) else np.nan
        )

        lo = mu - 1.96*se_mu
        hi = mu + 1.96*se_mu

        rows.append({
            "hypothesis": keys[0],
            "metric": keys[1],
            "contrast": keys[2],
            "model_family": keys[3],
            "modality": keys[4],
            "endpoint_n": k,
            "random_effect_mean": mu,
            "random_effect_se": se_mu,
            "ci_low": lo,
            "ci_high": hi,
            "tau2": tau2,
            "Q": Q,
            "p_value": p,
            "note":
                "Supportive random-effects synthesis across endpoints; "
                "outer folds are not treated as independent replicates."
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["fdr_family"] = (
            out["hypothesis"].astype(str)
            + "|" + out["metric"].astype(str)
            + "|" + out["contrast"].astype(str)
        )
        out["q_value"] = np.nan

        for fam,idx in out.groupby("fdr_family").groups.items():
            out.loc[idx,"q_value"] = bh_adjust(
                out.loc[idx,"p_value"].to_numpy()
            )

    return out


def main():
    print("="*80)
    print("FACTORIAL-OOD PHASE 7 — PRIMARY FACTORIAL OOD ANALYSIS")
    print("="*80)

    lock_path = P7 / "P7_00_factorial_analysis_lock_v1.0.json"
    if not lock_path.exists():
        raise RuntimeError(
            "Phase 7 preflight lock is missing. "
            "Run phase7_preflight.py first."
        )

    lock = json.load(open(lock_path))
    lock_sha = sha256_file(lock_path)

    add_test(
        "P7-T01",
        "Phase 7 lock exists before factorial analysis",
        lock.get(
            "created_before_factorial_performance_inspection"
        ) is True,
        lock_sha
    )

    # Phase 6 independent validation must be PASS.
    val_path = P6 / "P6_INDEPENDENT_VALIDATION_SUMMARY.json"
    if not val_path.exists():
        raise RuntimeError(
            "Missing P6_INDEPENDENT_VALIDATION_SUMMARY.json"
        )

    v = json.load(open(val_path))
    add_test(
        "P7-T02",
        "Phase 6 independent validation is PASS",
        v.get("decision") == "PASS",
        v
    )

    pred_path = P6 / "P6_03_locked_parent_OOF_predictions.csv"
    pred = pd.read_csv(pred_path)

    add_test(
        "P7-T03",
        "Locked Phase 6 OOF table has 150,000 rows",
        len(pred) == 150000,
        len(pred)
    )

    # ------------------------------------------------------------
    # Discover + validate parent-level factorial assignment
    # ------------------------------------------------------------
    assignment = discover_factorial_assignment()

    counts = assignment["factorial_cell"].value_counts().to_dict()

    add_test(
        "P7-T04",
        "Factorial assignment contains all 2,017 parents",
        len(assignment) == 2017,
        len(assignment)
    )

    add_test(
        "P7-T05",
        "Parent factorial counts reproduce frozen v0.2 counts",
        all(
            int(counts.get(k,0)) == val
            for k,val in EXPECTED_CELL_COUNTS.items()
        ),
        counts
    )

    assignment.to_csv(
        P7 / "P7_01_factorial_parent_assignment_VALIDATED.csv",
        index=False
    )

    # Validate parent mapping against all 160 P4_10 rows.
    units = (
        pred[["endpoint","outer_fold","compound_id","y_true"]]
        .drop_duplicates()
    )

    if units.duplicated(["endpoint","compound_id"]).any():
        raise RuntimeError(
            "Phase 6 endpoint-compound units are not unique."
        )

    units["compound_id"] = units["compound_id"].astype(str)

    u = units.merge(
        assignment[["compound_id","outer_fold","factorial_cell"]],
        on="compound_id",
        how="left",
        suffixes=("_p6","_p4"),
        validate="many_to_one"
    )

    if u["factorial_cell"].isna().any():
        raise RuntimeError(
            "Some Phase 6 endpoint compounds lack factorial assignment."
        )

    if not (
        u["outer_fold_p6"].astype(int)
        ==
        u["outer_fold_p4"].astype(int)
    ).all():
        raise RuntimeError(
            "Phase 4 factorial assignment outer-fold mismatch."
        )

    calc = (
        u.groupby(
            ["endpoint","outer_fold_p6","factorial_cell"],
            as_index=False
        )
        .agg(
            n_labeled=("y_true","size"),
            active_n=("y_true","sum")
        )
        .rename(columns={"outer_fold_p6":"outer_fold"})
    )
    calc["inactive_n"] = calc["n_labeled"] - calc["active_n"]

    p410 = pd.read_csv(
        P4 / "outputs" /
        "P4_10_primary_endpoint_factorial_support_by_outer_fold.csv"
    )

    check = p410.merge(
        calc,
        on=["endpoint","outer_fold","factorial_cell"],
        how="outer",
        suffixes=("_P4","_RECONSTRUCTED"),
        indicator=True
    )

    for c in ["n_labeled","active_n","inactive_n"]:
        check[f"{c}_match"] = (
            check[f"{c}_P4"]
            ==
            check[f"{c}_RECONSTRUCTED"]
        )

    exact160 = (
        len(check) == 160
        and (check["_merge"] == "both").all()
        and check[
            [
                "n_labeled_match",
                "active_n_match",
                "inactive_n_match"
            ]
        ].all().all()
    )

    check.to_csv(
        P7 / "P7_10_factorial_assignment_reconstruction_check.csv",
        index=False
    )

    add_test(
        "P7-T06",
        "All 160 P4_10 endpoint×fold factorial support rows reproduce exactly",
        exact160,
        f"rows={len(check)}"
    )

    if not exact160:
        raise RuntimeError(
            "Factorial mapping failed exact P4_10 reconstruction."
        )

    # ------------------------------------------------------------
    # Join immutable predictions to frozen factorial cells
    # ------------------------------------------------------------
    pred["compound_id"] = pred["compound_id"].astype(str)

    joined = pred.merge(
        assignment[["compound_id","factorial_cell"]],
        on="compound_id",
        how="left",
        validate="many_to_one"
    )

    add_test(
        "P7-T07",
        "All 150,000 OOF rows receive a factorial cell",
        joined["factorial_cell"].notna().all(),
        int(joined["factorial_cell"].isna().sum())
    )

    joined_path = P7 / "P7_02_locked_OOF_with_factorial_cells.csv"
    joined.to_csv(joined_path,index=False)

    learned = joined[
        joined["model_family"].isin(FAMILIES)
        & joined["modality"].isin(MODALITIES)
    ].copy()

    model_keys = [
        (fam,mod)
        for fam in FAMILIES
        for mod in MODALITIES
    ]

    present_keys = set(map(
        tuple,
        learned[
            ["model_family","modality"]
        ].drop_duplicates().to_numpy()
    ))

    add_test(
        "P7-T08",
        "Exactly 12 learned family-modality combinations present",
        present_keys == set(model_keys),
        len(present_keys)
    )

    # Every pooled Primary endpoint/cell must have both classes.
    support = (
        units.merge(
            assignment[["compound_id","factorial_cell"]],
            on="compound_id",
            how="left"
        )
        .groupby(
            ["endpoint","factorial_cell"],
            as_index=False
        )
        .agg(
            n=("y_true","size"),
            active_n=("y_true","sum")
        )
    )
    support["inactive_n"] = support["n"]-support["active_n"]

    add_test(
        "P7-T09",
        "All 32 pooled endpoint×factorial cells contain both classes",
        len(support) == 32
        and (support["active_n"] > 0).all()
        and (support["inactive_n"] > 0).all(),
        {
            "rows":len(support),
            "min_active":int(support["active_n"].min()),
            "min_inactive":int(support["inactive_n"].min())
        }
    )

    # ------------------------------------------------------------
    # Descriptive control-cell metrics
    # ------------------------------------------------------------
    controls = joined[
        ~joined["model_family"].isin(FAMILIES)
    ].copy()

    control_rows = []

    for keys,g in controls.groupby(
        ["endpoint","model_family","modality","factorial_cell"],
        sort=False
    ):
        row = {
            "endpoint":keys[0],
            "model_family":keys[1],
            "modality":keys[2],
            "factorial_cell":keys[3],
        }
        row.update(full_metric_bundle(g))
        control_rows.append(row)

    control_metrics = pd.DataFrame(control_rows)
    control_metrics.to_csv(
        P7 / "P7_04_control_factorial_cell_metrics.csv",
        index=False
    )

    # ------------------------------------------------------------
    # 2,000 paired parent-cluster bootstrap
    # ------------------------------------------------------------
    print(
        f"Starting {BOOTSTRAPS} parent-cluster bootstrap resamples "
        f"for each of 8 endpoints using {JOBS} workers..."
    )

    results = Parallel(
        n_jobs=JOBS,
        backend="loky",
        verbose=10
    )(
        delayed(bootstrap_endpoint)(
            ep, joined_path, model_keys
        )
        for ep in ENDPOINTS
    )

    cell_rows = []
    contrast_rows = []

    for r in results:
        cell_rows.extend(r["cell_rows"])
        contrast_rows.extend(r["contrast_rows"])

    cell_metrics = pd.DataFrame(cell_rows)
    contrasts = pd.DataFrame(contrast_rows)

    add_test(
        "P7-T10",
        "Learned factorial cell table has 384 rows",
        len(cell_metrics) == 8*12*4,
        len(cell_metrics)
    )

    add_test(
        "P7-T11",
        "All primary AUPRC/MCC cell bootstraps meet ≥90% valid criterion",
        (
            cell_metrics[
                "AUPRC_bootstrap_supported"
            ].all()
            and cell_metrics[
                "MCC_bootstrap_supported"
            ].all()
        )
    )

    cell_metrics.to_csv(
        P7 / "P7_03_learned_factorial_cell_metrics.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Multiplicity control
    # ------------------------------------------------------------
    contrasts["fdr_family"] = (
        contrasts["hypothesis"].astype(str)
        + "|" + contrasts["metric"].astype(str)
        + "|" + contrasts["contrast"].astype(str)
    )

    contrasts["q_value"] = np.nan

    for fam,idx in contrasts.groupby("fdr_family").groups.items():
        use = contrasts.loc[idx].copy()

        p = use["bootstrap_p_two_sided"].where(
            use["bootstrap_supported"],
            np.nan
        ).to_numpy(float)

        contrasts.loc[idx,"q_value"] = bh_adjust(p)

    contrasts["ci_excludes_zero"] = (
        (contrasts["ci_low"] > 0)
        |
        (contrasts["ci_high"] < 0)
    )

    contrasts["fdr_significant_0_05"] = (
        contrasts["q_value"] <= 0.05
    )

    add_test(
        "P7-T12",
        "All confirmatory contrast bootstrap distributions meet ≥90% validity",
        contrasts["bootstrap_supported"].all(),
        int((~contrasts["bootstrap_supported"]).sum())
    )

    qok = (
        contrasts["q_value"].dropna().between(0,1).all()
    )

    add_test(
        "P7-T13",
        "All BH-FDR q-values lie in [0,1]",
        qok
    )

    contrasts.to_csv(
        P7 / "P7_05_confirmatory_factorial_contrasts.csv",
        index=False
    )

    contrasts[
        contrasts["hypothesis"] == "H3_FUSION_ROBUSTNESS"
    ].to_csv(
        P7 / "P7_06_fusion_robustness.csv",
        index=False
    )

    contrasts[
        contrasts["hypothesis"] == "OPTIMISM"
    ].to_csv(
        P7 / "P7_07_optimism_contrasts.csv",
        index=False
    )

    contrasts[
        contrasts["hypothesis"] == "H4_RELIABILITY_SHIFT"
    ].to_csv(
        P7 / "P7_08_reliability_contrasts.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Supportive across-endpoint random-effects synthesis
    # ------------------------------------------------------------
    synth = random_effects_synthesis(contrasts)

    synth.to_csv(
        P7 / "P7_09_random_effects_endpoint_synthesis.csv",
        index=False
    )

    add_test(
        "P7-T14",
        "Eight endpoint bootstrap archives were generated",
        len(list(BOOTDIR.glob("P7_BOOT_*.npz"))) == 8,
        len(list(BOOTDIR.glob("P7_BOOT_*.npz")))
    )

    # Preserve pooled cell support table.
    support.to_csv(
        P7 / "P7_13_primary_endpoint_pooled_factorial_support.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Final integrity / status
    # ------------------------------------------------------------
    tests_df = pd.DataFrame(tests)

    tests_df.to_csv(
        P7 / "P7_11_integrity_tests.csv",
        index=False
    )

    if (tests_df["status"] != "PASS").any():
        print(
            tests_df[
                tests_df["status"] != "PASS"
            ].to_string(index=False)
        )
        raise RuntimeError("Phase 7 integrity test failure.")

    status = {
        "phase": 7,
        "status": "PASS",
        "analysis": "Primary Factorial OOD Analysis",
        "protocol": "v0.2",
        "phase7_lock_sha256": lock_sha,
        "bootstrap_resamples_per_endpoint": BOOTSTRAPS,
        "bootstrap_unit": "unique parent compound",
        "primary_endpoint_n": 8,
        "learned_model_combinations": 12,
        "factorial_cells": CELLS,
        "factorial_parent_counts": {
            k:int(v)
            for k,v in counts.items()
        },
        "p4_10_exact_reconstruction": True,
        "learned_factorial_cell_metric_rows": int(
            len(cell_metrics)
        ),
        "confirmatory_contrast_rows": int(
            len(contrasts)
        ),
        "integrity_tests": (
            f"{len(tests_df)}/{len(tests_df)} PASS"
        ),
        "interpretation_status":
            "Statistical outputs generated; scientific interpretation "
            "must occur only from these locked Phase-7 outputs.",
    }

    with open(P7 / "P7_FINAL_STATUS.json","w") as f:
        json.dump(status,f,indent=2)

    # ------------------------------------------------------------
    # SHA manifest
    # ------------------------------------------------------------
    manifest = P7 / "SHA256SUMS_PHASE7.txt"

    files = [
        p for p in P7.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS_PHASE7.txt"
        and p.name != "Phase7_UPLOAD_TO_CHAT.zip"
    ]

    with open(manifest,"w") as f:
        for p in sorted(files):
            f.write(
                f"{sha256_file(p)}  "
                f"{p.relative_to(P7)}\n"
            )

    # ------------------------------------------------------------
    # Upload-friendly complete Phase 7 package
    # ------------------------------------------------------------
    upload = P7 / "Phase7_UPLOAD_TO_CHAT.zip"

    with zipfile.ZipFile(
        upload,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as zf:
        for p in sorted(P7.rglob("*")):
            if not p.is_file() or p == upload:
                continue
            zf.write(
                p,
                arcname=str(p.relative_to(P7))
            )

    up_sha = sha256_file(upload)

    status["upload_zip"] = str(upload)
    status["upload_zip_sha256"] = up_sha
    status["upload_zip_size_bytes"] = upload.stat().st_size

    with open(P7 / "P7_FINAL_STATUS.json","w") as f:
        json.dump(status,f,indent=2)

    print()
    print("="*80)
    print("PHASE 7 LOCAL COMPUTATION PASS")
    print(
        "Frozen parent cells:",
        {k:int(counts.get(k,0)) for k in CELLS}
    )
    print("P4_10 reconstruction: 160/160 EXACT")
    print(
        f"Bootstrap: {BOOTSTRAPS} parent-cluster "
        f"resamples × 8 endpoints"
    )
    print(
        "Learned factorial-cell rows:",
        len(cell_metrics)
    )
    print(
        "Confirmatory contrast rows:",
        len(contrasts)
    )
    print(
        "Integrity:",
        f"{len(tests_df)}/{len(tests_df)} PASS"
    )
    print("Upload to ChatGPT:", upload)
    print("Upload ZIP SHA256:", up_sha)
    print(
        "Do not rerun/reselect models based on the Phase 7 results."
    )
    print("="*80)


if __name__ == "__main__":
    main()
