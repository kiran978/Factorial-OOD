#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FACTORIAL-OOD — Phase 10B.2
Publication refinement of Figure 1 and Figure 2 (v2)

RENDER-ONLY:
- no retraining
- no prediction recalculation
- no threshold changes
- no support-definition changes
- no new inferential testing

Builds:
  Figure 1 v2 — Factorial validation framework
  Figure 2 v2 — Prevalence / metric discordance / optimism

Outputs:
  Phase10B_Output_v2/
    Figure1/Figure1_factorial_validation_framework_v2.pdf|svg|png
    Figure2/Figure2_prevalence_metric_discordance_v2.pdf|svg|png
    source_data/*.csv
    PHASE10B_FIG1_FIG2_V2_STATUS.json
    SHA256SUMS_PHASE10B_FIG1_FIG2_V2.txt
    PHASE10B_FIG1_FIG2_V2_UPLOAD.zip
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import transforms


# =============================================================================
# PROJECT CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Phase10B_Output_v2"

PRIMARY_CHEM = "CHEM_PRIMARY_HYBRID_050"
PRIMARY_BIO = "BIO_PRIMARY_Q75_MEDIAN"

CELLS = ["C0/B0", "C1/B0", "C0/B1", "C1/B1"]
FAMILIES = ["ELASTIC_NET", "RANDOM_FOREST", "HIST_GRAD_BOOST"]
MODALITIES = ["STRUCTURE", "HTTR", "EARLY_FUSION", "LATE_FUSION"]

FAMILY_LABEL = {
    "ELASTIC_NET": "Elastic Net",
    "RANDOM_FOREST": "Random Forest",
    "HIST_GRAD_BOOST": "HistGradientBoost",
}

FAMILY_SHORT = {
    "ELASTIC_NET": "EN",
    "RANDOM_FOREST": "RF",
    "HIST_GRAD_BOOST": "HGB",
}

MOD_LABEL = {
    "STRUCTURE": "Structure",
    "HTTR": "HTTr",
    "EARLY_FUSION": "Early fusion",
    "LATE_FUSION": "Late fusion",
}

MOD_MARKER = {
    "STRUCTURE": "o",
    "HTTR": "^",
    "EARLY_FUSION": "s",
    "LATE_FUSION": "D",
}

FULL_WIDTH_MM = 170.0
FULL_WIDTH_IN = FULL_WIDTH_MM / 25.4
DPI = 600

# inputs
P9A_CELLS = ROOT / "Phase9_Output/P9A/P9A_02_cell_metrics.csv"
P9A_RE = ROOT / "Phase9_Output/P9A/P9A_04_random_effects.csv"
P9A_STATUS = ROOT / "Phase9_Output/P9A/P9A_FINAL_STATUS.json"
CONTEXT_MANIFEST = ROOT / "Phase5B_Output/qa/P5B_context_row_index_manifest.csv"
P3_CHEM = ROOT / "Phase3/outputs/P3_03_chemical_support_by_parent.csv"
P3_BIO = ROOT / "Phase3/outputs/P3_04_biological_support_by_parent.csv"


# =============================================================================
# PLOT STYLE
# =============================================================================

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required frozen input: {path}")


def choose_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found: {candidates}")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label):
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fontsize=7.0,
    bold=False,
    fc="white",
    ec="black",
    lw=0.8,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=("bold" if bold else "normal"),
    )
    return patch


def add_arrow(
    ax,
    start,
    end,
    lw=0.9,
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color="black",
        )
    )


def save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# LOAD + VALIDATE FROZEN INPUTS
# =============================================================================

for p in [P9A_CELLS, P9A_RE, P9A_STATUS, CONTEXT_MANIFEST, P3_CHEM, P3_BIO]:
    require_exists(p)

with P9A_STATUS.open() as fh:
    p9a_status = json.load(fh)

if p9a_status.get("status") != "PASS":
    raise RuntimeError("P9A final status is not PASS.")

cells_df = pd.read_csv(P9A_CELLS)
re_df = pd.read_csv(P9A_RE)
context_df = pd.read_csv(CONTEXT_MANIFEST)
chem_df = pd.read_csv(P3_CHEM)
bio_df = pd.read_csv(P3_BIO)


# =============================================================================
# RECONSTRUCT GLOBAL FACTORIAL COUNTS FROM FROZEN PHASE 3 SUPPORT TABLES
# =============================================================================

merge_keys = ["compound_id"]
if "outer_fold" in chem_df.columns and "outer_fold" in bio_df.columns:
    merge_keys.append("outer_fold")

support_df = chem_df.merge(
    bio_df,
    on=merge_keys,
    how="inner",
    validate="one_to_one",
)

if len(support_df) != 2017:
    raise RuntimeError(
        f"Frozen Phase 3 support merge must contain 2,017 rows; observed {len(support_df)}."
    )

if support_df["compound_id"].nunique() != 2017:
    raise RuntimeError(
        f"Frozen support table must contain 2,017 unique parent compounds; observed "
        f"{support_df['compound_id'].nunique()}."
    )

if "chemical_state" not in support_df.columns:
    raise RuntimeError("Missing chemical_state in P3_03_chemical_support_by_parent.csv")

if "biological_state" not in support_df.columns:
    raise RuntimeError("Missing biological_state in P3_04_biological_support_by_parent.csv")

support_df["factorial_cell_global"] = (
    support_df["chemical_state"].astype(str)
    + "/"
    + support_df["biological_state"].astype(str)
)

global_factorial_counts = {
    cell: int((support_df["factorial_cell_global"] == cell).sum())
    for cell in CELLS
}

expected_global_counts = {
    "C0/B0": 1219,
    "C1/B0": 574,
    "C0/B1": 135,
    "C1/B1": 89,
}

if global_factorial_counts != expected_global_counts:
    raise RuntimeError(
        "Global frozen factorial counts failed reconstruction.\n"
        f"Observed: {global_factorial_counts}\n"
        f"Expected: {expected_global_counts}"
    )


# =============================================================================
# CONTEXT COUNTS
# =============================================================================

if len(context_df) != 4021:
    raise RuntimeError(f"Expected 4,021 context rows; observed {len(context_df)}.")

context_col = choose_col(context_df, ["cell_type", "context", "context_name"])

raw_counts = context_df[context_col].astype(str).value_counts().to_dict()
context_counts = {}

for key, value in raw_counts.items():
    k = key.strip().upper()
    if k == "MCF7":
        context_counts["MCF7"] = int(value)
    elif k == "U2OS":
        context_counts["U2OS"] = int(value)
    elif k in {"HEPRG", "HEPARG"}:
        context_counts["HepRG"] = int(value)
    else:
        context_counts[key.strip()] = int(value)

expected_context_counts = {"MCF7": 1674, "U2OS": 1173, "HepRG": 1174}
if context_counts != expected_context_counts:
    raise RuntimeError(
        "Context counts differ from frozen expected values.\n"
        f"Observed: {context_counts}\n"
        f"Expected: {expected_context_counts}"
    )


# =============================================================================
# PRIMARY-DEFINITION P9A DATA FOR FIGURE 2
# =============================================================================

primary_cells = cells_df[
    (cells_df["chemical_definition"] == PRIMARY_CHEM)
    & (cells_df["biological_definition"] == PRIMARY_BIO)
].copy()

primary_re = re_df[
    (re_df["chemical_definition"] == PRIMARY_CHEM)
    & (re_df["biological_definition"] == PRIMARY_BIO)
].copy()

if primary_cells["endpoint"].nunique() != 8:
    raise RuntimeError(
        f"Expected 8 frozen primary endpoints; observed {primary_cells['endpoint'].nunique()}."
    )

n_col = choose_col(primary_cells, ["n", "n_labeled"])
active_col = choose_col(primary_cells, ["active_n"])
inactive_col = choose_col(primary_cells, ["inactive_n"])
prev_col = choose_col(primary_cells, ["prevalence"])

prevalence_df = (
    primary_cells[
        ["endpoint", "factorial_cell", n_col, active_col, inactive_col, prev_col]
    ]
    .drop_duplicates()
    .rename(
        columns={
            n_col: "n",
            active_col: "active_n",
            inactive_col: "inactive_n",
            prev_col: "prevalence",
        }
    )
    .copy()
)

if prevalence_df.duplicated(["endpoint", "factorial_cell"]).any():
    raise RuntimeError("Endpoint × factorial-cell prevalence is not unique.")

pooled_prev_df = (
    prevalence_df.groupby("factorial_cell", as_index=False)
    .agg(active_n=("active_n", "sum"), n=("n", "sum"))
)

pooled_prev_df["pooled_prevalence"] = pooled_prev_df["active_n"] / pooled_prev_df["n"]

joint_effects = primary_re[
    (primary_re["analysis_family"] == "P9A_FACTORIAL")
    & (primary_re["contrast"] == "joint_vs_reference")
    & (primary_re["metric"].isin(["AUROC", "AUPRC"]))
].copy()

joint_pivot = joint_effects.pivot_table(
    index=["model_family", "modality"],
    columns="metric",
    values=["random_effect_mean", "ci_low", "ci_high", "q_value"],
    aggfunc="first",
)

joint_pivot.columns = [f"{a}_{b}" for a, b in joint_pivot.columns]
joint_pivot = joint_pivot.reset_index()

if len(joint_pivot) != 12:
    raise RuntimeError(
        f"Expected 12 model-family × modality rows for Figure 2B; observed {len(joint_pivot)}."
    )

optimism_df = primary_re[
    (primary_re["analysis_family"] == "P9A_OPTIMISM")
    & (primary_re["contrast"] == "pooled_OOF_minus_joint_OOD")
    & (primary_re["metric"] == "AUROC")
].copy()

if len(optimism_df) != 12:
    raise RuntimeError(
        f"Expected 12 AUROC optimism rows for Figure 2C; observed {len(optimism_df)}."
    )


# =============================================================================
# SOURCE DATA EXPORT
# =============================================================================

SOURCE_DIR = OUT / "source_data"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    [{"factorial_cell": c, "parent_n": global_factorial_counts[c]} for c in CELLS]
).to_csv(SOURCE_DIR / "Fig1B_factorial_parent_counts.csv", index=False)

pd.DataFrame(
    [{"context": c, "context_row_n": context_counts[c]} for c in ["MCF7", "U2OS", "HepRG"]]
).to_csv(SOURCE_DIR / "Fig1A_context_counts.csv", index=False)

prevalence_df.to_csv(SOURCE_DIR / "Fig2A_endpoint_prevalence.csv", index=False)
pooled_prev_df.to_csv(SOURCE_DIR / "Fig2A_pooled_prevalence.csv", index=False)
joint_pivot.to_csv(SOURCE_DIR / "Fig2B_AUROC_AUPRC_joint_effects.csv", index=False)
optimism_df.to_csv(SOURCE_DIR / "Fig2C_AUROC_optimism_random_effects.csv", index=False)


# =============================================================================
# FIGURE 1 V2
# =============================================================================

def make_figure1_v2():
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 7.5))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.985,
        bottom=0.06,
        top=0.975,
        wspace=0.18,
        hspace=0.22,
    )

    # -------------------------------------------------------------------------
    # Panel A — dataset flow
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, 0.08, 0.80, 0.84, 0.11, "Public HTTr and Tox21 assay resources", fontsize=7.5, bold=True)
    add_box(ax, 0.08, 0.62, 0.84, 0.11, "2,017 curated unique parent compounds", fontsize=7.6)
    add_box(ax, 0.08, 0.44, 0.84, 0.11, "4,021 parent–context observations", fontsize=7.6)
    add_box(ax, 0.08, 0.26, 0.84, 0.11, "41 candidate endpoints screened", fontsize=7.4)
    add_box(ax, 0.25, 0.08, 0.50, 0.10, "8 frozen primary endpoints", fontsize=7.6, bold=True)

    add_arrow(ax, (0.50, 0.79), (0.50, 0.74))
    add_arrow(ax, (0.50, 0.61), (0.50, 0.56))
    add_arrow(ax, (0.50, 0.43), (0.50, 0.38))
    add_arrow(ax, (0.50, 0.25), (0.50, 0.19))

    ax.text(
        0.50,
        0.01,
        "Context rows:  MCF7 = 1,674   |   U2OS = 1,173   |   HepRG = 1,174",
        ha="center",
        va="bottom",
        fontsize=6.9,
    )

    # -------------------------------------------------------------------------
    # Panel B — factorial support plane
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cell_face = {
        "C0/B0": "white",
        "C1/B0": "0.94",
        "C0/B1": "0.90",
        "C1/B1": "0.85",
    }

    cell_pos = {
        "C0/B0": (0, 0),
        "C1/B0": (1, 0),
        "C0/B1": (0, 1),
        "C1/B1": (1, 1),
    }

    cell_desc = {
        "C0/B0": "Reference",
        "C1/B0": "Chemical novelty",
        "C0/B1": "Biological novelty",
        "C1/B1": "Joint novelty",
    }

    for cell in CELLS:
        x, y = cell_pos[cell]
        rect = Rectangle((x, y), 1, 1, facecolor=cell_face[cell], edgecolor="black", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + 0.50, y + 0.64, cell, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(
            x + 0.50,
            y + 0.38,
            cell_desc[cell],
            ha="center",
            va="center",
            fontsize=7.1,
        )
        ax.text(
            x + 0.50,
            y + 0.20,
            f"N = {global_factorial_counts[cell]:,}",
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )

    ax.text(1.0, -0.14, "Chemical novelty  →", ha="center", va="center", fontsize=7.3)
    ax.text(-0.16, 1.0, "Biological novelty  ↑", ha="center", va="center", fontsize=7.3, rotation=90)
    ax.set_title("Frozen factorial applicability-domain design", pad=8)

    # -------------------------------------------------------------------------
    # Panel C — modality architecture (corrected)
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.96,
        "Prespecified feature families and fusion logic",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
    )

    # input features
    add_box(ax, 0.05, 0.77, 0.25, 0.10, "Morgan +\nphysicochemical features", fontsize=7.0)
    add_box(ax, 0.70, 0.77, 0.25, 0.10, "Fold-specific HTTr\nPCA/latent features", fontsize=7.0)

    # single-modality models
    add_box(ax, 0.05, 0.57, 0.25, 0.10, "Structure model", fontsize=7.1, bold=True)
    add_box(ax, 0.70, 0.57, 0.25, 0.10, "HTTr model", fontsize=7.1, bold=True)

    add_arrow(ax, (0.175, 0.77), (0.175, 0.68))
    add_arrow(ax, (0.825, 0.77), (0.825, 0.68))

    # early fusion feature-level path
    add_box(ax, 0.36, 0.72, 0.28, 0.08, "Feature concatenation", fontsize=6.9)
    add_box(ax, 0.36, 0.56, 0.28, 0.10, "Early-fusion model", fontsize=7.1, bold=True)
    ax.text(0.50, 0.83, "Feature level", ha="center", va="bottom", fontsize=6.7)

    add_arrow(ax, (0.30, 0.82), (0.36, 0.76))
    add_arrow(ax, (0.70, 0.82), (0.64, 0.76))
    add_arrow(ax, (0.50, 0.72), (0.50, 0.67))

    # late fusion prediction-level path
    add_box(ax, 0.36, 0.31, 0.28, 0.08, "Weighted probability fusion", fontsize=6.9)
    add_box(ax, 0.36, 0.16, 0.28, 0.10, "Late-fusion prediction", fontsize=7.1, bold=True)
    ax.text(0.50, 0.43, "Prediction level", ha="center", va="bottom", fontsize=6.7)

    add_arrow(ax, (0.30, 0.62), (0.36, 0.35))
    add_arrow(ax, (0.70, 0.62), (0.64, 0.35))
    add_arrow(ax, (0.50, 0.31), (0.50, 0.27))

    # family note
    ax.text(
        0.50,
        0.02,
        "Each modality evaluated with Elastic Net, Random Forest, and HistGradientBoost",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )

    # -------------------------------------------------------------------------
    # Panel D — robustness hierarchy
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.50,
        0.95,
        "Prespecified robustness hierarchy",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
    )

    stages = [
        ("Primary", "Grouped factorial OOD benchmark"),
        ("Support / AD", "Continuous support and applicability domain"),
        ("Robustness", "P9A: support-definition / severity combinations"),
        ("Robustness", "P9B: Route-B mean vs median aggregation"),
        ("Robustness", "P9C: context and missingness sensitivity"),
        ("Robustness", "P9D: four frozen alternative split sets"),
        ("Transport", "P9V4: leave-one-context-out biological transport"),
    ]

    y = 0.84
    for i, (group, text) in enumerate(stages):
        add_box(ax, 0.18, y - 0.045, 0.74, 0.08, text, fontsize=6.9)
        add_box(ax, 0.05, y - 0.045, 0.10, 0.08, group, fontsize=6.1, bold=True, fc="0.95")
        if i < len(stages) - 1:
            add_arrow(ax, (0.55, y - 0.048), (0.55, y - 0.095))
        y -= 0.11

    ax.text(
        0.50,
        0.02,
        "Stress tests qualify the primary claim; they do not replace the primary analysis",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )

    save_figure(fig, OUT / "Figure1/Figure1_factorial_validation_framework_v2")


# =============================================================================
# FIGURE 2 V2
# =============================================================================

def make_figure2_v2():
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 6.8))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.985,
        bottom=0.08,
        top=0.97,
        height_ratios=[1.00, 1.35],
        wspace=0.30,
        hspace=0.38,
    )

    # -------------------------------------------------------------------------
    # Panel A — prevalence by factorial regime
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")

    rng = np.random.default_rng(20260907)

    for i, cell in enumerate(CELLS):
        g = prevalence_df[prevalence_df["factorial_cell"] == cell].copy()
        jitter = rng.uniform(-0.09, 0.09, len(g))
        ax.scatter(
            np.full(len(g), i) + jitter,
            g["prevalence"].astype(float) * 100.0,
            s=18,
            alpha=0.60,
        )

        pooled_prev = float(
            pooled_prev_df.loc[
                pooled_prev_df["factorial_cell"] == cell,
                "pooled_prevalence",
            ].iloc[0]
            * 100.0
        )

        ax.scatter(
            [i],
            [pooled_prev],
            marker="D",
            s=58,
            zorder=5,
            label=("Pooled endpoint prevalence" if i == 0 else None),
        )

        ax.text(
            i,
            pooled_prev + 2.2,
            f"{pooled_prev:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(range(4), CELLS)
    ax.set_xlabel("Factorial support regime")
    ax.set_ylabel("Active prevalence (%)")
    ax.set_title("Outcome prevalence rises across novelty regimes", pad=8)
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)

    # -------------------------------------------------------------------------
    # Panel B — AUROC vs AUPRC discordance
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    clean_axes(ax)
    ax.axhline(0, linewidth=0.8, color="black")
    ax.axvline(0, linewidth=0.8, color="black")

    xvals = joint_pivot["random_effect_mean_AUROC"].astype(float).to_numpy()
    yvals = joint_pivot["random_effect_mean_AUPRC"].astype(float).to_numpy()

    xpad = max(0.01, 0.08 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-9))
    ypad = max(0.01, 0.08 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9))

    xmin = np.nanmin(xvals) - xpad
    xmax = np.nanmax(xvals) + xpad
    ymin = min(0.0, np.nanmin(yvals) - ypad)
    ymax = np.nanmax(yvals) + ypad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # dashed rectangle highlighting discordant quadrant
    rect = Rectangle(
        (xmin, 0),
        width=(0 - xmin),
        height=(ymax - 0),
        fill=False,
        linestyle="--",
        linewidth=0.9,
        edgecolor="black",
    )
    ax.add_patch(rect)

    ax.text(
        xmin + 0.02 * (xmax - xmin),
        ymax - 0.05 * (ymax - ymin),
        "Metric-discordant regime\nΔAUROC < 0, ΔAUPRC > 0",
        ha="left",
        va="top",
        fontsize=6.9,
        fontweight="bold",
    )

    label_offsets = {
        ("ELASTIC_NET", "STRUCTURE"): (-5, 2),
        ("ELASTIC_NET", "HTTR"): (4, 2),
        ("ELASTIC_NET", "EARLY_FUSION"): (4, -8),
        ("ELASTIC_NET", "LATE_FUSION"): (-12, -2),
        ("RANDOM_FOREST", "STRUCTURE"): (-10, 4),
        ("RANDOM_FOREST", "HTTR"): (4, 5),
        ("RANDOM_FOREST", "EARLY_FUSION"): (4, -9),
        ("RANDOM_FOREST", "LATE_FUSION"): (-10, -7),
        ("HIST_GRAD_BOOST", "STRUCTURE"): (-12, 5),
        ("HIST_GRAD_BOOST", "HTTR"): (4, 5),
        ("HIST_GRAD_BOOST", "EARLY_FUSION"): (4, -10),
        ("HIST_GRAD_BOOST", "LATE_FUSION"): (-12, -8),
    }

    for mod in MODALITIES:
        g = joint_pivot[joint_pivot["modality"] == mod].copy()

        ax.scatter(
            g["random_effect_mean_AUROC"].astype(float),
            g["random_effect_mean_AUPRC"].astype(float),
            s=46,
            marker=MOD_MARKER[mod],
            label=MOD_LABEL[mod],
            alpha=0.90,
        )

        for _, row in g.iterrows():
            dx, dy = label_offsets.get((row["model_family"], row["modality"]), (3, 3))
            ax.annotate(
                FAMILY_SHORT[row["model_family"]],
                (
                    float(row["random_effect_mean_AUROC"]),
                    float(row["random_effect_mean_AUPRC"]),
                ),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.0,
            )

    ax.set_xlabel("ΔAUROC: joint OOD − reference")
    ax.set_ylabel("ΔAUPRC: joint OOD − reference")
    ax.set_title("Joint novelty induces metric discordance", pad=8)
    ax.legend(frameon=False, loc="lower left", ncol=2)

    # -------------------------------------------------------------------------
    # Panel C — pooled AUROC optimism forest plot, grouped by family
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    panel_label(ax, "C")
    clean_axes(ax)
    ax.axvline(0, linewidth=0.8, color="black")

    opt = optimism_df.copy()
    opt["family_order"] = opt["model_family"].map({f: i for i, f in enumerate(FAMILIES)})
    opt["mod_order"] = opt["modality"].map({m: i for i, m in enumerate(MODALITIES)})
    opt = opt.sort_values(["family_order", "mod_order"]).reset_index(drop=True)

    y_lookup = {}
    y = 0
    gaps = []
    for family in FAMILIES:
        fam_rows = opt[opt["model_family"] == family]
        for mod in MODALITIES:
            mask = (opt["model_family"] == family) & (opt["modality"] == mod)
            if mask.any():
                idx = opt.index[mask][0]
                y_lookup[idx] = y
                y += 1
        gaps.append(y - 0.5)
        y += 0.7  # gap between families

    opt["y"] = [y_lookup[i] for i in opt.index]
    opt = opt.sort_values("y", ascending=False).reset_index(drop=True)

    # draw family separators
    family_midpoints = {}
    for family in FAMILIES:
        yy = opt.loc[opt["model_family"] == family, "y"].to_numpy(float)
        family_midpoints[family] = float(np.mean(yy))
        y_min = float(np.min(yy)) - 0.5
        ax.hlines(y=y_min, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linewidth=0.0)

    # CIs + markers
    for _, row in opt.iterrows():
        est = float(row["random_effect_mean"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        qv = float(row["q_value"]) if pd.notna(row["q_value"]) else np.nan
        yy = float(row["y"])
        marker = MOD_MARKER[row["modality"]]

        ax.hlines(yy, lo, hi, linewidth=1.0, color="black")

        if np.isfinite(qv) and qv < 0.05:
            ax.scatter(est, yy, marker=marker, s=34, facecolors="black", edgecolors="black", zorder=3)
        else:
            ax.scatter(est, yy, marker=marker, s=34, facecolors="white", edgecolors="black", zorder=3)

    # y labels
    yticklabels = [f"{MOD_LABEL[m]}" for m in opt["modality"]]
    ax.set_yticks(opt["y"], yticklabels)

    # family labels in left margin
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for family in FAMILIES:
        ax.text(
            -0.17,
            family_midpoints[family],
            FAMILY_SHORT[family],
            transform=trans,
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
        )

    # separators between families
    fam_sorted = [family_midpoints[f] for f in FAMILIES]
    yy_by_family = {
        f: sorted(opt.loc[opt["model_family"] == f, "y"].to_numpy(float))
        for f in FAMILIES
    }
    # draw between families
    for f1, f2 in zip(FAMILIES[:-1], FAMILIES[1:]):
        boundary = (max(yy_by_family[f2]) + min(yy_by_family[f1])) / 2.0
        ax.axhline(boundary, linewidth=0.6, linestyle="--", color="0.6")

    ax.set_xlabel("Pooled OOF AUROC − joint-OOD AUROC")
    ax.set_title("Pooled evaluation can overestimate joint-OOD AUROC", pad=10)

    filled_handle = plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", label="q < 0.05")
    open_handle = plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="black", label="q ≥ 0.05")
    modality_handles = [
        plt.Line2D([], [], marker=MOD_MARKER[m], linestyle="None", color="black", label=MOD_LABEL[m])
        for m in MODALITIES
    ]

    ax.legend(
        handles=modality_handles + [filled_handle, open_handle],
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.5,
        columnspacing=1.0,
    )

    ax.text(
        0.99,
        0.02,
        "Positive effects indicate optimistic pooled validation relative to joint OOD",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
    )

    save_figure(fig, OUT / "Figure2/Figure2_prevalence_metric_discordance_v2")


# =============================================================================
# BUILD FIGURES
# =============================================================================

OUT.mkdir(parents=True, exist_ok=True)
make_figure1_v2()
make_figure2_v2()


# =============================================================================
# STATUS JSON
# =============================================================================

status = {
    "project": "FACTORIAL-OOD",
    "phase": "10B.2",
    "status": "PASS",
    "version": "v2",
    "render_only": True,
    "model_retraining": False,
    "prediction_recalculation": False,
    "threshold_change": False,
    "support_definition_change": False,
    "new_inferential_testing": False,
    "primary_chemical_definition": PRIMARY_CHEM,
    "primary_biological_definition": PRIMARY_BIO,
    "figure_outputs": [
        "Figure1/Figure1_factorial_validation_framework_v2",
        "Figure2/Figure2_prevalence_metric_discordance_v2",
    ],
    "factorial_counts": global_factorial_counts,
    "context_counts": context_counts,
    "source_sha256": {
        "P3_chemical_support": sha256(P3_CHEM),
        "P3_biological_support": sha256(P3_BIO),
        "P9A_cell_metrics": sha256(P9A_CELLS),
        "P9A_random_effects": sha256(P9A_RE),
        "P5B_context_manifest": sha256(CONTEXT_MANIFEST),
    },
    "notes": [
        "Figure 1 Panel C corrected to distinguish feature-level early fusion from prediction-level late fusion.",
        "Figure 2 Panel B now highlights the metric-discordant quadrant without implying universality.",
        "Figure 2 Panel C uses filled/open markers to encode FDR support and reduce label clutter.",
    ],
}

status_path = OUT / "PHASE10B_FIG1_FIG2_V2_STATUS.json"
with status_path.open("w") as fh:
    json.dump(status, fh, indent=2)


# =============================================================================
# CHECKSUM MANIFEST
# =============================================================================

manifest_path = OUT / "SHA256SUMS_PHASE10B_FIG1_FIG2_V2.txt"

files_to_hash = sorted(
    p
    for p in OUT.rglob("*")
    if p.is_file()
    and p.name not in {
        manifest_path.name,
        "PHASE10B_FIG1_FIG2_V2_UPLOAD.zip",
    }
)

with manifest_path.open("w") as fh:
    for p in files_to_hash:
        fh.write(f"{sha256(p)}  {p.relative_to(OUT)}\n")


# =============================================================================
# ZIP PACKAGE
# =============================================================================

upload_zip = OUT / "PHASE10B_FIG1_FIG2_V2_UPLOAD.zip"
with zipfile.ZipFile(upload_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in files_to_hash + [manifest_path]:
        zf.write(p, arcname=str(p.relative_to(OUT)))


# =============================================================================
# FINAL CONSOLE OUTPUT
# =============================================================================

print()
print("=" * 100)
print("PHASE 10B.2 FIGURE REFINEMENT v2 PASS")
print("Figure 1:", OUT / "Figure1/Figure1_factorial_validation_framework_v2.pdf")
print("Figure 2:", OUT / "Figure2/Figure2_prevalence_metric_discordance_v2.pdf")
print("Vector outputs: PDF + SVG")
print(f"Review outputs: PNG @ {DPI} dpi")
print("Render-only: no retraining / no prediction recalculation / no new inference.")
print("Upload:", upload_zip)
print("Upload SHA256:", sha256(upload_zip))
print("=" * 100)
