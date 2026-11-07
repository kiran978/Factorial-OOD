#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# ============================================================================
# PROJECT LOCK
# ============================================================================

ROOT = Path(
    "/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem"
)

OUT = ROOT / "Phase10B_Output"

PRIMARY_CHEM = "CHEM_PRIMARY_HYBRID_050"
PRIMARY_BIO = "BIO_PRIMARY_Q75_MEDIAN"

CELLS = [
    "C0/B0",
    "C1/B0",
    "C0/B1",
    "C1/B1",
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

DPI = 600
FULL_WIDTH_IN = 170 / 25.4


# ============================================================================
# INPUTS
# ============================================================================

P9A_COUNTS = (
    ROOT
    / "Phase9_Output/P9A/"
    "P9A_01_definition_cell_counts.csv"
)

P9A_CELLS = (
    ROOT
    / "Phase9_Output/P9A/"
    "P9A_02_cell_metrics.csv"
)

P9A_RE = (
    ROOT
    / "Phase9_Output/P9A/"
    "P9A_04_random_effects.csv"
)

P9A_STATUS = (
    ROOT
    / "Phase9_Output/P9A/"
    "P9A_FINAL_STATUS.json"
)

CONTEXT_MANIFEST = (
    ROOT
    / "Phase5B_Output/qa/"
    "P5B_context_row_index_manifest.csv"
)


P3_CHEM = (
    ROOT
    / "Phase3/outputs/"
    "P3_03_chemical_support_by_parent.csv"
)

P3_BIO = (
    ROOT
    / "Phase3/outputs/"
    "P3_04_biological_support_by_parent.csv"
)


# ============================================================================
# VISUAL STYLE
# ============================================================================

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ============================================================================
# HELPERS
# ============================================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as fh:
        for chunk in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fontsize=7.2,
):

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            "round,pad=0.012,"
            "rounding_size=0.018"
        ),
        fill=False,
        linewidth=0.8,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def add_arrow(ax, start, end):

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.8,
        )
    )


def save_figure(fig, stem):

    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# PREFLIGHT
# ============================================================================

required = [
    P9A_COUNTS,
    P9A_CELLS,
    P9A_RE,
    P9A_STATUS,
    CONTEXT_MANIFEST,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing frozen input:\n{path}"
        )


status = json.load(
    open(P9A_STATUS)
)

if status.get("status") != "PASS":

    raise RuntimeError(
        "P9A status is not PASS."
    )


counts = pd.read_csv(
    P9A_COUNTS
)

cells = pd.read_csv(
    P9A_CELLS
)

random_effects = pd.read_csv(
    P9A_RE
)

contexts = pd.read_csv(
    CONTEXT_MANIFEST
)


# ============================================================================
# PRIMARY DEFINITION ONLY
# ============================================================================

primary_cells = cells[
    (
        cells["chemical_definition"]
        == PRIMARY_CHEM
    )
    &
    (
        cells["biological_definition"]
        == PRIMARY_BIO
    )
].copy()


primary_re = random_effects[
    (
        random_effects["chemical_definition"]
        == PRIMARY_CHEM
    )
    &
    (
        random_effects["biological_definition"]
        == PRIMARY_BIO
    )
].copy()


if primary_cells["endpoint"].nunique() != 8:

    raise RuntimeError(
        "Expected exactly 8 primary endpoints."
    )


# ============================================================================
# FIGURE 1 SOURCE DATA
# Global factorial counts must use all 2,017 frozen parents.
# P9A_01 counts are endpoint-specific eligible subsets and are not used here.
# ============================================================================

if not P3_CHEM.exists():
    raise FileNotFoundError(
        f"Missing frozen chemical-support file: {P3_CHEM}"
    )

if not P3_BIO.exists():
    raise FileNotFoundError(
        f"Missing frozen biological-support file: {P3_BIO}"
    )

chem_support = pd.read_csv(P3_CHEM)
bio_support = pd.read_csv(P3_BIO)

merge_keys = ["compound_id"]

if (
    "outer_fold" in chem_support.columns
    and "outer_fold" in bio_support.columns
):
    merge_keys.append("outer_fold")

support = chem_support.merge(
    bio_support,
    on=merge_keys,
    how="inner",
    validate="one_to_one",
)

if len(support) != 2017:
    raise RuntimeError(
        "Frozen Phase 3 support merge must contain "
        f"2,017 parents; observed {len(support)}."
    )

if support["compound_id"].nunique() != 2017:
    raise RuntimeError(
        "Frozen support table does not contain "
        "2,017 unique parent compounds."
    )

if "chemical_state" not in support.columns:
    raise RuntimeError(
        "chemical_state is missing from "
        "P3_03_chemical_support_by_parent.csv"
    )

if "biological_state" not in support.columns:
    raise RuntimeError(
        "biological_state is missing from "
        "P3_04_biological_support_by_parent.csv"
    )

support["factorial_cell_global"] = (
    support["chemical_state"].astype(str)
    + "/"
    + support["biological_state"].astype(str)
)

factorial_counts = {
    cell: int(
        (
            support["factorial_cell_global"]
            == cell
        ).sum()
    )
    for cell in CELLS
}

expected_counts = {
    "C0/B0": 1219,
    "C1/B0": 574,
    "C0/B1": 135,
    "C1/B1": 89,
}

if factorial_counts != expected_counts:
    raise RuntimeError(
        "Global frozen factorial counts failed reconstruction.\n"
        f"Observed: {factorial_counts}\n"
        f"Expected: {expected_counts}"
    )

print(
    "Global factorial counts reconstructed:",
    factorial_counts,
)

context_column = None

for candidate in [
    "cell_type",
    "context",
    "context_name",
]:

    if candidate in contexts.columns:

        context_column = candidate
        break


if context_column is None:

    raise RuntimeError(
        "Context column not found."
    )


context_counts_raw = (
    contexts[context_column]
    .astype(str)
    .value_counts()
    .to_dict()
)


context_counts = {}

for key, value in (
    context_counts_raw.items()
):

    k = key.strip()

    if k.upper() == "MCF7":
        k = "MCF7"

    elif k.upper() == "U2OS":
        k = "U2OS"

    elif k.upper() in {
        "HEPRG",
        "HEPARG",
    }:
        k = "HepRG"

    context_counts[k] = int(value)


expected_contexts = {
    "MCF7": 1674,
    "U2OS": 1173,
    "HepRG": 1174,
}


if context_counts != expected_contexts:

    raise RuntimeError(
        "Frozen context counts differ.\n"
        f"Observed: {context_counts}"
    )


# ============================================================================
# FIGURE 2 SOURCE DATA
# ============================================================================

prevalence = (
    primary_cells[
        [
            "endpoint",
            "factorial_cell",
            "n",
            "active_n",
            "inactive_n",
            "prevalence",
        ]
    ]
    .drop_duplicates()
    .copy()
)


if prevalence.duplicated(
    [
        "endpoint",
        "factorial_cell",
    ]
).any():

    raise RuntimeError(
        "Duplicate endpoint × cell "
        "prevalence detected."
    )


pooled_prevalence = (
    prevalence
    .groupby(
        "factorial_cell",
        as_index=False,
    )
    .agg(
        active_n=("active_n", "sum"),
        n=("n", "sum"),
    )
)


pooled_prevalence[
    "pooled_prevalence"
] = (
    pooled_prevalence["active_n"]
    /
    pooled_prevalence["n"]
)


joint_effects = primary_re[
    (
        primary_re["analysis_family"]
        == "P9A_FACTORIAL"
    )
    &
    (
        primary_re["contrast"]
        == "joint_vs_reference"
    )
    &
    (
        primary_re["metric"].isin(
            [
                "AUROC",
                "AUPRC",
            ]
        )
    )
].copy()


metric_effects = (
    joint_effects
    .pivot_table(
        index=[
            "model_family",
            "modality",
        ],
        columns="metric",
        values=[
            "random_effect_mean",
            "ci_low",
            "ci_high",
            "q_value",
        ],
        aggfunc="first",
    )
)


metric_effects.columns = [
    f"{a}_{b}"
    for a, b in metric_effects.columns
]


metric_effects = (
    metric_effects
    .reset_index()
)


if len(metric_effects) != 12:

    raise RuntimeError(
        "Expected exactly 12 "
        "model/modality combinations."
    )


optimism = primary_re[
    (
        primary_re["analysis_family"]
        == "P9A_OPTIMISM"
    )
    &
    (
        primary_re["contrast"]
        == "pooled_OOF_minus_joint_OOD"
    )
    &
    (
        primary_re["metric"]
        == "AUROC"
    )
].copy()


if len(optimism) != 12:

    raise RuntimeError(
        "Expected 12 AUROC optimism rows."
    )


# ============================================================================
# SOURCE DATA EXPORT
# ============================================================================

SOURCE_DATA = (
    OUT
    / "source_data"
)

SOURCE_DATA.mkdir(
    parents=True,
    exist_ok=True,
)


pd.DataFrame(
    [
        {
            "factorial_cell": cell,
            "parent_n":
            factorial_counts[cell],
        }
        for cell in CELLS
    ]
).to_csv(
    SOURCE_DATA
    / "Fig1B_factorial_parent_counts.csv",
    index=False,
)


pd.DataFrame(
    [
        {
            "context": context,
            "context_row_n":
            context_counts[context],
        }
        for context in [
            "MCF7",
            "U2OS",
            "HepRG",
        ]
    ]
).to_csv(
    SOURCE_DATA
    / "Fig1A_context_counts.csv",
    index=False,
)


prevalence.to_csv(
    SOURCE_DATA
    / "Fig2A_endpoint_prevalence.csv",
    index=False,
)


pooled_prevalence.to_csv(
    SOURCE_DATA
    / "Fig2A_pooled_prevalence.csv",
    index=False,
)


metric_effects.to_csv(
    SOURCE_DATA
    / "Fig2B_AUROC_AUPRC_joint_effects.csv",
    index=False,
)


optimism.to_csv(
    SOURCE_DATA
    / "Fig2C_AUROC_optimism_random_effects.csv",
    index=False,
)


# ============================================================================
# FIGURE 1
# ============================================================================

fig = plt.figure(
    figsize=(
        FULL_WIDTH_IN,
        7.7,
    )
)


gs = fig.add_gridspec(
    2,
    2,
    left=0.07,
    right=0.98,
    bottom=0.06,
    top=0.98,
    wspace=0.23,
    hspace=0.25,
)


# PANEL A

ax = fig.add_subplot(
    gs[0, 0]
)

panel_label(
    ax,
    "A",
)

ax.set_axis_off()

ax.set_xlim(
    0,
    1,
)

ax.set_ylim(
    0,
    1,
)


flow = [
    (
        0.05,
        0.79,
        "Public HTTr / Tox21 data",
    ),
    (
        0.05,
        0.59,
        "2,017 curated unique\nparent compounds",
    ),
    (
        0.05,
        0.39,
        "4,021 parent–context\nobservations",
    ),
    (
        0.05,
        0.19,
        "41 candidate endpoints\n→ 8 frozen primary endpoints",
    ),
]


for i, (
    x,
    y,
    text,
) in enumerate(flow):

    add_box(
        ax,
        x,
        y,
        0.90,
        0.12,
        text,
    )

    if i < len(flow) - 1:

        add_arrow(
            ax,
            (
                0.50,
                y - 0.01,
            ),
            (
                0.50,
                flow[i + 1][1]
                + 0.13,
            ),
        )


ax.text(
    0.5,
    0.04,
    (
        f"MCF7: "
        f"{context_counts['MCF7']:,}"
        "   |   "
        f"U2OS: "
        f"{context_counts['U2OS']:,}"
        "   |   "
        f"HepRG: "
        f"{context_counts['HepRG']:,}"
    ),
    ha="center",
    fontsize=7,
)


# PANEL B

ax = fig.add_subplot(
    gs[0, 1]
)

panel_label(
    ax,
    "B",
)

ax.set_xlim(
    -0.1,
    2.05,
)

ax.set_ylim(
    -0.1,
    2.05,
)

ax.set_aspect(
    "equal"
)


for x in [0, 1]:

    for y in [0, 1]:

        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                fill=False,
                linewidth=1,
            )
        )


cell_position = {
    "C0/B0": (0.5, 0.5),
    "C1/B0": (1.5, 0.5),
    "C0/B1": (0.5, 1.5),
    "C1/B1": (1.5, 1.5),
}


description = {
    "C0/B0": "Reference",
    "C1/B0": "Chemical novelty",
    "C0/B1": "Biological novelty",
    "C1/B1": "Joint novelty",
}


for cell in CELLS:

    x, y = cell_position[cell]

    ax.text(
        x,
        y + 0.12,
        cell,
        ha="center",
        fontweight="bold",
        fontsize=9,
    )

    ax.text(
        x,
        y - 0.10,
        (
            f"{description[cell]}"
            f"\nN = "
            f"{factorial_counts[cell]:,}"
        ),
        ha="center",
        va="center",
        fontsize=7,
    )


ax.annotate(
    "Increasing chemical novelty",
    xy=(1.95, -0.04),
    xytext=(0.05, -0.04),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.8,
    ),
    ha="center",
    fontsize=7,
)


ax.annotate(
    "Increasing biological novelty",
    xy=(-0.04, 1.95),
    xytext=(-0.04, 0.05),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.8,
    ),
    ha="center",
    fontsize=7,
    rotation=90,
)


ax.set_xticks([])
ax.set_yticks([])


for spine in ax.spines.values():
    spine.set_visible(False)


ax.set_title(
    "Frozen v0.2 factorial support design",
    pad=7,
)


# PANEL C

ax = fig.add_subplot(
    gs[1, 0]
)

panel_label(
    ax,
    "C",
)

ax.set_axis_off()

ax.set_xlim(
    0,
    1,
)

ax.set_ylim(
    0,
    1,
)


ax.text(
    0.5,
    0.94,
    (
        "Elastic Net  |  "
        "Random Forest  |  "
        "HistGradientBoost"
    ),
    ha="center",
    fontsize=7,
)


add_box(
    ax,
    0.04,
    0.76,
    0.38,
    0.11,
    "Molecular structure",
)


add_box(
    ax,
    0.58,
    0.76,
    0.38,
    0.11,
    "HTTr response profile",
)


add_box(
    ax,
    0.04,
    0.53,
    0.38,
    0.11,
    "Structure model",
)


add_box(
    ax,
    0.58,
    0.53,
    0.38,
    0.11,
    "HTTr model",
)


add_arrow(
    ax,
    (0.23, 0.75),
    (0.23, 0.65),
)


add_arrow(
    ax,
    (0.77, 0.75),
    (0.77, 0.65),
)


add_box(
    ax,
    0.31,
    0.30,
    0.38,
    0.11,
    "Early fusion\njoint feature model",
)


add_arrow(
    ax,
    (0.23, 0.52),
    (0.42, 0.42),
)


add_arrow(
    ax,
    (0.77, 0.52),
    (0.58, 0.42),
)


add_box(
    ax,
    0.31,
    0.07,
    0.38,
    0.11,
    "Late fusion\nprobability combination",
)


add_arrow(
    ax,
    (0.23, 0.52),
    (0.42, 0.19),
)


add_arrow(
    ax,
    (0.77, 0.52),
    (0.58, 0.19),
)


# PANEL D

ax = fig.add_subplot(
    gs[1, 1]
)

panel_label(
    ax,
    "D",
)

ax.set_axis_off()

ax.set_xlim(
    0,
    1,
)

ax.set_ylim(
    0,
    1,
)


stages = [
    "Primary grouped factorial OOD benchmark",
    "Continuous support / applicability domain",
    "25 support-definition / severity combinations",
    "Mean vs median Route-B aggregation",
    "Context + missingness sensitivity",
    "Four frozen alternative split sets",
    "Leave-one-context-out biological transport",
]


top = 0.92
gap = 0.126


for i, text in enumerate(
    stages
):

    y = top - i * gap

    add_box(
        ax,
        0.08,
        y - 0.055,
        0.84,
        0.085,
        text,
        fontsize=6.8,
    )

    if i < len(stages) - 1:

        add_arrow(
            ax,
            (
                0.50,
                y - 0.058,
            ),
            (
                0.50,
                y - gap + 0.043,
            ),
        )


ax.text(
    0.5,
    0.01,
    (
        "Prespecified stress tests qualify "
        "claims; they do not replace "
        "the primary analysis"
    ),
    ha="center",
    fontsize=6.2,
)


save_figure(
    fig,
    OUT
    / "Figure1/"
    "Figure1_factorial_validation_framework",
)


# ============================================================================
# FIGURE 2
# ============================================================================

fig = plt.figure(
    figsize=(
        FULL_WIDTH_IN,
        6.7,
    )
)


gs = fig.add_gridspec(
    2,
    2,
    left=0.09,
    right=0.98,
    bottom=0.08,
    top=0.97,
    height_ratios=[
        1,
        1.30,
    ],
    wspace=0.30,
    hspace=0.36,
)


# PANEL A

ax = fig.add_subplot(
    gs[0, 0]
)

panel_label(
    ax,
    "A",
)


rng = np.random.default_rng(
    20260907
)


for index, cell in enumerate(
    CELLS
):

    group = prevalence[
        prevalence[
            "factorial_cell"
        ]
        == cell
    ]

    jitter = rng.uniform(
        -0.10,
        0.10,
        len(group),
    )

    ax.scatter(
        np.full(
            len(group),
            index,
        )
        + jitter,
        group["prevalence"]
        * 100,
        s=18,
        alpha=0.55,
    )


    pooled = (
        pooled_prevalence.loc[
            pooled_prevalence[
                "factorial_cell"
            ]
            == cell,
            "pooled_prevalence",
        ]
        .iloc[0]
        * 100
    )


    ax.scatter(
        [index],
        [pooled],
        marker="D",
        s=60,
        zorder=5,
        label=(
            "Pooled prevalence"
            if index == 0
            else None
        ),
    )


    ax.text(
        index,
        pooled + 2.5,
        f"{pooled:.1f}%",
        ha="center",
        fontsize=7,
    )


ax.set_xticks(
    range(4),
    CELLS,
)

ax.set_xlabel(
    "Factorial support regime"
)

ax.set_ylabel(
    "Active prevalence (%)"
)

ax.legend(
    frameon=False,
    loc="upper left",
)

clean_axes(
    ax
)

ax.set_title(
    "Prevalence shifts across factorial regimes",
    pad=7,
)


# PANEL B

ax = fig.add_subplot(
    gs[0, 1]
)

panel_label(
    ax,
    "B",
)

ax.axhline(
    0,
    linewidth=0.8,
)

ax.axvline(
    0,
    linewidth=0.8,
)


for modality in MODALITIES:

    group = metric_effects[
        metric_effects[
            "modality"
        ]
        == modality
    ]


    ax.scatter(
        group[
            "random_effect_mean_AUROC"
        ],
        group[
            "random_effect_mean_AUPRC"
        ],
        marker=MOD_MARKER[
            modality
        ],
        s=46,
        alpha=0.9,
        label=MOD_LABEL[
            modality
        ],
    )


    for _, row in (
        group.iterrows()
    ):

        ax.annotate(
            FAMILY_SHORT[
                row[
                    "model_family"
                ]
            ],
            (
                row[
                    "random_effect_mean_AUROC"
                ],
                row[
                    "random_effect_mean_AUPRC"
                ],
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )


ax.text(
    0.03,
    0.97,
    (
        "AUPRC improves\n"
        "while AUROC declines"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7,
    fontweight="bold",
)


ax.set_xlabel(
    "ΔAUROC: joint OOD − reference"
)

ax.set_ylabel(
    "ΔAUPRC: joint OOD − reference"
)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower left",
)

clean_axes(
    ax
)

ax.set_title(
    "Metric discordance under joint novelty",
    pad=7,
)


# PANEL C

ax = fig.add_subplot(
    gs[1, :]
)

panel_label(
    ax,
    "C",
)

ax.axvline(
    0,
    linewidth=0.8,
)


optimism[
    "family_order"
] = optimism[
    "model_family"
].map(
    {
        family: i
        for i, family
        in enumerate(FAMILIES)
    }
)


optimism[
    "modality_order"
] = optimism[
    "modality"
].map(
    {
        modality: i
        for i, modality
        in enumerate(MODALITIES)
    }
)


optimism = (
    optimism
    .sort_values(
        [
            "family_order",
            "modality_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


y_positions = (
    np.arange(
        len(optimism)
    )[::-1]
)


for modality in MODALITIES:

    indices = np.where(
        optimism[
            "modality"
        ].to_numpy()
        == modality
    )[0]


    group = optimism.iloc[
        indices
    ]


    yy = y_positions[
        indices
    ]


    estimate = group[
        "random_effect_mean"
    ].to_numpy(
        float
    )


    low = group[
        "ci_low"
    ].to_numpy(
        float
    )


    high = group[
        "ci_high"
    ].to_numpy(
        float
    )


    ax.errorbar(
        estimate,
        yy,
        xerr=np.vstack(
            [
                estimate - low,
                high - estimate,
            ]
        ),
        fmt=MOD_MARKER[
            modality
        ],
        markersize=5,
        capsize=2,
        linestyle="none",
        label=MOD_LABEL[
            modality
        ],
    )


    for (
        x,
        y,
        q,
    ) in zip(
        estimate,
        yy,
        group[
            "q_value"
        ].to_numpy(
            float
        ),
    ):

        if (
            np.isfinite(q)
            and q < 0.05
        ):

            ax.text(
                x,
                y + 0.25,
                "q<0.05",
                ha="center",
                fontsize=5.5,
            )


labels = [
    (
        f"{FAMILY_SHORT[row.model_family]}"
        f" — "
        f"{MOD_LABEL[row.modality]}"
    )
    for row in optimism.itertuples()
]


ax.set_yticks(
    y_positions,
    labels,
)


ax.set_xlabel(
    "Pooled OOF AUROC − joint-OOD AUROC"
)


ax.text(
    0.99,
    0.03,
    (
        "Positive values indicate "
        "optimistic pooled evaluation"
    ),
    transform=ax.transAxes,
    ha="right",
    fontsize=6.2,
)


ax.legend(
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        1.01,
    ),
)


clean_axes(
    ax
)


ax.set_title(
    (
        "Pooled-validation optimism is "
        "directional but not universal"
    ),
    pad=17,
)


save_figure(
    fig,
    OUT
    / "Figure2/"
    "Figure2_prevalence_metric_discordance",
)


# ============================================================================
# STATUS / CHECKSUMS
# ============================================================================

status = {
    "project": "FACTORIAL-OOD",
    "phase": "10B.1",
    "status": "PASS",
    "render_only": True,
    "figures": [
        "Figure 1",
        "Figure 2",
    ],
    "model_retraining": False,
    "prediction_recalculation": False,
    "new_inferential_testing": False,
    "threshold_change": False,
    "support_definition_change": False,
    "primary_chemical_definition":
        PRIMARY_CHEM,
    "primary_biological_definition":
        PRIMARY_BIO,
    "factorial_counts":
        factorial_counts,
    "context_counts":
        context_counts,
    "source_sha256": {
        "P9A_counts":
            sha256(P9A_COUNTS),
        "P9A_cells":
            sha256(P9A_CELLS),
        "P9A_random_effects":
            sha256(P9A_RE),
        "P5B_context_manifest":
            sha256(CONTEXT_MANIFEST),
    },
}


with open(
    OUT
    / "PHASE10B_FIG1_FIG2_STATUS.json",
    "w",
) as fh:

    json.dump(
        status,
        fh,
        indent=2,
    )


manifest = (
    OUT
    / "SHA256SUMS_PHASE10B_FIG1_FIG2.txt"
)


files = sorted(
    path
    for path in OUT.rglob("*")
    if (
        path.is_file()
        and path.name
        not in {
            manifest.name,
            "PHASE10B_FIG1_FIG2_UPLOAD.zip",
        }
    )
)


with open(
    manifest,
    "w",
) as fh:

    for path in files:

        fh.write(
            f"{sha256(path)}  "
            f"{path.relative_to(OUT)}\n"
        )


upload = (
    OUT
    / "PHASE10B_FIG1_FIG2_UPLOAD.zip"
)


with zipfile.ZipFile(
    upload,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as zf:

    for path in (
        files
        + [manifest]
    ):

        zf.write(
            path,
            arcname=str(
                path.relative_to(OUT)
            ),
        )


print()
print("=" * 96)

print(
    "PHASE 10B.1 FIGURE BUILD PASS"
)

print(
    "Figure 1:",
    OUT
    / "Figure1/"
    "Figure1_factorial_validation_framework.pdf",
)

print(
    "Figure 2:",
    OUT
    / "Figure2/"
    "Figure2_prevalence_metric_discordance.pdf",
)

print(
    "Vector outputs: PDF + SVG"
)

print(
    "Review outputs: PNG @ 600 dpi"
)

print(
    "No retraining / no prediction "
    "recalculation / no new inference."
)

print(
    "Upload:",
    upload,
)

print(
    "Upload SHA256:",
    sha256(upload),
)

print("=" * 96)
