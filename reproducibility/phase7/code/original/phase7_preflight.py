#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import json, hashlib

ROOT = Path("/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem")
P4 = ROOT / "Phase4"
P6 = ROOT / "Phase6_Output"
P7 = ROOT / "Phase7_Output"
P7.mkdir(exist_ok=True)

EXPECTED_COUNTS = {
    "C0/B0": 1219,
    "C1/B0": 574,
    "C0/B1": 135,
    "C1/B1": 89,
}

LOCK = {
    "project": "FACTORIAL-OOD",
    "phase": 7,
    "protocol": "v0.2",
    "analysis": "Primary Factorial OOD Analysis",
    "created_before_factorial_performance_inspection": True,
    "inferential_unit": "unique curated parent compound",
    "primary_cells": ["C0/B0","C1/B0","C0/B1","C1/B1"],
    "primary_metrics": ["AUPRC","MCC"],
    "reliability_metrics": ["Brier","ECE_10bin"],
    "descriptive_reliability": [
        "calibration_intercept","calibration_slope",
        "AUROC","balanced_accuracy","sensitivity","specificity"
    ],
    "confirmatory_models": {
        "families": [
            "ELASTIC_NET",
            "RANDOM_FOREST",
            "HIST_GRAD_BOOST"
        ],
        "modalities": [
            "STRUCTURE",
            "HTTR",
            "EARLY_FUSION",
            "LATE_FUSION"
        ]
    },
    "controls": "descriptive only",
    "bootstrap": {
        "unit": "parent compound",
        "resamples": 2000,
        "paired_across_models": True,
        "ci": "percentile 95%",
        "minimum_valid_fraction_for_confirmatory_inference": 0.90,
        "seed_namespace": "FACTORIAL-OOD-P7-v1"
    },
    "contrasts": {
        "chemical_effect": "M(C1/B0)-M(C0/B0)",
        "biological_effect": "M(C0/B1)-M(C0/B0)",
        "joint_vs_reference": "M(C1/B1)-M(C0/B0)",
        "joint_vs_chemical": "M(C1/B1)-M(C1/B0)",
        "joint_vs_biological": "M(C1/B1)-M(C0/B1)",
        "factorial_interaction":
            "M(C1/B1)-M(C1/B0)-M(C0/B1)+M(C0/B0)",
        "fusion_robustness":
            "[(Fusion-Structure)C1/B1]-[(Fusion-Structure)C0/B0]",
        "conventional_optimism":
            "M(all pooled OOF)-M(C1/B1)"
    },
    "multiple_testing":
        "Benjamini-Hochberg FDR within each named hypothesis/metric/contrast family",
    "interpretation_rule":
        "No model, endpoint, threshold, cell definition or support cutoff may change after factorial results are inspected."
}

lock_path = P7 / "P7_00_factorial_analysis_lock_v1.0.json"
with open(lock_path, "w") as f:
    json.dump(LOCK, f, indent=2)

h = hashlib.sha256(lock_path.read_bytes()).hexdigest()

print("="*78)
print("FACTORIAL-OOD PHASE 7 PREFLIGHT")
print("="*78)
print("Phase 7 analysis lock SHA256:", h)

# ---------------------------------------------------------
# Verify Phase 6 independent validation
# ---------------------------------------------------------
val = P6 / "P6_INDEPENDENT_VALIDATION_SUMMARY.json"
if val.exists():
    v = json.load(open(val))
    print("\nPhase 6 independent validation:")
    print(json.dumps(v, indent=2))
    if v.get("decision") != "PASS":
        raise RuntimeError("Phase 6 independent validation is not PASS.")
else:
    print("\nWARNING: independent validation summary file not found.")
    original = P6 / "P6_FINAL_STATUS.json"
    if not original.exists():
        raise RuntimeError("No Phase 6 final status found.")
    v = json.load(open(original))
    if v.get("status") != "PASS":
        raise RuntimeError("Phase 6 local status is not PASS.")

# ---------------------------------------------------------
# Inspect P4_09 / P4_10 support files
# ---------------------------------------------------------
files = sorted(list(P4.rglob("P4_09*.csv")) +
               list(P4.rglob("P4_10*.csv")))

print("\nFrozen Phase-4 support-table candidates:")
if not files:
    print("NONE FOUND")
    print("\nAll Phase4 CSVs containing support/factorial keywords:")
    files = sorted([
        p for p in P4.rglob("*.csv")
        if any(k in p.name.lower()
               for k in ["support","factorial","ood","cell"])
    ])

for p in files:
    try:
        df = pd.read_csv(p)
        print("\n" + "-"*78)
        print("FILE:", p)
        print("SHAPE:", df.shape)
        print("COLUMNS:")
        for c in df.columns:
            print("  ", c)

        if "split_set" in df.columns:
            print("split_set values:",
                  df["split_set"].astype(str).value_counts().to_dict())

        # Find likely factorial-cell columns.
        likely = [
            c for c in df.columns
            if "factorial" in c.lower() and "cell" in c.lower()
        ]

        for c in likely:
            print(f"\nValue counts for {c}:")
            print(df[c].astype(str).value_counts(dropna=False).head(20))

        # Show likely C/B-state columns.
        statecols = [
            c for c in df.columns
            if (
                ("chemical" in c.lower() or "biological" in c.lower())
                and any(k in c.lower()
                        for k in ["state","class","support","ood"])
            )
        ]

        for c in statecols:
            print(f"\nCandidate state/support column {c}:")
            print(df[c].astype(str).value_counts(dropna=False).head(15))

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print("ERROR reading", p, type(e).__name__, e)

print("\nExpected frozen primary parent counts from Protocol v0.2:")
print(EXPECTED_COUNTS)
print("TOTAL:", sum(EXPECTED_COUNTS.values()))

print("\nPrimary outer split manifest:")
outer_files = sorted(P4.rglob("P4_01*.csv"))
for p in outer_files:
    df = pd.read_csv(p)
    print(p)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    if "outer_fold" in df.columns:
        print("outer-fold counts:",
              df["outer_fold"].value_counts().sort_index().to_dict())

print("\nNo endpoint performance or factorial model metric was calculated.")
print("PREFLIGHT COMPLETE")
print("="*78)
