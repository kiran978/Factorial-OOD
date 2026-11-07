#!/usr/bin/env python3
"""Check reproduced Phase 7 tables against archived references."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
REF = ROOT / 'reference_outputs' / 'Phase7_Output'
NEW = ROOT / 'Phase7_Output'
TABLES = [
    'P7_02_locked_OOF_with_factorial_cells.csv',
    'P7_03_learned_factorial_cell_metrics.csv',
    'P7_04_control_factorial_cell_metrics.csv',
    'P7_05_confirmatory_factorial_contrasts.csv',
    'P7_06_fusion_robustness.csv',
    'P7_07_optimism_contrasts.csv',
    'P7_08_reliability_contrasts.csv',
    'P7_09_random_effects_endpoint_synthesis.csv',
    'P7_10_factorial_assignment_reconstruction_check.csv',
    'P7_13_primary_endpoint_pooled_factorial_support.csv',
]
failed = False
for name in TABLES:
    old, new = REF / name, NEW / name
    if not old.is_file() or not new.is_file():
        print(f'MISSING: {name}'); failed = True; continue
    a, b = pd.read_csv(old, low_memory=False), pd.read_csv(new, low_memory=False)
    if len(a) != len(b) or list(a.columns) != list(b.columns):
        print(f'SCHEMA_OR_ROWS_DIFFER: {name}'); failed = True; continue
    mismatch = 0
    max_delta = 0.0
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            with np.errstate(invalid='ignore', over='ignore'):
                same = np.isclose(xv, yv, atol=1e-12, rtol=1e-9, equal_nan=True)
                ok = np.isfinite(xv) & np.isfinite(yv)
                if ok.any(): max_delta = max(max_delta, float(np.max(np.abs(xv[ok]-yv[ok]))))
        else:
            same = (x.astype('string').fillna('<NA>').to_numpy() == y.astype('string').fillna('<NA>').to_numpy())
        mismatch += int(np.count_nonzero(~same))
    print(f'{"MATCH" if mismatch == 0 else "DIFFER"}: {name} | mismatched_cells={mismatch} max_numeric_difference={max_delta:.3g}')
    failed |= mismatch != 0
raise SystemExit(1 if failed else 0)
