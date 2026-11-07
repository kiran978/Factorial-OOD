# Factorial OOD — Reviewer Comment 3 reproducibility release

This release contains the recovered original Phase 1–5 scripts/packages, primary Phase 7
analysis script, frozen Phase 4/6 inputs, Phase 7 analysis lock, reference outputs and
numerical-comparison verification. The verified primary analysis was reproduced in an
isolated Windows Python 3.10 environment using 2,000 parent-cluster bootstrap resamples
per endpoint for eight endpoints; the Phase 7 script reported 14/14 internal tests PASS.

## Independent Phase 7 reproduction

Run from the **root of an extracted copy** of this release using Python 3.10:

```powershell
python -m pip install -r requirements-phase7.txt
$env:P7_JOBS = "2"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -u .\phase7_factorial_analysis.py
python .\verify_phase7_results.py
```

**Important:** The primary analysis writes to `Phase7_Output` inside this copy. Do not
run it in an archival/original-data folder; preserve a pristine copy of this release.
The portable runner makes only one change to the original source: `ROOT` resolves to
the release directory instead of a hard-coded Linux home directory. The unmodified
original is at `code/original/phase7_factorial_analysis.original.py`.

## Data and source layout

* `Phase3/`, `Phase4/`: copied CSV data used to reconstruct factorial assignments.
* `Phase6_Output/`: independent validation summary and locked out-of-fold predictions.
* `Phase7_Output/P7_00_factorial_analysis_lock_v1.0.json`: prespecified lock.
* `reference_outputs/Phase7_Output/`: untouched frozen Phase 7 output CSVs.
* `upstream_packages/`: original archived Phase 1–5 project packages.
* `upstream_code/`: extracted Python/R/notebook scripts from those upstream packages.
* `code/original/`: original later-phase scripts, including Figure 2 source candidates.
* `verify_phase7_results.py`: independent comparison of regenerated CSVs against
  frozen reference outputs using numeric tolerances of rtol=1e-9 and atol=1e-12.

## Scope and limitations

The primary Phase 7 statistical outputs were independently reproduced from the frozen
Phase 4 and Phase 6 inputs. This is **not** a claim that the complete pipeline was
re-executed from the raw HTTr feather matrix. The upstream Phase 1–5 scripts are
preserved for inspection, but have not yet undergone end-to-end re-execution, and some
scripts may contain original machine-specific absolute paths. Figure-generation
scripts have been recovered but their exact final-version identity and pixel/figure
reproduction must be validated separately before claiming Figure 2 itself was
reproduced. Two environment-sensitive metadata tables may differ in text fields;
consult the local verification evidence for exact differences.

Repository: https://github.com/kiran978/Factorial-OOD
Upstream data DOI (as supplied by the study author): https://doi.org/10.5281/zenodo.24025829
Earlier software archive DOI (as supplied by the study author): https://doi.org/10.5281/zenodo.22647791
These links should be checked when publishing this package.
