# Factorial-ODD
This repository contains the frozen computational outputs, source code, provenance records, validation manifests, figures, tables, and reproducibility metadata supporting the manuscript “Factorial chemical-biological validation reveals hidden optimism in multimodal QSAR under joint distribution shift.

## Independently verified primary Phase 7 reproducibility

The reproducibility materials are provided under [`reproducibility/phase7/`](reproducibility/phase7/README_REPRODUCIBILITY.md). They include recovered Phase 1-5 scripts, the **original** and **portable** primary Phase 7 analysis code, frozen Phase 3/4/6 inputs, a prespecified Phase 7 analysis lock, reference tables, software dependencies, and an independent results-comparison script.

Using frozen Phase 4 and Phase 6 inputs, the primary Phase 7 analysis was independently re-executed in an isolated Python 3.10 environment. It completed 2,000 parent-cluster bootstrap resamples for each of eight endpoints, reproduced 160/160 factorial support records, passed 14/14 built-in integrity checks, and reproduced the principal scientific result tables (`P7_03`, `P7_05`, `P7_09`) within `rtol=1e-9`, `atol=1e-12`.

**Scope limitation:** The primary Phase 7 statistical analysis has been independently verified, but complete end-to-end regeneration of Phases 1-6 from the raw HTTr feather dataset, and final visual reproduction of manuscript Figure 2, have not been established. A portable Figure 2 v2 rendering script and its additional Phase 9/Phase 5B inputs are now included, but final-artwork identity and pixel equivalence remain to be verified. Differences in Phase 7 metadata text (`assignment_source` provenance strings and Unicode encoding of test descriptions) do not change the reproduced numerical contrasts. See `reproducibility/phase7/REPRODUCTION_AUDIT.json`.
