# Figure 2 v2 â€” reproducibility supplement

The recovered `phase10b_fig1_fig2_v2.py` script is the original Phase 10B.2 publication-refinement plotting code, with only its machine-specific `ROOT` path replaced by the release directory. It builds Figure 1 v2 and Figure 2 v2 from frozen Phase 3, Phase 5B and Phase 9A tables. These frozen inputs have been included. This render-only script does not re-fit models or recalculate primary factorial contrasts.

To render from a fresh extracted checkout, install the Phase 7 requirements and a compatible Matplotlib (for example `matplotlib==3.10.7` on Python 3.10) and run:

```powershell
python -m pip install -r requirements-phase7.txt matplotlib==3.10.7
python -u .\phase10b_fig1_fig2_v2.py
```

Outputs should appear under `Phase10B_Output_v2/Figure2/`, including `Figure2_prevalence_metric_discordance_v2.pdf`, `.svg` and `.png`. The source code checks the Phase 9A `PASS` marker, 2,017 factorial parent assignments and expected biological-context counts. The source is recovered and its inputs are supplied, but this release has not yet established identity to the exact figure artwork used in every manuscript revision, nor pixel-level output equality across rendering environments.