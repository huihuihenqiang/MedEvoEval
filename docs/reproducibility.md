# Reproduction

Run commands from the code root after installing `requirements.txt`. Python 3.10+ is required. No TeX installation is needed.

| Task | Command | Output |
|---|---|---|
| Tables and plots | `python -B scripts/reproduce_results.py` | `outputs/reproduction/` |
| Tables only | `python -B scripts/reproduce_results.py --skip-figures` | `outputs/reproduction/results/` |
| Engine configuration check | `python -B scripts/run_paper.py --data-dir ../MedEvoEval_data` | Configuration and case count |
| Controlled engine checks | `python -B scripts/validate_release.py --output-dir outputs/check_01` | `outputs/check_01/validation.json` |
| Consultant routing | `python -B scripts/verify_consultant_route.py` | Requested model checks |
| Memory protocol plan | `python -B scripts/run_longitudinal.py` | Stage order, ranges and memory initialization |

Analysis verifies input hashes before reading scalar records. It reproduces paired statistics, Holm correction and vector figures. Supplementary reference tables and aggregate audits are supplied under `results/`; input/coverage descriptions are in `results/README.md`.

Controlled software checks use fixed responses with network access blocked. Live engine runs require environment credentials and `--execute`. Memory-conditioned configurations require a populated `--memory-dir`; the longitudinal command constructs and passes it automatically. Use new run names and output directories to preserve completed runs.

The separate demonstration uses synthetic examples and a simplified scorer. The clinical engine and `results/inputs/` provide the paper workflow. The data package supplies case views and selected process records; paper numerical analyses use the scalar records keyed by experiment and condition.
