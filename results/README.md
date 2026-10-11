# Paper results

Run `python -B scripts/reproduce_results.py` from the code root. Inputs are in `inputs/`; recomputed tables and plots are written under `outputs/reproduction/`. `frozen_input_hashes.json` verifies the input files. The supplied tables are the paper's reference results.

## Analysis map

| Analysis | Input | Results |
|---|---|---|
| Four-model comparison | `dynamic_models.json` | `condition_summary.csv`, `final_holm_tests.csv` |
| Qwen-primary MDT | `mdt.json`, `mdt_behavior.json` | `condition_summary.csv`, `final_holm_tests.csv` |
| E0–E3 memory | `stream_memory.json` | `condition_summary.csv` |
| 200-pair held-out | `heldout_n200.json` | `heldout_n200_paired.csv`, `heldout_n200_statistics.csv` |
| Memory ablations | `ablation.json`, `placebo_*.json` | `condition_summary.csv`, `final_holm_tests.csv` |
| Strong-primary MDT | `strong_*.json` | `condition_summary.csv`, `final_holm_tests.csv` |
| Stage response/retention | `stage_metrics.json`, `bwt.json` | `stage_utility.csv`, `update_phase_utility.csv`, `final_holm_tests.csv` |
| Interface intervention | `interface_ab.json` | `final_holm_tests.csv` |
| Physician review | `physician_review_summary.json` | `r1_physician_*`, `r1_direction_contrasts.csv` |
| Scorer/matching | `rejudge_*.json`, `judge_free.json` | `judge_reliability.*`, `judge_free_summary.csv`, `heldout_rejudge_utility_*` |
| Weight sensitivity | Model, stream and held-out inputs | `weight_sensitivity_*`, `utility_sensitivity.csv` |
| Resources | `resource_summary.json` | `resource_summary.csv`, `resource_summary.json` |

Input paths in this table are relative to `inputs/`. Tables with supplementary aggregate estimates are supplied directly; the main reproduction command reconstructs the paired, scorer and weight analyses, renders published physician aggregates, and generates ten figures.

## Scoring and statistics

Utility is `0.6*S - 0.2*(turns/10) - 0.1*(tests/3) - 0.1*invalid_test_rate`. All input and output utility fields use this definition. Five utility-weight settings vary cost weights; the 66-point grid varies diagnosis/evidence/plan weights.

The held-out comparison pairs 200 scenario IDs, 500–699, across seven endpoints. Tests are two-sided paired Wilcoxon or exact McNemar, with Holm adjustment within the seven-endpoint family. Mean-difference intervals use 100,000 paired bootstrap samples, seeds 507–513. Other paired comparisons use 20,000 samples, seed 7. The complete analysis contains 18 Holm families and 141 tests. Stage trends use ten stage means.

Physician scoring covers the same 80 outputs for both reviewers, joined by `audit_id` and condition. Binary kappa groups exact/mostly-correct diagnoses. The 24 direction comparisons compare group means on diagnosis, evidence and plan; 21 agree across both physicians and automatic scoring.

The 50-case transformation audit is supplied as the confirmed aggregate. Alternative judges and diagnosis matching cover 2,520 outputs: four models at 80 each, four stream stages at 500 each, and a 100-pair held-out subset. The repeated scorer covers 520 outputs. Qwen-Max uses deterministic rubric matching for 107 extraction failures. The strong-primary analysis includes all 100 pairs, with MDT invoked in 85 episodes.

Resource quantities are marked measured, estimated or partial-segment coverage. The project estimate includes development runs. Memory-card summaries and published audit aggregates are included as supporting inputs. Record IDs and pairing remain as defined in the paper.
