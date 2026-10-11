# Experiment settings

The primary doctor uses `models.attending`, the consultant uses `models.mdt_expert`, and the manager uses `models.manager`.

| Configuration | Primary | Consultant | Manager | Token profile |
|---|---|---|---|---|
| `benchmark_*.yaml` | Named model | Disabled | Kimi-K2.5 | `main` |
| `mdt_qwen.yaml` | Qwen-Turbo | Qwen-Turbo | Kimi-K2.5 | `main` |
| `mdt_kimi.yaml` | Qwen-Turbo | Kimi-K2.5 | Kimi-K2.5 | `main` |
| `strong_primary_nomdt.yaml` | GLM-5 | Disabled | Kimi-K2.5 | `extended_manager` |
| `strong_primary_mdt.yaml` | GLM-5 | GLM-5 | Kimi-K2.5 | `extended_manager` |
| `e0.yaml`–`e3.yaml`, `h0.yaml`, `h3.yaml` | Qwen-Turbo | Disabled | Kimi-K2.5 | `main` |

The `release.token_profile` field selects caps automatically. Doctor/patient/single-consultant caps are 1,200/500/500 in both profiles. Main manager caps are extraction 1,200, learning 1,800 and other manager requests 1,200. Extended-manager caps are extraction 2,400, learning 3,600 and other manager requests 2,400. The explicit `--token-profile` option overrides the configuration.

The Qwen-primary study contains 100 assigned episodes per arm; the Qwen and Kimi consultants are invoked in 98 and 99 episodes. The strong-primary study contains 100 paired episodes, including all assigned episodes. Results use the same condition keys throughout; `results/display_labels.json` maps keys to display names.
