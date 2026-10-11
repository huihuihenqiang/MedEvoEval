# Source and analysis mapping

The clinical engine is derived from MedAgentSim and retains its CC BY-NC-SA 4.0 license. `engine_manifest.json` identifies the included modules. Prompts, actions, scoring and memory are implemented in `medsim/clinical/`.

`results/inputs/source_manifest.json` identifies every published statistical input by relative path and hash. MDT inputs contain the three Qwen-primary arms, including the Kimi consultant, with a separate GLM-5 primary comparison. Input values use the scoring definitions in `results/README.md`.

The data package preserves MedQA source identifiers, case mappings and trajectory linkage. Clinical scores and actions are retained; published utility is computed with the configured action budget.
