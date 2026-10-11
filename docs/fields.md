# File and field guide

## Data package interface

The engine consumes <data-dir>/chinese_medevoeval_medqa700.jsonl. It requires case_id, task, patient_view, exam_view and manager_view. Public release IDs are case_001..case_700; the engine's scenario_id is the zero-based row index. case_scenario_map.csv maps these IDs and preserves content hashes. The data package owns its corpus schemas, source attribution and process-record descriptions.

## Real-engine output

A live batch writes runs/<run-id>/cases.jsonl (full episodes), case_summaries.jsonl (per-case summaries), batch_summary.json and summary.json beneath --output-dir. Full episodes contain episode_id, scenario_id, run_id, case, config_snapshot, full_dialogue, event_log, metrics, doctor_decisions, manager_feedbacks, mdt_called and consultant_opinion. Metrics include num_turns, num_tests, invalid_test_rate, diagnosis_score, evidence_score, plan_score, total_score and utility. The real episode structure is described in schemas/real_engine_episode_schema.json; nested dataclasses are defined in medsim/clinical/schema.py.

The original schemas/episode_schema.json describes the lightweight demo's episode_id/case_id/events/scores contract. It does not describe the real-engine full record. schemas/process_trajectory_schema.json describes sanitized process data. These distinct formats must not be silently interchanged.

Raw request logs are under llm_inputs/requests.jsonl and include trace_label, requested model, system/user prompt, attempt kind and timestamp. These are local execution traces, not release-ready sanitized records. model identifies the requested provider model, not an independent server-weight attestation.

## Frozen scalar inputs and tables

results/inputs/*.json are arrays of scalar records or documented cached summaries. source_manifest.json records source filenames/hashes. frozen_input_hashes.json protects the exact input bytes. Common row keys include condition, scenario_id, diagnosis_score, evidence_score, plan_score, total_score, turns, tests, invalid_test_rate and utility; exact keys depend on the source experiment. Physician audit records instead use audit_id plus condition; episode_id alone is not unique.

condition_summary.csv contains section, raw condition, n and metric means. Recomputed rows add condition_display. final_holm_tests.csv contains family_id, comparison, metric, paired sample size, method, mean/delta/interval when available, raw_p and Holm-adjusted p. Recomputed rows add comparison_display. A cached stage diagnostic is explicitly marked and is not an invented case-level inference.

R1 outputs record 80 audited model outputs and descriptive 24 direction comparisons. Held-out tables contain 200 paired IDs and seven endpoints. Weight tables enumerate 66 outcome-weight triples; utility always uses turn divisor 10. Strong-primary MDT uses all 100 pairs rather than only its 85 triggered cases. See results/README.md for estimator details and caveats.

All utility fields use turn divisor 10. Paper input identities and hashes are recorded in results/inputs/source_manifest.json.
