# Validation

Offline verification covers 19 engine modules, 14 configurations, consultant dispatch and five controlled engine/adapter episodes. Configuration checks verify the main and extended-manager token caps. No model API calls are made.

The statistical reproduction covers 13 numerical CSV tables and eight vector figures. The 12,258 compared numerical table cells match the paper reference values within 1e-12. The analysis contains 18 Holm families and 141 tests.

Run `scripts/verify_package.py`, `scripts/verify_configuration.py`, `scripts/verify_consultant_route.py` and `scripts/validate_release.py` as documented in README.md. Results are recorded in `validation_report.json`.
