from __future__ import annotations


def _norm(text: str) -> str:
    return str(text or "").strip().lower()


class ExamService:
    """Strict test lookup service used by REQUEST_TEST actions."""

    def __init__(self, exam_view: dict, allowed_catalog: list[str] | None = None) -> None:
        self.tests = dict((exam_view or {}).get("tests") or _flatten_exam_view(exam_view or {}))
        catalog = allowed_catalog or list((exam_view or {}).get("test_catalog") or self.tests.keys())
        self.allowed = {_norm(item): str(item).strip() for item in catalog if str(item).strip()}
        self.case_tests = {_norm(name): (name, value) for name, value in self.tests.items()}

    def request_test(self, test_name: str) -> dict:
        key = _norm(test_name)
        requested = str(test_name or "").strip() or "unspecified test"
        if key not in self.allowed:
            return {
                "test_name": requested,
                "returned": False,
                "result_text": "Test is outside the allowed catalog.",
                "match_type": "not_allowed",
            }
        if key not in self.case_tests:
            return {
                "test_name": self.allowed[key],
                "returned": False,
                "result_text": "No result is available for this case.",
                "match_type": "not_in_case",
            }
        matched_name, result = self.case_tests[key]
        return {
            "test_name": matched_name,
            "returned": True,
            "result_text": str(result),
            "match_type": "exact",
        }


def _flatten_value(value: object) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            text = _flatten_value(item)
            if text:
                parts.append(f"{key}：{text}")
        return "；".join(parts)
    if isinstance(value, list):
        return "；".join(str(item) for item in value if str(item).strip())
    return str(value or "").strip()


def _flatten_exam_view(exam_view: dict) -> dict[str, str]:
    tests: dict[str, str] = {}
    for category, payload in exam_view.items():
        if category in {"tests", "test_catalog"}:
            continue
        category_text = _flatten_value(payload)
        if category_text:
            tests[str(category)] = category_text
        if isinstance(payload, dict):
            for sub_name, sub_payload in payload.items():
                sub_text = _flatten_value(sub_payload)
                if sub_text:
                    tests[str(sub_name)] = sub_text
    return tests
