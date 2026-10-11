from __future__ import annotations


class MockMDTConsultant:
    def review(self, known_history: list[str], released_results: list[str]) -> dict:
        evidence = (known_history + released_results)[:3]
        return {
            "advice_action": "ASK" if len(evidence) < 2 else "FINALIZE",
            "consult_opinion": "Mock MDT review: continue only if key evidence is still missing.",
            "supporting_evidence": evidence,
            "confidence": 0.7,
        }

