from __future__ import annotations


class PatientSimulator:
    """Toy patient simulator that reveals one hidden fact per ASK action."""

    def __init__(self, patient_view: dict) -> None:
        self.patient_view = dict(patient_view or {})
        self.facts = list(self.patient_view.get("history_facts") or _derive_history_facts(self.patient_view))

    def respond(self, question: str, disclosed_facts: list[str]) -> dict:
        hidden = [fact for fact in self.facts if fact not in disclosed_facts]
        revealed = hidden[:1]
        if revealed:
            utterance = revealed[0]
        else:
            utterance = self.patient_view.get("chief_complaint") or "No additional symptoms are available."
        return {
            "speaker": "patient",
            "text": utterance,
            "revealed_facts": revealed,
            "question": question,
        }


def _derive_history_facts(patient_view: dict) -> list[str]:
    facts: list[str] = []
    demographics = patient_view.get("demographics") or {}
    if demographics.get("age") not in (None, ""):
        facts.append(f"年龄：{demographics.get('age')}岁")
    if demographics.get("sex"):
        facts.append(f"性别：{demographics.get('sex')}")
    if patient_view.get("chief_complaint"):
        facts.append(f"主诉：{patient_view.get('chief_complaint')}")
    hpi = patient_view.get("history_of_present_illness") or {}
    if hpi.get("duration"):
        facts.append(f"病程：{hpi.get('duration')}")
    for symptom in hpi.get("symptoms") or []:
        facts.append(f"症状：{symptom}")
    for feature in hpi.get("progression_features") or []:
        facts.append(f"进展特点：{feature}")
    if patient_view.get("past_medical_history"):
        facts.append(f"既往史：{patient_view.get('past_medical_history')}")
    social = patient_view.get("social_history") or {}
    for key, value in social.items():
        if value:
            facts.append(f"{key}：{value}")
    ros = patient_view.get("review_of_systems") or {}
    for item in ros.get("positive_findings") or []:
        facts.append(f"伴随阳性：{item}")
    for item in ros.get("negative_findings") or []:
        facts.append(f"否认：{item}")
    return facts
