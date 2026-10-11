from __future__ import annotations

from typing import Any, Dict, List, Optional

from medsim.clinical.schema import ClinicalRuntimeConfig, DialogueLine, ManagerAssessment


def estimate_text_tokens(text: str) -> int:
    words = len((text or "").split())
    return int(words * 1.3) + 1


def dialogue_lines_to_text(lines: List[DialogueLine]) -> str:
    return "\n".join(f"{line.speaker}: {line.text}" for line in lines)


def _truncate_text(text: str, token_limit: int) -> str:
    compact = " ".join((text or "").split())
    if not compact:
        return ""
    approx_limit = max(12, token_limit // 2)
    return " ".join(compact.split()[:approx_limit])


def compress_kb_entries(entries: List[Dict[str, Any]], token_limit: int, kind: str) -> List[str]:
    snippets: List[str] = []
    for entry in entries[:3]:
        experience = str(entry.get("experience", "")).strip()
        if experience:
            snippets.append(_truncate_text(f"{kind}: {experience}", token_limit))
            continue
        base_text = " | ".join(f"{key}={value}" for key, value in entry.items() if value not in (None, "", [], {}))
        snippets.append(_truncate_text(f"{kind}: {base_text}", token_limit))
    return snippets


def compress_rule_cards(rule_cards: List[Dict[str, Any]], token_limit: int) -> List[str]:
    snippets: List[str] = []
    for entry in rule_cards[:3]:
        card_type = str(entry.get("card_type", "")).strip().lower() or "rule"
        rule_text = str(entry.get("rule_text", "")).strip()
        if not rule_text:
            continue
        snippets.append(_truncate_text(f"[{card_type}] {rule_text}", token_limit))
    return snippets


def build_doctor_context(
    config: ClinicalRuntimeConfig,
    dialogue_lines: List[DialogueLine],
    released_results: List[str],
    success_entries: List[Dict[str, Any]],
    correction_entries: List[Dict[str, Any]],
    manager_feedback: Optional[ManagerAssessment],
    mdt_summary: str,
    constraints: List[str],
    rule_cards: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    del manager_feedback
    recent_dialogue = list(dialogue_lines)
    success_snippets = compress_kb_entries(success_entries, config.memory.kb_snippet_token_limit, "success_case")
    correction_snippets = compress_kb_entries(correction_entries, config.memory.kb_snippet_token_limit, "failure_case")
    rule_card_snippets = compress_rule_cards(rule_cards or [], config.memory.kb_snippet_token_limit)

    rendered_prompt = "\n\n".join(
        [
            f"完整对话:\n{dialogue_lines_to_text(recent_dialogue) or '无。'}",
            "已释放检查结果:\n" + ("\n".join(f"- {item}" for item in released_results) or "无。"),
            "最相关规则卡（最多3条）:\n" + ("\n".join(f"- {item}" for item in rule_card_snippets) or "无。"),
            f"MDT 摘要:\n{mdt_summary or '无。'}",
            "硬约束:\n" + ("\n".join(f"- {item}" for item in constraints) or "无。"),
        ]
    )

    return {
        "recent_dialogue_window": recent_dialogue,
        "released_test_results": list(released_results),
        "top3_success": success_snippets,
        "top3_correction": correction_snippets,
        "top_rule_cards": rule_card_snippets,
        "pending_manager_feedback": "",
        "mdt_summary": mdt_summary,
        "constraints": list(constraints),
        "estimated_tokens": estimate_text_tokens(rendered_prompt),
        "rendered_prompt": rendered_prompt,
    }
