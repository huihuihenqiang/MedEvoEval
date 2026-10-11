from __future__ import annotations

from typing import Any, Dict, List

from medsim.clinical.schema import DoctorCaseView, ManagerView, PatientView, dataclass_to_dict


def _join_items(items: List[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return "无"
    return "\n".join(f"- {item}" for item in cleaned)


def _join_indexed_items(mapping: Dict[str, str]) -> str:
    if not mapping:
        return "无"
    return "\n".join(f"- {key}: {value}" for key, value in mapping.items())


def build_json_object_only_instruction() -> str:
    return "Return exactly one JSON object. Do not add markdown fences or explanation."


def build_json_repair_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Your previous output was invalid JSON. "
        "Return exactly one valid JSON object only, no markdown, no explanation."
    )


def build_json_compact_repair_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Your previous output was invalid or truncated JSON. "
        "Regenerate exactly one valid JSON object. "
        "Keep all string fields concise. "
        "Do not use markdown fences or extra text."
    )


def build_patient_prompt(patient_view: PatientView, disclosed_facts: List[str]) -> str:
    del disclosed_facts
    return (
        "你现在扮演门诊仿真中的患者。\n"
        "所有对外可见内容必须使用中文。\n"
        "你只能回答患者自己知道、能感受到、能回忆的信息。\n"
        f"患者资料：{dataclass_to_dict(patient_view.patient_profile)}"
    )


def build_patient_user_prompt(question: str) -> str:
    return (
        f"医生提问：{question}\n"
        '请只返回一个 JSON 对象，且仅包含字段 "utterance"。'
    )


def build_doctor_prompt(
    doctor_view: DoctorCaseView,
    context_prompt: str,
    allow_request_test: bool = True,
    allow_call_mdt: bool = True,
) -> str:
    if allow_request_test and allow_call_mdt:
        action_enum = "ASK|REQUEST_TEST|CALL_MDT|FINALIZE"
    elif allow_request_test and not allow_call_mdt:
        action_enum = "ASK|REQUEST_TEST|FINALIZE"
    elif (not allow_request_test) and allow_call_mdt:
        action_enum = "ASK|CALL_MDT|FINALIZE"
    else:
        action_enum = "ASK|FINALIZE"
    doctor_json_template = (
        '{\n'
        f'  "action": "{action_enum}",\n'
        '  "utterance": "string",\n'
        '  "question": "string",\n'
        '  "test_name": "string",\n'
        '  "rationale": ["string"],\n'
        '  "working_diagnoses": ["string"],\n'
        '  "needs_mdt": false,\n'
        '  "diagnosis": ["string"],\n'
        '  "evidence": ["string"],\n'
        '  "plan": ["string"],\n'
        '  "followup": ["string"]\n'
        "}"
    )
    if allow_request_test and allow_call_mdt:
        action_rules = (
            "允许动作只有 ASK、REQUEST_TEST、CALL_MDT、FINALIZE。\n"
            "REQUEST_TEST 用于申请查体、辅助检查和影像学检查等，且 test_name 必须从可申请项目目录中精确选择。\n"
            "CALL_MDT 表示调用一次顾问医生进行诊断复核，不是多人投票。\n"
            "顾问意见仅供参考，最终决策仍由你作出。\n"
            "只有在 question_count >= 4 或 test_count >= 1 且尚未调用过顾问时，才允许 CALL_MDT。\n"
            "每个病例最多调用一次顾问医生。\n"
        )
    elif allow_request_test and (not allow_call_mdt):
        action_rules = (
            "允许动作只有 ASK、REQUEST_TEST、FINALIZE。\n"
            "当前 MDT 未启用，禁止输出会诊复核动作。\n"
            "REQUEST_TEST 用于申请查体、辅助检查和影像学检查等，且 test_name 必须从可申请项目目录中精确选择。\n"
        )
    elif (not allow_request_test) and allow_call_mdt:
        action_rules = (
            "允许动作只有 ASK、CALL_MDT、FINALIZE。\n"
            "当前已达到检查次数上限，禁止输出 REQUEST_TEST。\n"
            "CALL_MDT 表示调用一次顾问医生进行诊断复核，不是多人投票。\n"
            "顾问意见仅供参考，最终决策仍由你作出。\n"
            "只有在 question_count >= 4 或 test_count >= 1 且尚未调用过顾问时，才允许 CALL_MDT。\n"
            "每个病例最多调用一次顾问医生。\n"
        )
    else:
        action_rules = (
            "允许动作只有 ASK、FINALIZE。\n"
            "当前已达到检查次数上限，禁止输出 REQUEST_TEST。\n"
            "当前 MDT 未启用，禁止输出会诊复核动作。\n"
        )

    return (
        "你现在扮演门诊接诊医生。\n"
        "所有对外可见内容必须使用中文。\n"
        "你只能输出一个 JSON 对象。\n"
        f"{action_rules}"
        "检查名匹配规则是：去首尾空格后大小写不敏感，但不得改写目录项语义。\n"
        "如果动作是 FINALIZE，必须同时给出 diagnosis、evidence、plan、followup。\n"
        "FINALIZE 时 diagnosis 数组只能包含 1 个明确且简短的诊断短语，不要并列多个候选诊断。\n"
        "你只能使用已经释放给医生的信息，不能捏造检查结果。\n"
        "输出字段必须与下方模板一致，不得新增字段：\n"
        f"{doctor_json_template}\n"
        f"可申请项目目录：\n{_join_items(doctor_view.available_test_catalog)}\n"
        f"当前上下文：\n{context_prompt}"
    )


def build_doctor_user_prompt(
    turn_index: int,
    question_count: int,
    test_count: int,
    released_results: List[str],
    tests_requested: List[str],
    allow_request_test: bool = True,
    allow_call_mdt: bool = True,
    allow_call_mdt_now: bool = False,
    strongly_suggest_mdt: bool = False,
    mdt_called: bool = False,
) -> str:
    if allow_request_test:
        rules = [
            "1) action=REQUEST_TEST 时 test_name 必须非空，且必须来自可申请项目目录。",
            "2) action=ASK 时 question 必须非空，且最多包含两个问题。注意不要和已经提出的问题重复（历史对话记录里面），不要提出已经提过的问题！",
            "3) action=FINALIZE 时 diagnosis/evidence/plan/followup 都必须非空数组，且 diagnosis 长度必须为 1。",
            "4) 除 ASK 外，question 置为空字符串。",
            "5) utterance、question、rationale、plan、followup 使用中文。",
        ]
    else:
        rules = [
            "1) 当前检查次数已达上限，不得选择 REQUEST_TEST。",
            "2) action=ASK 时 question 必须非空，且最多包含两个问题。注意不要和已经提出的问题重复（历史对话记录里面），不要提出已经提过的问题！",
            "3) action=FINALIZE 时 diagnosis/evidence/plan/followup 都必须非空数组，且 diagnosis 长度必须为 1。",
            "4) 除 ASK 外，question 置为空字符串。",
            "5) utterance、question、rationale、plan、followup 使用中文。",
        ]
    mdt_hint_lines: List[str] = []
    if allow_call_mdt:
        if mdt_called:
            mdt_hint_lines.append("6) 顾问医生已调用过一次，禁止再次输出 CALL_MDT。")
        elif allow_call_mdt_now:
            mdt_hint_lines.append("6) 你已达到门槛，可以选择 CALL_MDT 进行一次顾问复核。")
        else:
            mdt_hint_lines.append("6) 未达到门槛（question_count >= 4 或 test_count >= 1）前，禁止输出 CALL_MDT。")
    else:
        mdt_hint_lines.append("6) 当前 MDT 未启用，禁止输出会诊复核动作。")
    if strongly_suggest_mdt:
        mdt_hint_lines.append(
            "7) 你已接近资源上限。若当前诊断仍不稳定，建议优先考虑 CALL_MDT 进行一次复核，或在现有证据基础上尽快 FINALIZE。"
        )
    return (
        f"当前医生回合：{turn_index}\n"
        f"已追问次数：{question_count}\n"
        f"已申请项目次数：{test_count}\n"
        f"已获得结果：{released_results}\n"
        f"已申请过的项目：{tests_requested}\n"
        "输出时请遵守：\n"
        + "\n".join(rules + mdt_hint_lines)
    )


def should_allow_call_mdt(question_count: int, test_count: int, mdt_called: bool) -> bool:
    if mdt_called:
        return False
    return int(question_count) >= 4 or int(test_count) >= 1


def should_strongly_suggest_mdt(question_count: int, test_count: int, mdt_called: bool) -> bool:
    if mdt_called:
        return False
    return int(question_count) >= 8 or int(test_count) >= 3


def build_consultant_prompt(context_prompt: str) -> str:
    consultant_json_template = (
        "{\n"
        '  "advice_action": "ASK|FINALIZE",\n'
        '  "utterance": "string",\n'
        '  "consult_opinion": "string",\n'
        '  "working_diagnoses": ["string"],\n'
        '  "key_evidence_for": ["string"],\n'
        '  "key_evidence_against": ["string"],\n'
        '  "missing_points": ["string"],\n'
        '  "recommended_question": ["string"],\n'
        '  "recommended_diagnosis": ["string"],\n'
        '  "confidence": 0.0,\n'
        '  "rationale": ["string"]\n'
        "}"
    )
    return (
        "你现在扮演门诊顾问医生，用于对主治医生当前诊断思路做一次复核。\n"
        "所有对外可见内容必须使用中文。\n"
        "你不能申请检查，不能直接向环境输出最终诊断。\n"
        "你只能输出一个 JSON 对象，字段必须与模板完全一致，不得新增字段。\n"
        "advice_action 只能是 ASK 或 FINALIZE。\n"
        "若 advice_action=ASK：recommended_question 必须非空，且最多 2 个具体问题；recommended_diagnosis 必须为空数组。\n"
        "若 advice_action=FINALIZE：recommended_diagnosis 必须是长度为 1 的数组；recommended_question 必须为空数组。\n"
        "confidence 必须是 0 到 1 的数字。\n"
        "不得捏造未出现的检查结果。\n"
        "输出模板：\n"
        f"{consultant_json_template}\n"
        f"当前上下文：\n{context_prompt}"
    )


def build_consultant_user_prompt(
    turn_index: int,
    question_count: int,
    test_count: int,
    released_results: List[str],
    tests_requested: List[str],
) -> str:
    return (
        f"当前医生回合：{turn_index}\n"
        f"已追问次数：{question_count}\n"
        f"已申请项目次数：{test_count}\n"
        f"已获得结果：{released_results}\n"
        f"已申请过的项目：{tests_requested}\n"
        "请给出复核意见：补充鉴别诊断、支持证据、反对证据、缺失点，并建议继续问还是可终诊。"
    )


def format_consultant_feedback_for_doctor(consultant_json: Dict[str, Any]) -> str:
    advice_action = str(consultant_json.get("advice_action", "")).strip().upper() or "ASK"
    confidence = consultant_json.get("confidence", 0.0)
    working_diagnoses = consultant_json.get("working_diagnoses", []) or []
    evidence_for = consultant_json.get("key_evidence_for", []) or []
    evidence_against = consultant_json.get("key_evidence_against", []) or []
    missing_points = consultant_json.get("missing_points", []) or []
    recommended_question = consultant_json.get("recommended_question", []) or []
    recommended_diagnosis = consultant_json.get("recommended_diagnosis", []) or []
    rationale = consultant_json.get("rationale", []) or []
    consult_opinion = str(consultant_json.get("consult_opinion", "")).strip() or "顾问未提供额外意见。"
    utterance = str(consultant_json.get("utterance", "")).strip() or consult_opinion
    return (
        "以下是顾问医生意见，请仅作为辅助参考，最终决策仍由你作出。\n"
        f"- 顾问建议动作：{advice_action}\n"
        f"- 顾问说明：{utterance}\n"
        f"- 复核结论：{consult_opinion}\n"
        f"- 当前鉴别诊断：{working_diagnoses}\n"
        f"- 支持证据：{evidence_for}\n"
        f"- 反对证据：{evidence_against}\n"
        f"- 缺失点：{missing_points}\n"
        f"- 建议追问：{recommended_question}\n"
        f"- 建议终诊：{recommended_diagnosis}\n"
        f"- 置信度：{confidence}\n"
        f"- 复核理由：{rationale}"
    )


def build_manager_extraction_user_prompt() -> str:
    return (
        "请只返回 JSON，并仅包含 matched_evidence_points、incorrect_evidence_points、"
        "matched_management_points、incorrect_management_points 四个数组字段。"
    )


def build_manager_extraction_prompt(
    manager_view: ManagerView,
    final_diagnosis: List[str],
    final_evidence: List[str],
    final_plan: List[str],
    final_followup: List[str],
    evidence_key_points: Dict[str, str],
    evidence_error_points: Dict[str, str],
    management_key_points: Dict[str, str],
    management_error_points: Dict[str, str],
) -> str:
    diagnosis = manager_view.ground_truth.diagnosis
    return (
        "你现在扮演门诊管理者的评分信息抽取器。\n"
        "你不能直接打分，只能抽取医生输出命中的标准点和错误点。\n"
        "你必须仅输出一个 JSON 对象，并严格遵守以下规则：\n"
        "1) 只允许四个字段：matched_evidence_points、incorrect_evidence_points、matched_management_points、incorrect_management_points。\n"
        "2) 每个字段都必须是数组，且元素必须是预定义编号（如 E1/EE1/M1/MM1）。\n"
        "3) 不得输出任何额外字段、解释、总结或自由文本。\n"
        "4) 如果没有命中，返回空数组。\n"
        f"标准诊断显示名：{diagnosis.display_name}\n"
        f"标准诊断别名：\n{_join_items(manager_view.diagnosis_aliases)}\n"
        f"证据标准点（可选）：\n{_join_indexed_items(evidence_key_points)}\n"
        f"证据错误点（可选）：\n{_join_indexed_items(evidence_error_points)}\n"
        f"处理标准点（可选）：\n{_join_indexed_items(management_key_points)}\n"
        f"处理错误点（可选）：\n{_join_indexed_items(management_error_points)}\n"
        f"医生诊断：\n{_join_items(final_diagnosis)}\n"
        f"医生依据：\n{_join_items(final_evidence)}\n"
        f"医生处理方案：\n{_join_items(final_plan)}\n"
        f"医生随访建议：\n{_join_items(final_followup)}"
    )


def build_manager_learning_user_prompt(is_success: bool) -> str:
    mode_text = "成功经验" if is_success else "失败经验"
    return (
        f"请生成{mode_text}，只返回一个 JSON 对象，且仅包含字段 experience。\n"
        "experience 必须是一句中文长句，且必须包含以下三段并用分号隔开："
        "病例签名：...；决策摘要：...；反思规则：..."
    )


def build_manager_learning_prompt(
    manager_view: ManagerView,
    visit_summary: str,
    final_diagnosis: List[str],
    final_evidence: List[str],
    final_plan: List[str],
    final_followup: List[str],
    matched_evidence_points: List[str],
    matched_management_points: List[str],
    incorrect_evidence_points: List[str],
    incorrect_management_points: List[str],
    diagnosis_grade: str,
    diagnosis_score: float,
    evidence_score: float,
    plan_score: float,
    total_score: float,
    is_success: bool,
) -> str:
    mode_text = "成功经验" if is_success else "失败经验"
    success_rule = (
        "反思规则必须明确：最关键证据是什么、最高效路径是什么、最值得复用的诊断收敛逻辑是什么。"
    )
    failure_rule = (
        "反思规则必须在成功经验要求基础上，额外明确：错在哪里、为什么错、正确改法是什么、"
        "下次遇到什么征象要警惕。"
    )
    return (
        "你现在扮演门诊管理者经验萃取器。\n"
        "你只能输出一个 JSON 对象，且只能包含字段 experience，不得新增字段。\n"
        "所有对外文本必须使用中文。\n"
        "experience 必须是一句中文长句，严格包含三段："
        "病例签名：...；决策摘要：...；反思规则：...\n"
        "病例签名要包含：疾病类型、主诉/核心症状、关键场景，用于快速检索和快速识别病型。\n"
        "决策摘要要包含：关键证据对、最高效路径、诊断收敛动作。\n"
        f"{success_rule if is_success else failure_rule}\n"
        f"当前经验类型：{mode_text}\n"
        f"标准诊断：{manager_view.ground_truth.diagnosis.display_name}\n"
        f"标准诊断别名：\n{_join_items(manager_view.diagnosis_aliases)}\n"
        f"医生最终诊断：\n{_join_items(final_diagnosis)}\n"
        f"医生依据：\n{_join_items(final_evidence)}\n"
        f"医生处理方案：\n{_join_items(final_plan)}\n"
        f"医生随访建议：\n{_join_items(final_followup)}\n"
        f"命中证据点：\n{_join_items(matched_evidence_points)}\n"
        f"命中处理点：\n{_join_items(matched_management_points)}\n"
        f"错误证据点：\n{_join_items(incorrect_evidence_points)}\n"
        f"错误处理点：\n{_join_items(incorrect_management_points)}\n"
        f"评分概况：diagnosis_grade={diagnosis_grade}，diagnosis_score={diagnosis_score:.2f}，"
        f"evidence_score={evidence_score:.2f}，plan_score={plan_score:.2f}，total_score={total_score:.2f}\n"
        f"就诊摘要：\n{visit_summary}"
    )


def build_mdt_expert_prompt(problem: str, specialty: str, evidence: List[str]) -> str:
    return (
        "你现在扮演 MDT 会诊专家。\n"
        "所有对外可见内容必须使用中文。\n"
        "你只能输出一个 JSON 对象。\n"
        f"专科：{specialty}\n"
        f"临床问题：{problem}\n"
        f"当前证据：\n{_join_items(evidence)}"
    )


def build_mdt_user_prompt() -> str:
    return (
        "请返回一个 JSON 对象，字段固定为 "
        "specialty、assessment、supporting_evidence、recommended_next_step、confidence；"
        "assessment 和 recommended_next_step 必须使用中文。"
    )


def build_manager_rule_cards_user_prompt(is_success: bool) -> str:
    if is_success:
        expected = "success 和 decision"
    else:
        expected = "avoid 和 success"
    return (
        "请只返回一个 JSON 对象，且仅包含字段 rule_cards。\n"
        "rule_cards 必须是长度为 2 的数组，每项包含 card_type 与 rule_text。\n"
        f"本次必须输出的 card_type 组合：{expected}。\n"
        "格式要求：\n"
        "- success: rule_text 必须包含 IF 和 THEN。\n"
        "- avoid: rule_text 必须包含 IF 和 AVOID。\n"
        "- decision: rule_text 必须包含 IF 且包含 ACTION=ASK/FINALIZE/CALL_MDT 之一。"
    )


def build_manager_rule_cards_prompt(
    manager_view: ManagerView,
    visit_summary: str,
    final_diagnosis: List[str],
    final_evidence: List[str],
    final_plan: List[str],
    final_followup: List[str],
    matched_evidence_points: List[str],
    matched_management_points: List[str],
    incorrect_evidence_points: List[str],
    incorrect_management_points: List[str],
    diagnosis_grade: str,
    diagnosis_score: float,
    evidence_score: float,
    plan_score: float,
    total_score: float,
    is_success: bool,
) -> str:
    mode_text = "成功病例" if is_success else "失败病例"
    expected_combo = "success + decision" if is_success else "avoid + success"
    return (
        "你现在扮演门诊管理者规则卡生成器。\n"
        "你只能输出一个 JSON 对象，且只能包含字段 rule_cards，不得新增字段。\n"
        "每张卡必须可直接给医生模型参考，简洁、可执行、可迁移。\n"
        "rule_cards 必须恰好 2 张。\n"
        f"当前模式：{mode_text}，必须输出：{expected_combo}。\n"
        "规则卡类型与格式：\n"
        "1) success: IF ... THEN ...\n"
        "2) avoid: IF ... AVOID ...\n"
        "3) decision: IF ... ACTION=ASK/FINALIZE/CALL_MDT\n"
        "如果生成 decision，ACTION 只能是 ASK/FINALIZE/CALL_MDT 之一。\n"
        f"标准诊断：{manager_view.ground_truth.diagnosis.display_name}\n"
        f"标准诊断别名：\n{_join_items(manager_view.diagnosis_aliases)}\n"
        f"医生最终诊断：\n{_join_items(final_diagnosis)}\n"
        f"医生依据：\n{_join_items(final_evidence)}\n"
        f"医生处理方案：\n{_join_items(final_plan)}\n"
        f"医生随访建议：\n{_join_items(final_followup)}\n"
        f"命中证据点：\n{_join_items(matched_evidence_points)}\n"
        f"命中处理点：\n{_join_items(matched_management_points)}\n"
        f"错误证据点：\n{_join_items(incorrect_evidence_points)}\n"
        f"错误处理点：\n{_join_items(incorrect_management_points)}\n"
        f"评分概况：diagnosis_grade={diagnosis_grade}，diagnosis_score={diagnosis_score:.2f}，"
        f"evidence_score={evidence_score:.2f}，plan_score={plan_score:.2f}，total_score={total_score:.2f}\n"
        f"就诊摘要：\n{visit_summary}"
    )
