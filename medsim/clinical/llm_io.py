from __future__ import annotations

import ast
import json
import logging
import os
import re
import atexit
import threading
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional

from medsim.clinical.prompts import (
    build_json_compact_repair_prompt,
    build_json_object_only_instruction,
    build_json_repair_prompt,
)
from medsim.clinical.schema import ClinicalRuntimeConfig
from medsim.query_model import query_model


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_INPUT_LOG_FILE = PROJECT_ROOT / "output" / "llm_inputs" / "requests.jsonl"
_JSONL_BUFFER: Dict[Path, List[str]] = {}
_JSONL_BUFFER_LOCK = threading.Lock()
_JSONL_BUFFER_FLUSH_EVERY = 10
LLM_REQUEST_TIMEOUT_SECONDS = 1800.0
COMPLETION_AUDIT_KEY = "__medsim_completion_audit__"


class LLMJSONParseError(RuntimeError):
    """Doctor output could not be parsed without generating new content."""


def _looks_like_placeholder_key(value: str) -> bool:
    normalized = (value or "").strip().lower()
    return normalized in {"", "sk-your-dashscope-api-key", "your-api-key", "sk-xxx", "xxx"}


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_balanced_json_objects(text: str) -> List[str]:
    candidates: List[str] = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
        for candidate in _extract_balanced_json_objects(cleaned):
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object.")
    return payload


def _deterministic_json_object(text: str) -> tuple[Dict[str, Any], List[str]]:
    """Conservatively repair JSON syntax without generating new content."""

    cleaned = _strip_fences(text)
    methods: List[str] = []
    translated = cleaned.translate(
        str.maketrans(
            {
                "\u201c": '"',
                "\u201d": '"',
                "\u2018": "'",
                "\u2019": "'",
                "\uff02": '"',
            }
        )
    )
    if translated != cleaned:
        cleaned = translated
        methods.append("normalize_quotes")

    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first >= 0 and last > first and (first != 0 or last != len(cleaned) - 1):
        cleaned = cleaned[first : last + 1]
        methods.append("extract_object")

    no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
    if no_trailing != cleaned:
        cleaned = no_trailing
        methods.append("remove_trailing_commas")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if payload is None:
        # literal_eval handles single quotes and Python booleans but cannot
        # execute code, so this remains a format-only transformation.
        try:
            payload = ast.literal_eval(cleaned)
            methods.append("python_literal_to_json")
        except (ValueError, SyntaxError):
            payload = None

    if payload is None:
        # Quote bare object keys, never values.
        quoted = re.sub(
            r"(?m)(^|[{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:",
            r'\1"\2":',
            cleaned,
        )
        if quoted != cleaned:
            try:
                payload = json.loads(quoted)
                methods.append("quote_bare_keys")
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        if payload is None:
            raise JSONDecodeError("deterministic JSON repair failed", cleaned, 0)
        raise ValueError("LLM response is not a JSON object.")
    return payload, methods or ["deterministic_parse"]


def _truncate_text(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...<truncated>"


def _max_tokens_for_trace(trace_label: str) -> int:
    label = (trace_label or "").strip().lower()
    if label == "doctor":
        # Doctor output can be longer (FINALIZE with diagnosis/evidence/plan/followup).
        return 1200
    if label.startswith("mdt"):
        return 700
    if label == "manager_extraction":
        return 2400
    if label == "manager_learning":
        return 3600
    if label.startswith("manager"):
        return 2400
    return 500


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    should_flush = False
    with _JSONL_BUFFER_LOCK:
        buffer = _JSONL_BUFFER.setdefault(path, [])
        buffer.append(line)
        should_flush = len(buffer) >= _JSONL_BUFFER_FLUSH_EVERY
    if should_flush:
        _flush_jsonl_buffer(path)


def _flush_jsonl_buffer(path: Optional[Path] = None) -> None:
    with _JSONL_BUFFER_LOCK:
        if path is None:
            targets = list(_JSONL_BUFFER.keys())
        else:
            targets = [path]
        pending: Dict[Path, List[str]] = {}
        for item in targets:
            lines = _JSONL_BUFFER.get(item) or []
            if not lines:
                continue
            pending[item] = lines[:]
            _JSONL_BUFFER[item] = []

    for item, lines in pending.items():
        item.parent.mkdir(parents=True, exist_ok=True)
        with open(item, "a", encoding="utf-8") as file:
            file.write("".join(lines))


atexit.register(_flush_jsonl_buffer)


def _record_llm_input(
    *,
    trace_label: str,
    model_name: str,
    attempt_kind: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "trace_label": trace_label or "unknown",
        "model": model_name,
        "attempt_kind": attempt_kind,
        "max_tokens": int(max_tokens),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    if extra_body:
        payload["extra_body"] = dict(extra_body)
    _append_jsonl(LLM_INPUT_LOG_FILE, payload)


class StructuredLLMClient:
    def __init__(self, config: ClinicalRuntimeConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        if not self.config.runtime.enable_llm:
            return False
        env_key = os.environ.get("DASHSCOPE_API_KEY", "")
        return not _looks_like_placeholder_key(env_key or self.config.api_keys.dashscope)

    def status_snapshot(self) -> Dict[str, Any]:
        configured = bool(self.config.runtime.enable_llm)
        env_key = os.environ.get("DASHSCOPE_API_KEY", "")
        key_present = not _looks_like_placeholder_key(env_key or self.config.api_keys.dashscope)
        if not configured:
            status = "disabled_by_config"
        elif not key_present:
            status = "missing_api_key"
        else:
            status = "ready"
        return {
            "status": status,
            "configured": configured,
            "api_key_present": key_present,
        }

    def actionable_feedback(self) -> str:
        snapshot = self.status_snapshot()
        status = snapshot["status"]
        if status == "disabled_by_config":
            return "LLM is disabled by runtime.enable_llm=false."
        if status == "missing_api_key":
            return "LLM is unavailable: DashScope API key is missing or placeholder."
        return "LLM is available."

    def _assert_available(self) -> None:
        snapshot = self.status_snapshot()
        if snapshot["status"] != "ready":
            raise RuntimeError(self.actionable_feedback())

    def complete_json(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        fallback_payload: Dict[str, Any],
        trace_label: str = "",
        request_extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del fallback_payload
        self._assert_available()

        prompt = f"{user_prompt}\n\n{build_json_object_only_instruction()}"
        max_tokens = _max_tokens_for_trace(trace_label)
        request_error: Exception | None = None
        response_text = ""
        for request_attempt in range(2):
            try:
                _record_llm_input(
                    trace_label=trace_label,
                    model_name=model_name,
                    attempt_kind=f"initial_request_{request_attempt + 1}",
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_tokens=max_tokens,
                    extra_body=request_extra_body,
                )
                response_text = query_model(
                    model_name,
                    prompt,
                    system_prompt,
                    tries=1,
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                    max_tokens=max_tokens,
                    extra_body=request_extra_body,
                )
                request_error = None
                break
            except Exception as exc:
                request_error = exc
                if request_attempt == 0:
                    logger.warning(
                        "LLM_REQUEST_RETRY trace=%s model=%s first_error=%s",
                        trace_label or "unknown",
                        model_name,
                        exc,
                    )
        if request_error is not None:
            logger.error(
                "LLM_REQUEST_FAILED trace=%s model=%s error=%s",
                trace_label or "unknown",
                model_name,
                request_error,
            )
            raise RuntimeError(f"LLM request failed for model={model_name}: {request_error}") from request_error

        parse_error: Exception | None = None
        completion_audit: Dict[str, Any] = {
            "initial_parse_valid": True,
            "deterministic_json_repair": False,
            "repair_methods": [],
        }
        try:
            payload = _extract_json_object(response_text)
        except Exception as exc:
            parse_error = exc
            payload = None
            completion_audit["initial_parse_valid"] = False
            normalization = getattr(self.config, "interface_normalization", None)
            normalization_enabled = bool(getattr(normalization, "enabled", False)) and trace_label == "doctor"
            deterministic_enabled = bool(getattr(normalization, "deterministic_json_repair", True))

            if normalization_enabled and deterministic_enabled:
                try:
                    payload, methods = _deterministic_json_object(response_text)
                    completion_audit["deterministic_json_repair"] = True
                    completion_audit["repair_methods"] = methods
                    logger.warning(
                        "LLM_PARSE_RECOVERED_DETERMINISTIC trace=%s model=%s methods=%s first_error=%s",
                        trace_label or "unknown",
                        model_name,
                        methods,
                        parse_error,
                    )
                except Exception as repair_exc:
                    payload = None
                    parse_error = repair_exc
            elif not normalization_enabled:
                # The raw-interface arm returns the provider response directly.
                repair_prompts = [
                    build_json_repair_prompt(user_prompt),
                    build_json_compact_repair_prompt(user_prompt),
                ]
                for idx, retry_prompt in enumerate(repair_prompts, start=1):
                    try:
                        _record_llm_input(
                            trace_label=trace_label,
                            model_name=model_name,
                            attempt_kind=f"parse_repair_{idx}",
                            system_prompt=system_prompt,
                            user_prompt=retry_prompt,
                            max_tokens=max_tokens,
                            extra_body=request_extra_body,
                        )
                        retry_text = query_model(
                            model_name,
                            retry_prompt,
                            system_prompt,
                            tries=1,
                            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                            max_tokens=max_tokens,
                            extra_body=request_extra_body,
                        )
                        payload = _extract_json_object(retry_text)
                        logger.warning(
                            "LLM_PARSE_RECOVERED trace=%s model=%s attempt=%s first_error=%s",
                            trace_label or "unknown",
                            model_name,
                            idx,
                            parse_error,
                        )
                        break
                    except Exception:
                        payload = None

        if payload is None:
            logger.error(
                "LLM_PARSE_FAILED trace=%s model=%s error=%s raw=%s",
                trace_label or "unknown",
                model_name,
                parse_error,
                _truncate_text(response_text),
            )
            normalization = getattr(self.config, "interface_normalization", None)
            if bool(getattr(normalization, "enabled", False)) and trace_label == "doctor":
                raise LLMJSONParseError(
                    f"Doctor JSON remains unparseable after deterministic repair: {parse_error}"
                ) from parse_error
            raise RuntimeError(f"LLM parse failed for model={model_name}: {parse_error}") from parse_error

        logger.debug(
            "LLM_PARSED_JSON trace=%s model=%s keys=%s",
            trace_label or "unknown",
            model_name,
            sorted(payload.keys()),
        )
        if trace_label == "doctor" and bool(
            getattr(getattr(self.config, "interface_normalization", None), "enabled", False)
        ):
            payload[COMPLETION_AUDIT_KEY] = completion_audit
        return payload
