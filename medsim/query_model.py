# === 中文维护说明 ===
# 文件作用：该文件属于当前主链路源码的一部分，负责本路径下的具体实现或辅助逻辑。
# 主链路位置：位于当前主链路中，用于支撑仿真、前端展示或 Prompt 调用。
# 核心入口：_load_api_settings_from_config, _get_api_settings, _resolve_dashscope_api_key, _resolve_dashscope_base_url, _normalize_model_name, _use_system_proxy, _post_chat_completion, _build_messages, _dashscope_chat_completion, query_model
# 维护标签：主入口

import logging
import os
import re
import time
import atexit
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config_clinic.yaml"


MODEL_ALIASES = {
    "deepseek": DEFAULT_MODEL,
    "deepseek-v3.2": DEFAULT_MODEL,
    "qwen_plus": "qwen-plus",
    "qwen_max": "qwen-max",
    "qwen_turbo": "qwen-turbo",
    "qwen-plus": "qwen-plus",
    "qwen-max": "qwen-max",
    "qwen-turbo": "qwen-turbo",
    "qwue": "qwen-plus",
    "qwue7": "qwen-plus",
}

_API_SETTINGS_CACHE: Optional[Dict[str, str]] = None
_SESSION_WITH_PROXY: Optional[requests.Session] = None
_SESSION_NO_PROXY: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


# 中文说明：_load_api_settings_from_config 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _load_api_settings_from_config() -> Dict[str, str]:
    config_path_str = os.environ.get("MEDSIM_CONFIG_PATH", "").strip()
    config_path = Path(config_path_str) if config_path_str else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_keys = config.get("api_keys", {}) or {}
    api_base = config.get("api_base", {}) or {}
    network = config.get("network", {}) or {}
    return {
        "dashscope_api_key": (api_keys.get("dashscope") or "").strip(),
        "dashscope_base_url": (api_base.get("dashscope") or "").strip(),
        "use_system_proxy": str(network.get("use_system_proxy", False)).strip().lower(),
    }


# 中文说明：_get_api_settings 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _get_api_settings() -> Dict[str, str]:
    global _API_SETTINGS_CACHE
    if _API_SETTINGS_CACHE is None:
        _API_SETTINGS_CACHE = _load_api_settings_from_config()
    return _API_SETTINGS_CACHE


# 中文说明：_resolve_dashscope_api_key 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _resolve_dashscope_api_key() -> str:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        api_key = _get_api_settings().get("dashscope_api_key", "")
    if not api_key:
        raise RuntimeError(
            "Missing DashScope API key. Set DASHSCOPE_API_KEY or config api_keys.dashscope in medsim/configs/config_clinic.yaml."
        )
    return api_key


# 中文说明：_resolve_dashscope_base_url 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _resolve_dashscope_base_url() -> str:
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip()
    if not base_url:
        base_url = _get_api_settings().get("dashscope_base_url", "")
    if not base_url:
        base_url = DEFAULT_BASE_URL
    return base_url.rstrip("/")


# 中文说明：_normalize_model_name 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _normalize_model_name(model_str: Optional[str]) -> str:
    if not model_str:
        return DEFAULT_MODEL
    model = model_str.strip()
    if not model:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(model.lower(), model)


# 中文说明：_use_system_proxy 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _use_system_proxy() -> bool:
    env_value = os.environ.get("MEDSIM_USE_SYSTEM_PROXY", "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False
    return _get_api_settings().get("use_system_proxy", "false") in {"1", "true", "yes", "on"}


def _get_http_session(use_proxy: bool) -> requests.Session:
    global _SESSION_WITH_PROXY, _SESSION_NO_PROXY
    with _SESSION_LOCK:
        if use_proxy:
            if _SESSION_WITH_PROXY is None:
                _SESSION_WITH_PROXY = requests.Session()
                _SESSION_WITH_PROXY.trust_env = True
            return _SESSION_WITH_PROXY
        if _SESSION_NO_PROXY is None:
            _SESSION_NO_PROXY = requests.Session()
            _SESSION_NO_PROXY.trust_env = False
        return _SESSION_NO_PROXY


def _close_http_sessions() -> None:
    global _SESSION_WITH_PROXY, _SESSION_NO_PROXY
    with _SESSION_LOCK:
        if _SESSION_WITH_PROXY is not None:
            _SESSION_WITH_PROXY.close()
            _SESSION_WITH_PROXY = None
        if _SESSION_NO_PROXY is not None:
            _SESSION_NO_PROXY.close()
            _SESSION_NO_PROXY = None


atexit.register(_close_http_sessions)


# 中文说明：_post_chat_completion 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _post_chat_completion(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float):
    session = _get_http_session(_use_system_proxy())
    return session.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )


# 中文说明：_build_messages 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _build_messages(
    prompt: str,
    system_prompt: str,
    image_requested: bool,
    scene: Optional[Any],
) -> List[Dict[str, Any]]:
    if not image_requested:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    if scene is None or not hasattr(scene, "image_url"):
        raise ValueError("image_requested=True but scene.image_url is missing.")

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": str(scene.image_url)}},
            ],
        },
    ]


# 中文说明：_dashscope_chat_completion 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def _dashscope_chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    timeout: float,
    max_tokens: int = 400,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    api_key = _resolve_dashscope_api_key()
    base_url = _resolve_dashscope_base_url()

    payload: Dict[str, Any] = {
        "model": _normalize_model_name(model),
        "messages": messages,
        "temperature": 0.05,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        payload["extra_body"] = dict(extra_body)

    response = _post_chat_completion(
        url=f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


# 中文说明：query_model 是当前文件中的一个独立步骤或辅助步骤，需要结合调用方一起理解。
def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    tries: int = 1,
    timeout: float = 30.0,
    image_requested: bool = False,
    scene: Optional[Any] = None,
    max_prompt_len: int = 2 ** 14,
    clip_prompt: bool = False,
    max_tokens: int = 400,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Query LLM via DashScope-compatible chat completion endpoint only."""

    last_error: Optional[Exception] = None
    for _ in range(tries):
        try:
            if clip_prompt:
                prompt = prompt[:max_prompt_len]

            messages = _build_messages(prompt, system_prompt, image_requested, scene)
            answer = _dashscope_chat_completion(
                model=model_str or DEFAULT_MODEL,
                messages=messages,
                timeout=timeout,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return re.sub(r"\s+", " ", answer).strip()
        except Exception as exc:
            last_error = exc
            logger.warning("query_model failed for model=%s: %s", model_str, exc)
            time.sleep(min(timeout, 3.0))

    if last_error:
        raise RuntimeError(f"Model query failed after retries: {last_error}")
    raise RuntimeError("Model query failed after retries.")
