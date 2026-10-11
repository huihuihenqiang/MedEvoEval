from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat-completions client using stdlib only."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 90,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("MODEL_NAME")
        self.timeout_seconds = int(timeout_seconds)
        self.temperature = float(temperature)
        if not self.api_key:
            raise RuntimeError("Missing LLM_API_KEY. Set it in the environment or a local .env file.")
        if not self.base_url:
            raise RuntimeError("Missing LLM_BASE_URL. Example: https://api.openai.com/v1")
        if not self.model:
            raise RuntimeError("Missing MODEL_NAME. Example: gpt-4.1 or qwen-turbo")

    def chat_json(self, messages: list[dict[str, str]], max_tokens: int = 700) -> dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail[:500]}") from exc
        content = response_payload["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        try:
            return json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON: {str(content)[:500]}") from exc
