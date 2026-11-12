import os
import json
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

# Recommended high-performing free models on OpenRouter
DEFAULT_FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "liquid/lfm-2.5-2.6b:free",
    "cohere/north-mini-code:free",
    "google/gemma-4-31b-it:free"
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient:
    """
    Client for OpenRouter API to interact with top-tier free reasoning models.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key and os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                        elif line.startswith("OPENROUTER_MODEL=") and not model:
                            model = line.split("=", 1)[1].strip()

        self.api_key = api_key
        self.model = model or os.getenv("OPENROUTER_MODEL") or "deepseek/deepseek-r1:free"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenRouter API.
        """
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please set the environment variable "
                "or pass api_key to OpenRouterClient.\n"
                "Get a free API key at https://openrouter.ai/keys"
            )

        selected_model = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/clash-iq",
            "X-Title": "Clash IQ AI Coach"
        }

        models_to_try = [selected_model]
        for m in DEFAULT_FREE_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        for current_model in models_to_try:
            payload: Dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if response_format:
                payload["response_format"] = response_format

            req = urllib.request.Request(
                OPENROUTER_API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8")
                last_error = f"Model {current_model} failed (HTTP {e.code}): {error_body}"
                continue
            except Exception as e:
                last_error = f"Model {current_model} failed: {str(e)}"
                continue

        raise RuntimeError(f"All free OpenRouter models failed. Last error: {last_error}")
