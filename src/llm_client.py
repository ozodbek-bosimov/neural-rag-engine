import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

# Automatically load .env if present
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class LLMClient:
    """Multi-provider LLM client supporting Google Gemini and OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if api_key and api_key.startswith("AQ.") else None)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY") or (api_key if api_key and api_key.startswith("sk-") else None)
        self.api_key = self.gemini_key or self.openrouter_key or api_key

        if self.gemini_key:
            self.provider = "Google Gemini"
            self.model = model or os.getenv("DEFAULT_MODEL") or "gemini-flash-lite-latest"
        elif self.openrouter_key:
            self.provider = "OpenRouter"
            self.model = model or "deepseek/deepseek-r1:free"
        else:
            self.provider = "Offline"
            self.model = "Offline Extractive Synthesizer"

    def generate(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        start = time.perf_counter()

        # 1. Google Gemini
        if self.gemini_key:
            try:
                answer = self._call_gemini(prompt, system_prompt)
                latency = round((time.perf_counter() - start) * 1000, 1)
                return {
                    "text": answer,
                    "model": self.model,
                    "provider": "Google Gemini",
                    "latency_ms": latency,
                    "is_fallback": False
                }
            except Exception as err:
                print(f"[!] Gemini primary call failed ({err}). Trying fallback model...")
                try:
                    alt_model = "gemini-flash-latest" if "lite" in self.model else "gemini-flash-lite-latest"
                    answer = self._call_gemini(prompt, system_prompt, model_override=alt_model)
                    latency = round((time.perf_counter() - start) * 1000, 1)
                    return {
                        "text": answer,
                        "model": alt_model,
                        "provider": "Google Gemini",
                        "latency_ms": latency,
                        "is_fallback": False
                    }
                except Exception as err2:
                    print(f"[!] Gemini fallback failed ({err2}). Switching to local extractor.")

        # 2. OpenRouter
        if self.openrouter_key:
            try:
                answer = self._call_openrouter(prompt, system_prompt)
                latency = round((time.perf_counter() - start) * 1000, 1)
                return {
                    "text": answer,
                    "model": self.model,
                    "provider": "OpenRouter",
                    "latency_ms": latency,
                    "is_fallback": False
                }
            except Exception as err:
                print(f"[!] OpenRouter call failed: {err}")

        # 3. Offline Extractive Fallback
        fallback_text = self._extractive_fallback(prompt)
        latency = round((time.perf_counter() - start) * 1000, 1)
        return {
            "text": fallback_text,
            "model": "Offline Extractive Synthesizer",
            "provider": "Local",
            "latency_ms": latency,
            "is_fallback": True
        }

    def _call_gemini(self, prompt: str, system_prompt: str = "", model_override: Optional[str] = None) -> str:
        target_model = model_override or self.model
        if not target_model.startswith("models/"):
            target_model = f"models/{target_model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{target_model}:generateContent?key={self.gemini_key}"
        full_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        payload = {
            "contents": [{"parts": [{"text": full_content}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openrouter(self, prompt: str, system_prompt: str = "") -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ozodbek-bosimov/neural-rag-engine",
            "X-Title": "Neural RAG Engine"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _extractive_fallback(self, prompt: str) -> str:
        context_lines = []
        in_context = False
        question = ""

        for line in prompt.split("\n"):
            line_str = line.strip()
            if "=== RETRIEVED CONTEXT ===" in line_str or "Context:" in line_str:
                in_context = True
                continue
            if "=== USER QUERY ===" in line_str or "User Query:" in line_str:
                in_context = False
                continue
            if in_context:
                if line_str and not line_str.startswith("[Source") and not line_str.startswith("==="):
                    context_lines.append(line_str)
            else:
                if line_str and not line_str.startswith("You are") and not line_str.startswith("Provide a") and not question:
                    question = line_str

        q_words = set(question.lower().split()) if question else set()
        ranked = []
        for line in context_lines:
            words = set(line.lower().split())
            overlap = len(words.intersection(q_words))
            ranked.append((overlap, line))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = [item[1] for item in ranked if item[1]][:5]
        extracted = "\n\n".join(top) if top else "\n\n".join(context_lines[:4])

        return (
            f"**Grounded Extraction:**\n\n"
            f"{extracted}\n\n"
            f"---\n"
            f"*Note: Running in offline mode. Configure an API key in Settings for generative synthesis.*"
        )
