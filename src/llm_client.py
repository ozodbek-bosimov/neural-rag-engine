import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


class LLMClient:
    """OpenRouter API client with automatic offline extractive synthesis fallback."""

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek/deepseek-r1:free"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        start = time.perf_counter()

        if self.api_key and ("sk-" in self.api_key):
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
                print(f"[!] OpenRouter error: {err}. Switching to offline extractor.")

        fallback_text = self._extractive_fallback(prompt)
        latency = round((time.perf_counter() - start) * 1000, 1)
        return {
            "text": fallback_text,
            "model": "Offline Extractive Synthesizer",
            "provider": "Local",
            "latency_ms": latency,
            "is_fallback": True
        }

    def _call_openrouter(self, prompt: str, system_prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ozodbek-bosimov/pytorch-rag-engine",
            "X-Title": "PyTorch RAG Engine"
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
            f"*Note: Running in offline grounded mode. Configure an OpenRouter API key in Settings for generative synthesis.*"
        )
