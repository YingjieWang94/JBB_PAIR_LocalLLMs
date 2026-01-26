import os
import requests

base_url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
model = os.environ.get("VLLM_MODEL", "lmsys/vicuna-13b-v1.5")

resp = requests.post(
    f"{base_url}/v1/chat/completions",
    json={
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "temperature": 0.2,
        "max_tokens": 64,
    },
    timeout=60,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
