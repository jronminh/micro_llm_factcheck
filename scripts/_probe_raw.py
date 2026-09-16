"""Probe nhanh raw JSON response tu llama-server cho model moi - de xem
content rong la do thieu field nao (vd reasoning_content cua model dual-mode
reasoning) hay model thuc su khong sinh gi."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

import llm

MODELS_DIR = Path(__file__).parent.parent / "models"

for label, filename, user_prompt in [
    ("qwen3-0.6b /no_think", "qwen3-0.6b-q4_k_m.gguf", "Chiều cao núi Everest là bao nhiêu mét? /no_think"),
]:
    llm.MODEL_PATH = MODELS_DIR / filename
    llm.ensure_server()
    payload = {
        "messages": [
            {"role": "system", "content": llm.SYSTEM_PROMPTS["synth"]},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
    }
    resp = requests.post(f"{llm.SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
    print(f"=== {label} ===")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:3000])
    print()
