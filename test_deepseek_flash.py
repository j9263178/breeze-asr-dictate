# -*- coding: utf-8 -*-
"""DeepSeek V4 Flash vs Gemini 3.5 Flash 速度 + 品質測試。
跑:  .venv\Scripts\python.exe test_deepseek_flash.py"""
import os, json, time
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
URL     = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    ("DeepSeek V4 Flash",  "deepseek/deepseek-v4-flash"),
    ("Gemini 3.5 Flash",   "google/gemini-3.5-flash"),
]

PROMPTS = [
    # 短問答:看 TTFB + 生成速度
    {
        "label": "[短問答] 解釋 async/await",
        "messages": [
            {"role": "system", "content":
                "You are a witty, concise friend. Reply in English, under 100 words. "
                "Occasionally use audio tags like [giggles] or [sighs] where natural."},
            {"role": "user", "content": "Explain async/await like I'm a smart friend, not a textbook."},
        ],
    },
    # 中等長度:看品質
    {
        "label": "[中等] 給建議",
        "messages": [
            {"role": "system", "content":
                "You are a witty, concise friend. Reply in English, under 150 words. "
                "Occasionally use audio tags like [giggles] or [sighs] where natural."},
            {"role": "user", "content":
                "I've been spending 3 hours a day on YouTube and feel guilty about it. "
                "What should I actually do about it?"},
        ],
    },
    # 中文輸入 → 英文輸出:測語言指令遵守
    {
        "label": "[語言切換] 中文問 → 英文回",
        "messages": [
            {"role": "system", "content":
                "Always reply in English even if the user writes in Chinese. "
                "Be casual and concise, under 100 words."},
            {"role": "user", "content": "你覺得 AI 會取代軟體工程師嗎?"},
        ],
    },
]

def stream_chat(model_id: str, messages: list) -> dict:
    """串流呼叫,回傳 {ttfb, total_time, tokens, tps, text}。"""
    body = {
        "model":  model_id,
        "messages": messages,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    t0 = time.perf_counter()
    r  = requests.post(URL, headers=headers, data=json.dumps(body),
                       stream=True, timeout=60)
    r.raise_for_status()

    ttfb       = None
    full_text  = ""
    token_count = 0

    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        text  = delta.get("content", "")
        if text:
            if ttfb is None:
                ttfb = time.perf_counter() - t0
            full_text   += text
            token_count += 1   # 每個 chunk ≈ 1 token(streaming)

    total = time.perf_counter() - t0
    tps   = token_count / total if total > 0 else 0
    return {
        "ttfb":       ttfb or 0,
        "total_time": total,
        "tokens":     token_count,
        "tps":        tps,
        "text":       full_text.strip(),
    }


def run():
    if not API_KEY:
        print("✗ .env 找不到 OPENROUTER_API_KEY"); raise SystemExit(1)
    print(f"API key: {API_KEY[:12]}…\n")

    for prompt in PROMPTS:
        print("=" * 60)
        print(f"  {prompt['label']}")
        print("=" * 60)
        results = []
        for name, model_id in MODELS:
            print(f"\n▶ {name}  ({model_id})")
            try:
                r = stream_chat(model_id, prompt["messages"])
                print(f"  TTFB:  {r['ttfb']*1000:.0f} ms")
                print(f"  Total: {r['total_time']:.2f}s  |  "
                      f"~{r['tokens']} chunks  |  {r['tps']:.1f} tok/s")
                print(f"  ─── output ───")
                print(f"  {r['text']}")
                results.append((name, r))
            except Exception as e:
                print(f"  ✗ 失敗: {e}")
        print()

    print("\n✓ 測試完成。")


if __name__ == "__main__":
    run()
