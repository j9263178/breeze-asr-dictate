# -*- coding: utf-8 -*-
"""快速驗證 Gemini API key + model 是否正常。
跑:  .venv\Scripts\python.exe test_gemini.py"""
import os, json, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
KEY = os.getenv("GEMINI_API_KEY", "")
if not KEY:
    print("✗ 找不到 GEMINI_API_KEY"); raise SystemExit(1)
print(f"Key prefix: {KEY[:12]}…\n")

MODELS = ["gemini-2.5-flash", "gemini-3.5-flash"]

for model in MODELS:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={KEY}")
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Say exactly: OK"}]}]
    }
    print(f"Testing {model} … ", end="", flush=True)
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          data=json.dumps(body), timeout=15)
        if r.status_code == 200:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text  = "".join(p.get("text","") for p in parts).strip()
            print(f"✓  reply: {text!r}")
        else:
            print(f"✗  HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"✗  {e}")
