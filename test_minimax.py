# -*- coding: utf-8 -*-
"""MiniMax TTS 測試:用 t2a_v2 生成幾個中文女聲,存成 mp3。
跑:  .venv\\Scripts\\python.exe test_minimax.py"""
import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
API_KEY  = os.getenv("MINIMAX_API_KEY", "")
GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")

# 國際版端點(二擇一,下面會自動 fallback)
ENDPOINTS = [
    "https://api.minimaxi.chat/v1/t2a_v2",
    "https://api.minimax.io/v1/t2a_v2",
]
MODEL = "speech-2.8-hd"
TEXT  = (
    "Hey, (emm) so two hours of phone scrolling a day isn't exactly criminal. "
    "(chuckle) But the real question is — do you feel empty after? "
    "(sighs) If putting it down feels like a loss, that's your answer. "
    "<#0.5#> Set a screen time limit. At least make yourself feel something."
)

VOICES = [
    "Chinese (Mandarin)_Warm_Girl",
]


def synth(endpoint: str, voice_id: str, out_path: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": MODEL,
        "text":  TEXT,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0, "vol": 1.8, "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000, "bitrate": 128000,
            "format": "mp3", "channel": 1,
        },
    }
    t0 = time.time()
    r = requests.post(f"{endpoint}?GroupId={GROUP_ID}", headers=headers,
                      data=json.dumps(body), timeout=60)
    elapsed = time.time() - t0
    j = r.json()
    base = j.get("base_resp", {})
    if base.get("status_code") != 0:
        return f"API 錯誤 {base.get('status_code')}: {base.get('status_msg')}"
    audio_hex = j.get("data", {}).get("audio", "")
    if not audio_hex:
        return f"沒有音訊資料: {json.dumps(j, ensure_ascii=False)[:200]}"
    audio = bytes.fromhex(audio_hex)
    with open(out_path, "wb") as f:
        f.write(audio)
    return f"OK {len(audio)} bytes, {elapsed:.2f}s → {out_path}"


if __name__ == "__main__":
    if not API_KEY or not GROUP_ID:
        print(f"✗ 缺少設定 API_KEY={bool(API_KEY)} GROUP_ID={bool(GROUP_ID)}")
        raise SystemExit(1)
    print(f"key prefix: {API_KEY[:8]}  group: {GROUP_ID}\n")

    # 先找能用的 endpoint
    endpoint = None
    for ep in ENDPOINTS:
        print(f"試 endpoint: {ep}")
        res = synth(ep, VOICES[0], "minimax_0.mp3")
        print(f"  {res}")
        if res.startswith("OK"):
            endpoint = ep
            break
    if not endpoint:
        print("\n✗ 兩個 endpoint 都失敗(看上面錯誤訊息)")
        raise SystemExit(1)

    print(f"\n用 {endpoint} 生成其餘聲音:\n")
    for i, v in enumerate(VOICES[1:], 1):
        res = synth(endpoint, v, f"minimax_{i}.mp3")
        print(f"[{v}] {res}")
    print("\n聽聽看 minimax_0.mp3 ~ minimax_4.mp3,告訴我哪個好。")
