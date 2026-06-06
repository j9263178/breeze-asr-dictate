# -*- coding: utf-8 -*-
"""ElevenLabs Flash v2.5 測試:streaming 播放幾個女聲 + 印 TTFB。
跑:  .venv\\Scripts\\python.exe test_eleven.py"""
import os
import json
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

MODEL = "eleven_flash_v2_5"
RATE  = 24000
TEXT  = "你好,這是 ElevenLabs 的語音測試,聽聽看中文自然度跟延遲怎麼樣,順便念一句 English mixed in。"

# free tier 可用的女聲
VOICES = [
    ("Sarah",   "EXAVITQu4vr4xnSDxMaL"),
    ("Lily",    "pFZP5JQG7iQjIQuC4Bku"),
    ("Matilda", "XrExE9yKIg1WjnnlVkGX"),
]


def stream_and_play(name, voice_id):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    params = {"output_format": f"pcm_{RATE}", "optimize_streaming_latency": 4}
    body = {
        "text": TEXT,
        "model_id": MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                           "use_speaker_boost": True},
    }
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}

    for attempt in range(2):
        t0 = time.time()
        r = requests.post(url, params=params, headers=headers,
                          data=json.dumps(body), stream=True, timeout=60)
        if r.status_code != 200:
            print(f"  attempt {attempt+1}: HTTP {r.status_code} {r.text[:120]}")
            r.close()
            continue
        stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16")
        stream.start()
        first_t = None; total = 0; leftover = b""
        try:
            for chunk in r.iter_content(chunk_size=None):
                if not chunk: continue
                if first_t is None:
                    first_t = time.time() - t0
                    print(f"  TTFB: {first_t*1000:.0f}ms")
                blob = leftover + chunk
                even = len(blob) - (len(blob) % 2)
                if even:
                    stream.write(np.frombuffer(blob[:even], dtype=np.int16))
                    total += even
                leftover = blob[even:]
        finally:
            time.sleep(0.2); stream.stop(); stream.close()
        tot = time.time() - t0
        audio = total / 2 / RATE
        print(f"  總 {tot:.2f}s, 音訊 {audio:.1f}s, 生成/音訊 {tot/audio:.2f}x")
        return
    print("  ✗ 兩次都失敗")


if __name__ == "__main__":
    if not API_KEY:
        print("✗ .env 找不到 ELEVENLABS_API_KEY"); raise SystemExit(1)
    print(f"key prefix: {API_KEY[:8]}, model: {MODEL}\n")
    for name, vid in VOICES:
        print(f"[{name}]")
        stream_and_play(name, vid)
        print()
