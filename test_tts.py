# -*- coding: utf-8 -*-
"""OpenAI gpt-4o-mini-tts 聲音比較 — 在你自己的終端機跑:  python test_tts.py
會用你 .env 的 CHATGPT_API_KEY 生成多個聲音,聽完挑編號告訴我。"""
import os
import ctypes
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

_BASE = Path(__file__).parent
load_dotenv(_BASE / ".env")

API_KEY = os.getenv("CHATGPT_API_KEY", "")
URL     = "https://api.openai.com/v1/audio/speech"
MODEL   = "gpt-4o-mini-tts"
TEXT    = "你好,這是語音測試。今天天氣不錯,我幫你把剛剛複製的內容整理成三個重點。"
# 用指令叫它講台灣口語、自然不念稿
INSTRUCT = "請用台灣人日常聊天的口吻說話,自然、親切、有溫度,不要像在念稿,語速適中。"

VOICES = ["coral", "sage", "shimmer", "nova", "ballad", "alloy"]


def play_mp3(path):
    w = ctypes.windll.winmm.mciSendStringW
    w(f'open "{path}" type mpegvideo alias m', None, 0, None)
    w("play m wait", None, 0, None)
    w("close m", None, 0, None)


def gen(voice, path):
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": MODEL, "voice": voice, "input": TEXT,
              "instructions": INSTRUCT, "response_format": "mp3"},
        timeout=60,
    )
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


if __name__ == "__main__":
    if not API_KEY:
        print("✗ .env 找不到 CHATGPT_API_KEY")
        raise SystemExit(1)
    for i, voice in enumerate(VOICES, 1):
        mp3 = os.path.join(tempfile.gettempdir(), f"tts_{voice}.mp3")
        print(f"\n[{i}] {voice}")
        try:
            gen(voice, mp3)
        except Exception as e:
            print(f"  ✗ 失敗: {type(e).__name__}: {e}")
            continue
        play_mp3(mp3)
    print("\n聽完了!告訴我哪個編號/名字最自然,或要再換語氣指令。")
