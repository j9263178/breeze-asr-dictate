# -*- coding: utf-8 -*-
"""edge-tts 台灣腔語音比較 — 在你自己的終端機跑:  python test_tts.py
會連續播放多種「聲音 + 語速 + 音調」組合,聽完挑編號告訴我。"""
import os
import asyncio
import tempfile
import ctypes

import edge_tts

TEXT = "你好,這是語音測試。今天天氣不錯,我幫你把剛剛複製的內容整理成三個重點。"

# (編號, 說明, 聲音, 語速, 音調)  ── 全部往「快 + 高」方向
VARIANTS = [
    ("1", "曉臻 快+略高",     "zh-TW-HsiaoChenNeural", "+12%", "+8Hz"),
    ("2", "曉臻 更快+略高",   "zh-TW-HsiaoChenNeural", "+20%", "+8Hz"),
    ("3", "曉臻 快+更高",     "zh-TW-HsiaoChenNeural", "+12%", "+18Hz"),
    ("4", "曉雨 快+略高",     "zh-TW-HsiaoYuNeural",   "+12%", "+8Hz"),
    ("5", "曉雨 更快+略高",   "zh-TW-HsiaoYuNeural",   "+20%", "+8Hz"),
    ("6", "曉雨 快+更高",     "zh-TW-HsiaoYuNeural",   "+12%", "+18Hz"),
]


def play_mp3(path: str):
    w = ctypes.windll.winmm.mciSendStringW
    w(f'open "{path}" type mpegvideo alias m', None, 0, None)
    w("play m wait", None, 0, None)
    w("close m", None, 0, None)


async def gen(text, voice, rate, pitch, path):
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(path)


if __name__ == "__main__":
    for num, desc, voice, rate, pitch in VARIANTS:
        mp3 = os.path.join(tempfile.gettempdir(), f"tts_{num}.mp3")
        print(f"\n[{num}] {desc}  (rate={rate}, pitch={pitch})")
        try:
            asyncio.run(gen(TEXT, voice, rate, pitch, mp3))
        except Exception as e:
            print(f"  ✗ 失敗: {type(e).__name__}: {e}")
            continue
        play_mp3(mp3)
    print("\n聽完了!告訴我哪個編號最自然,或要再微調語速/音調。")
