# -*- coding: utf-8 -*-
"""
Breeze-ASR-25 全域語音聽寫 + AI 問答
──────────────────────────────────────
【Copilot 鍵】          → 切換錄音(開始/停止),結果貼到游標處
【右Alt + Copilot 鍵】  → AI 模式:剪貼簿內容 + 語音問題送 LLM,回覆用台灣腔念出來
                          (AI_TTS=False 則改回貼上文字)

模型常駐 VRAM,只在啟動時載入一次。
結束程式:在這個視窗按 Ctrl+C。
"""

import os
import re
import time
import wave
import json
import ctypes
import asyncio
import tempfile
import threading
import winsound
from collections import deque
from pathlib import Path

_BASE = Path(__file__).parent  # 程式所在資料夾(任何路徑都適用)

# ── 載入 .env ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(_BASE / ".env")

import numpy as np
import requests
import sounddevice as sd
import torch
import keyboard
import pyperclip
from transformers import pipeline

try:
    import edge_tts
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False

# ───────────────────────── 設定 ─────────────────────────
MODEL_DIR    = _BASE / "models" / "Breeze-ASR-25"
SAMPLE_RATE  = 16_000
MIN_SECONDS  = 0.3
MAX_SECONDS  = 60
LANGUAGE     = "chinese"
RESTORE_CLIPBOARD = False
VOCAB_FILE   = _BASE / "vocab.txt"

# ── xAI (Grok) 設定 ───────────────────────────────────────
XAI_API_KEY    = os.getenv("XAI_API_KEY", "")
XAI_URL        = "https://api.x.ai/v1/responses"   # Responses API(支援 web_search 工具)
XAI_MODEL      = "grok-4-fast-non-reasoning"        # 非推理模型
AI_WEB_SEARCH  = True            # True = 開啟即時網路搜尋(模型自行判斷需不需要搜)
AI_HISTORY_TURNS = 15            # 保留最近幾輪對話(每輪 = user + assistant)
AI_SYSTEM_PROMPT = (
    "你是一個高效的中英雙語助理。"
    "使用者會提供一段剪貼簿內容（脈絡）和一個語音問題。"
    "請根據脈絡回答問題，回覆簡潔有力，不要過度解釋。"
    "若脈絡為空，直接回答問題即可。"
    "需要最新資訊時才使用網路搜尋。"
)

# 對話記憶:存 {"role": "user"|"assistant", "content": "..."}
# maxlen = AI_HISTORY_TURNS * 2,因為每輪有兩條訊息
_chat_history: deque = deque(maxlen=AI_HISTORY_TURNS * 2)

# ── AI 語音回覆(TTS)設定 ─────────────────────────────────
AI_TTS       = True                       # True = AI 回覆用台灣腔念出來(只念不貼);False = 貼文字
AI_TTS_VOICE = "zh-TW-HsiaoChenNeural"    # 曉臻(女,台灣腔)
AI_TTS_RATE  = "+20%"                      # 語速
AI_TTS_PITCH = "+18Hz"                     # 音調(偏高)
_tts_lock    = threading.Lock()

# ─────────────────────── 提示音 ───────────────────────────
def _make_tone(path, freq, ms, sr=44100, volume=0.35):
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    wave_data = np.sin(2 * np.pi * freq * t)
    wave_data += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    env = np.exp(-t * (4500 / ms))
    env[: int(sr * 0.005)] *= np.linspace(0, 1, int(sr * 0.005))
    audio = (wave_data * env * volume * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(audio.tobytes())

_snd_dir  = tempfile.gettempdir()
_SND_START    = os.path.join(_snd_dir, "dictate_start.wav")
_SND_STOP     = os.path.join(_snd_dir, "dictate_stop.wav")
_SND_ERR      = os.path.join(_snd_dir, "dictate_err.wav")
_SND_AI_START = os.path.join(_snd_dir, "dictate_ai_start.wav")
_SND_AI_DONE  = os.path.join(_snd_dir, "dictate_ai_done.wav")
_TTS_MP3      = os.path.join(_snd_dir, "dictate_ai_tts.mp3")

_make_tone(_SND_START,    988,  180)   # B5  清亮  = 普通錄音開始
_make_tone(_SND_STOP,     659,  220)   # E5  沉穩  = 停止/運算
_make_tone(_SND_ERR,      330,  320)   # E4  低    = 沒結果/出錯
_make_tone(_SND_AI_START, 1319, 180)   # E6  高亮  = AI 模式開始(比 B5 高一個大六度,一聽即知)
_make_tone(_SND_AI_DONE,  880,  280)   # A5  暖    = AI 回覆完成

def _play_file(path):
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

def beep_start():    _play_file(_SND_START)
def beep_stop():     _play_file(_SND_STOP)
def beep_error():    _play_file(_SND_ERR)
def beep_ai_start(): _play_file(_SND_AI_START)
def beep_ai_done():  _play_file(_SND_AI_DONE)

# ─────────────────────── 載入模型 ────────────────────────
print("載入 Breeze-ASR-25 中(約 20~30 秒)…")
_t0 = time.time()
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    dtype=torch.float16,
    device=0,
    chunk_length_s=30,
)
print(f"模型就緒,耗時 {time.time() - _t0:.1f}s。")

# ─────────────────────── 自訂詞彙偏置 ────────────────────
GEN_KWARGS = {"language": LANGUAGE, "task": "transcribe"}

def _load_vocab_prompt():
    if not os.path.exists(VOCAB_FILE):
        return
    terms = []
    with open(VOCAB_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    if not terms:
        return
    prompt = "、".join(terms)
    try:
        ids = asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        GEN_KWARGS["prompt_ids"] = ids.to(asr.model.device)
        print(f"已載入自訂詞彙 {len(terms)} 個(偏置生效)。")
    except Exception as e:
        print(f"⚠ 詞彙偏置載入失敗,改用預設:{e}")

_load_vocab_prompt()

if XAI_API_KEY:
    _search_note = "+網路搜尋" if AI_WEB_SEARCH else ""
    if AI_TTS and _HAS_TTS:
        _out_note = "語音回覆:曉臻台灣腔"
    elif AI_TTS and not _HAS_TTS:
        _out_note = "⚠ 未裝 edge-tts,改貼文字"
    else:
        _out_note = "文字貼上"
    print(f"Grok AI 模式已就緒({XAI_MODEL}{_search_note};{_out_note})。")
else:
    print("⚠ 未設定 XAI_API_KEY,AI 模式停用。")

print("按【Copilot 鍵】開始說話,再按一次轉錄。")
print("按【右Alt + Copilot 鍵】進入 AI 模式。")
print("結束請按 Ctrl+C。")

# ─────────────────────── 錄音串流 ────────────────────────
_frames    = []
_recording = False
_ai_mode   = False          # True = 本次錄音是 AI 問答模式
_ai_context = ""            # 錄音開始時的剪貼簿快照
_lock      = threading.Lock()
_timeout_timer = None

def _audio_callback(indata, frames, time_info, status):
    if _recording:
        _frames.append(indata.copy())

_stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
    callback=_audio_callback,
)
_stream.start()

# ─────────────────────── 貼上工具 ────────────────────────
def _paste_text(text: str):
    old = ""
    if RESTORE_CLIPBOARD:
        try:    old = pyperclip.paste()
        except: old = ""
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    if RESTORE_CLIPBOARD:
        time.sleep(0.25)
        try:    pyperclip.copy(old)
        except: pass

# ─────────────────────── 台灣腔語音(TTS)─────────────────
def _clean_for_speech(text: str) -> str:
    """念出來前移除 markdown / 引用網址,避免 TTS 把連結念出來。"""
    # [[1]](http...)、[文字](http...) → 只留文字(citation 標記直接拿掉)
    text = re.sub(r"\[\[\d+\]\]\([^)]*\)", "", text)        # [[1]](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)    # [文字](url)
    text = re.sub(r"https?://\S+", "", text)                # 裸網址
    text = text.replace("**", "").replace("*", "").replace("`", "").replace("#", "")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def _speak(text: str):
    """用 edge-tts 生成台灣腔語音並播放(blocking)。失敗會丟出例外。"""
    async def _gen():
        await edge_tts.Communicate(
            text, AI_TTS_VOICE, rate=AI_TTS_RATE, pitch=AI_TTS_PITCH
        ).save(_TTS_MP3)
    asyncio.run(_gen())
    w = ctypes.windll.winmm.mciSendStringW
    with _tts_lock:
        w(f'open "{_TTS_MP3}" type mpegvideo alias aitts', None, 0, None)
        w("play aitts wait", None, 0, None)   # wait = 播完才返回
        w("close aitts", None, 0, None)

# ─────────────────────── xAI (Grok) LLM ──────────────────
def _extract_answer(data: dict) -> str:
    """從 Responses API 回應抽出最終文字(output 裡 type=message 的 output_text)。"""
    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    parts.append(c["text"])
    return "\n".join(parts).strip()

def _ask_llm(context: str, question: str) -> str:
    """把 context(剪貼簿) + question(語音) 送到 Grok(可即時搜尋),回傳回覆並更新記憶。"""
    user_msg = ""
    if context.strip():
        user_msg += f"【剪貼簿內容】\n{context.strip()}\n\n"
    user_msg += f"【問題】\n{question.strip()}"

    # Responses API:system 放 instructions,歷史 + 本次 user 放 input
    input_msgs = list(_chat_history) + [{"role": "user", "content": user_msg}]

    payload = {
        "model":        XAI_MODEL,
        "instructions": AI_SYSTEM_PROMPT,
        "input":        input_msgs,
    }
    if AI_WEB_SEARCH:
        payload["tools"] = [{"type": "web_search"}]   # 模型自行判斷是否搜尋

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    resp = requests.post(XAI_URL, headers=headers,
                         data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    answer = _extract_answer(resp.json())

    # 把本輪存進記憶
    _chat_history.append({"role": "user",      "content": user_msg})
    _chat_history.append({"role": "assistant", "content": answer})

    return answer

# ─────────────────────── 轉錄 Worker ─────────────────────
def _transcribe(audio: np.ndarray) -> str:
    out = asr(
        {"raw": audio, "sampling_rate": SAMPLE_RATE},
        generate_kwargs=GEN_KWARGS,
        return_timestamps=True,
    )
    chunks = out.get("chunks") or []
    if chunks:
        parts = [c["text"].strip() for c in chunks if c.get("text", "").strip()]
        return " ".join(parts)
    return out["text"].strip()

def _dictate_worker(audio: np.ndarray):
    """普通聽寫模式。"""
    try:
        t0   = time.time()
        text = _transcribe(audio)
        print(f"  → ({time.time()-t0:.1f}s) {text!r}")
        if text:
            _paste_text(text)
        else:
            beep_error()
    except Exception as e:
        print(f"  ✗ 轉錄失敗: {e}")
        beep_error()

def _ai_worker(audio: np.ndarray, context: str):
    """AI 問答模式:ASR → LLM → 剪貼簿。"""
    try:
        t0       = time.time()
        question = _transcribe(audio)
        print(f"  → ASR ({time.time()-t0:.1f}s) {question!r}")
        if not question:
            beep_error()
            return

        turns = len(_chat_history) // 2
        print(f"  → 送 LLM … (脈絡 {len(context)} 字 / 歷史 {turns} 輪)")
        t1     = time.time()
        answer = _ask_llm(context, question)
        print(f"  → LLM ({time.time()-t1:.1f}s) {answer!r}")

        if AI_TTS and _HAS_TTS:
            print("  → 念出回覆中…")
            try:
                _speak(_clean_for_speech(answer))
                print("  ✓ 已念出。")
            except Exception as e:
                print(f"  ⚠ 語音失敗,改貼文字: {e}")
                _paste_text(answer)
        else:
            beep_ai_done()
            _paste_text(answer)
            print("  ✓ 回覆已貼上。")
    except Exception as e:
        print(f"  ✗ AI 模式失敗: {e}")
        beep_error()

# ─────────────────────── 熱鍵邏輯 ────────────────────────
def _start_recording(ai: bool):
    global _recording, _frames, _timeout_timer, _ai_mode, _ai_context
    with _lock:
        if _recording:
            return
        _frames    = []
        _recording = True
        _ai_mode   = ai
        _ai_context = ""
    if ai:
        # 先快照剪貼簿,讓後面錄音時使用者可以繼續複製新內容也沒關係
        try:    _ai_context = pyperclip.paste()
        except: _ai_context = ""
        beep_ai_start()
        print(f"★ AI 模式錄音中…(再按 Copilot 停止;最長 {MAX_SECONDS}s)")
    else:
        beep_start()
        print(f"● 錄音中…(再按一次停止;最長 {MAX_SECONDS}s 自動停)")
    _timeout_timer = threading.Timer(MAX_SECONDS, _auto_stop)
    _timeout_timer.daemon = True
    _timeout_timer.start()

def _auto_stop():
    print(f"⏱ 已達上限 {MAX_SECONDS}s,自動停止。")
    _stop_recording()

def _stop_recording():
    global _recording, _timeout_timer
    with _lock:
        if not _recording:
            return
        _recording = False
        frames     = list(_frames)
        ai         = _ai_mode
        ctx        = _ai_context
    if _timeout_timer is not None:
        _timeout_timer.cancel()
        _timeout_timer = None
    beep_stop()
    if not frames:
        return
    audio = np.concatenate(frames, axis=0).flatten().astype(np.float32)
    dur   = len(audio) / SAMPLE_RATE
    if dur < MIN_SECONDS:
        print(f"  (錄音太短 {dur:.2f}s,忽略)")
        return
    print(f"■ 停止,長度 {dur:.1f}s,{'AI 問答' if ai else '轉錄'}中…")
    if ai:
        threading.Thread(target=_ai_worker,     args=(audio, ctx),  daemon=True).start()
    else:
        threading.Thread(target=_dictate_worker, args=(audio,),     daemon=True).start()

def _toggle(ai: bool):
    if _recording:
        _stop_recording()
    else:
        _start_recording(ai)

def _on_f23(event):
    if event.event_type != "down":
        return
    ralt_held = keyboard.is_pressed("right alt")
    if ralt_held and not XAI_API_KEY:
        print("⚠ 未設定 XAI_API_KEY,AI 模式無法使用。")
        beep_error()
        return
    _toggle(ai=ralt_held)

keyboard.hook_key("f23", _on_f23, suppress=True)

# ─────────────────────── 主迴圈 ──────────────────────────
try:
    keyboard.wait()
except KeyboardInterrupt:
    pass
finally:
    _stream.stop()
    _stream.close()
    print("\n已結束。")
