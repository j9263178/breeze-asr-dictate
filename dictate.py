# -*- coding: utf-8 -*-
"""
Breeze-ASR-25 全域語音聽寫
─────────────────────────
按 Copilot 鍵   → 開始錄音(上揚提示音)
再按 Copilot 鍵 → 停止並轉錄(下降提示音),結果用剪貼簿貼到游標處

模型常駐 VRAM,只在啟動時載入一次。
結束程式:在這個視窗按 Ctrl+C。
"""

import os
import time
import wave
import tempfile
import threading
import winsound
from pathlib import Path

_BASE = Path(__file__).parent  # 程式所在資料夾(任何路徑都適用)

import numpy as np
import sounddevice as sd
import torch
import keyboard
import pyperclip
from transformers import pipeline

# ───────────────────────── 設定 ─────────────────────────
MODEL_DIR = _BASE / "models" / "Breeze-ASR-25"   # 自動抓程式所在資料夾,不連網
SAMPLE_RATE = 16_000          # Breeze-ASR-25 / Whisper 固定吃 16kHz
# 熱鍵 = Copilot 鍵(送 Win+Shift+F23,以獨特的 f23 辨識)
MIN_SECONDS = 0.3             # 太短的錄音忽略(手滑誤觸)
MAX_SECONDS = 60              # 錄音上限:超過自動停止轉錄(防忘記按第二下)
LANGUAGE = "chinese"          # 強制中文解碼(中英混合仍可正常輸出英文)
RESTORE_CLIPBOARD = False     # False=辨識結果留在剪貼簿(沒貼到可手動 Ctrl+V);True=還原原本內容
VOCAB_FILE = _BASE / "vocab.txt"  # 自訂詞彙(一行一個;# 開頭為註解)

# ─────────────────────── 提示音(柔和正弦鈴聲)───────────────────────
def _make_tone(path, freq, ms, sr=44100, volume=0.35):
    """產生帶淡入 + 指數衰減的正弦音檔(比方波 Beep 柔和很多)。"""
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    wave_data = np.sin(2 * np.pi * freq * t)
    # 加一點八度泛音讓音色更像鈴鐺
    wave_data += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    env = np.exp(-t * (4500 / ms))          # 指數衰減 = 鈴聲尾韻
    env[: int(sr * 0.005)] *= np.linspace(0, 1, int(sr * 0.005))  # 5ms 淡入去爆音
    audio = (wave_data * env * volume * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(audio.tobytes())

_snd_dir = tempfile.gettempdir()
_SND_START = os.path.join(_snd_dir, "dictate_start.wav")
_SND_STOP = os.path.join(_snd_dir, "dictate_stop.wav")
_SND_ERR = os.path.join(_snd_dir, "dictate_err.wav")
_make_tone(_SND_START, 988, 180)   # B5,清亮 = 開始(按下即可講)
_make_tone(_SND_STOP, 659, 220)    # E5,沉穩 = 停止/運算
_make_tone(_SND_ERR, 330, 320)     # E4,低 = 沒結果/出錯

def _play_file(path):
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

def _play_alias(alias):
    winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)

# 提示音來源:"system" = Windows 內建音效;"tone" = 上面自製正弦鈴聲
BEEP_MODE = "tone"

def beep_start():
    if BEEP_MODE == "system": _play_alias("SystemAsterisk")     # 資訊「叮」
    else: _play_file(_SND_START)

def beep_stop():
    if BEEP_MODE == "system": _play_alias("SystemExclamation")  # 提示音
    else: _play_file(_SND_STOP)

def beep_error():
    if BEEP_MODE == "system": _play_alias("SystemHand")         # 錯誤音
    else: _play_file(_SND_ERR)

# ─────────────────────── 載入模型 ───────────────────────
print("載入 Breeze-ASR-25 中(約 20~30 秒)…")
_t0 = time.time()
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    dtype=torch.float16,
    device=0,                 # GPU 0 (CUDA) = 你的 RTX 4060
    chunk_length_s=30,
)
print(f"模型就緒,耗時 {time.time() - _t0:.1f}s。")

# ─────────────────────── 自訂詞彙偏置(initial prompt)───────────────────────
# 把 vocab.txt 的詞餵成 Whisper 的 prompt,解碼時會偏向認成這些詞。
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
    prompt = "、".join(terms)          # 以頓號串成一段提示詞
    try:
        ids = asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        GEN_KWARGS["prompt_ids"] = ids.to(asr.model.device)
        print(f"已載入自訂詞彙 {len(terms)} 個(偏置生效)。")
    except Exception as e:
        print(f"⚠ 詞彙偏置載入失敗,改用預設:{e}")

_load_vocab_prompt()
print("按【Copilot 鍵】開始說話,再按一次轉錄。結束請按 Ctrl+C。")

# ─────────────────────── 錄音(串流常駐,旗標控制收音)───────────────────────
_frames = []
_recording = False
_lock = threading.Lock()
_timeout_timer = None

def _audio_callback(indata, frames, time_info, status):
    if _recording:
        _frames.append(indata.copy())

_stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_audio_callback
)
_stream.start()

# ─────────────────────── 轉錄 + 貼上 ───────────────────────
def _paste_text(text: str):
    old = ""
    if RESTORE_CLIPBOARD:
        try:
            old = pyperclip.paste()
        except Exception:
            old = ""
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    if RESTORE_CLIPBOARD:
        time.sleep(0.25)
        try:
            pyperclip.copy(old)
        except Exception:
            pass

def _transcribe_worker(audio: np.ndarray):
    try:
        t0 = time.time()
        out = asr(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            generate_kwargs=GEN_KWARGS,
            return_timestamps=True,        # 取得依停頓切分的小段
        )
        chunks = out.get("chunks") or []
        if chunks:
            # 各段之間用空格隔開(停頓處 = 空格),不加標點
            parts = [c["text"].strip() for c in chunks if c.get("text", "").strip()]
            text = " ".join(parts)
        else:
            text = out["text"].strip()
        print(f"  → ({time.time() - t0:.1f}s) {text!r}")
        if text:
            _paste_text(text)
        else:
            beep_error()
    except Exception as e:
        print(f"  ✗ 轉錄失敗: {e}")
        beep_error()

# ─────────────────────── 熱鍵:Copilot 鍵切換 ───────────────────────
def _start_recording():
    global _recording, _frames, _timeout_timer
    with _lock:
        if _recording:
            return
        _frames = []
        _recording = True
    # 啟動上限計時器:時間到自動停止
    _timeout_timer = threading.Timer(MAX_SECONDS, _auto_stop)
    _timeout_timer.daemon = True
    _timeout_timer.start()
    beep_start()
    print(f"● 錄音中…(再按一次停止;最長 {MAX_SECONDS}s 自動停)")

def _auto_stop():
    print(f"⏱ 已達上限 {MAX_SECONDS}s,自動停止。")
    _stop_recording()

def _stop_recording():
    global _recording, _timeout_timer
    with _lock:
        if not _recording:
            return
        _recording = False
        frames = _frames
    if _timeout_timer is not None:
        _timeout_timer.cancel()
        _timeout_timer = None
    beep_stop()
    if not frames:
        return
    audio = np.concatenate(frames, axis=0).flatten().astype(np.float32)
    dur = len(audio) / SAMPLE_RATE
    if dur < MIN_SECONDS:
        print(f"  (錄音太短 {dur:.2f}s,忽略)")
        return
    print(f"■ 停止,長度 {dur:.1f}s,轉錄中…")
    threading.Thread(target=_transcribe_worker, args=(audio,), daemon=True).start()

def _toggle():
    if _recording:
        _stop_recording()
    else:
        _start_recording()

def _on_f23(event):
    # Copilot 鍵會送 Win+Shift+F23,只在 f23 按下時切換
    if event.event_type == "down":
        _toggle()

# suppress=True:把 f23 吞掉,Windows 收不到完整 Win+Shift+F23 → 不會跳 Copilot
keyboard.hook_key("f23", _on_f23, suppress=True)

# ─────────────────────── 主迴圈 ───────────────────────
try:
    keyboard.wait()
except KeyboardInterrupt:
    pass
finally:
    _stream.stop()
    _stream.close()
    print("\n已結束。")
