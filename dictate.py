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
import base64
import random
import socket
import ctypes
from ctypes import wintypes
import asyncio
import tempfile
import threading
import winsound
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
OUTPUT_MODE  = "type"        # 聽寫輸出:"type"=直接模擬鍵盤輸入(不碰剪貼簿,推薦)
                             # "clipboard"=複製+Ctrl+V(會覆蓋你剪貼簿的內容)
RESTORE_CLIPBOARD = False    # 只在 OUTPUT_MODE="clipboard" 有效:True=貼完還原
# AI 回覆專用:長文 + 中文標點用 type 容易被 IME 攔截順序錯亂,改用 clipboard 較穩
AI_OUTPUT_MODE        = "clipboard"   # "clipboard"=剪貼簿(推薦) / "type"=直接打字
AI_RESTORE_CLIPBOARD  = True          # True=貼完還原剪貼簿(脈絡 dedup 還是 work)
VOCAB_FILE   = _BASE / "vocab.txt"

# ── 熱鍵設定 ──────────────────────────────────────────────
# 沒有 Copilot 鍵的使用者可改成其他按鍵，例如："f9"、"scroll lock"、"pause"
HOTKEY          = "f23"        # 主熱鍵：Copilot 鍵 = f23；無 Copilot 鍵請自行替換
HOTKEY_SUPPRESS = True         # True = 吞掉熱鍵事件（Copilot 鍵需要，避免跳出 Copilot 視窗）
                               # 改成其他鍵時通常可設 False
AI_MODIFIER     = "right alt"  # AI 模式的修飾鍵（同時按住此鍵 + 主熱鍵即觸發 AI 模式）

# ── 全自動模式設定 ────────────────────────────────────────
# Left Alt + 主熱鍵 切換開/關;開啟後每隔 AUTO_INTERVAL 秒自動 ASR → AI → 念出回覆
AUTO_MODIFIER    = "left alt"  # 自動模式修飾鍵
AUTO_INTERVAL    = 20          # 每幾秒處理一次(秒)
AUTO_MIN_RMS     = 0.002       # 靜音閾值:低於此值視為沒講話,跳過不送 AI(console 會印實際 RMS 方便調整)
AUTO_CONTEXT     = True        # True = 自動模式也帶剪貼簿脈絡(按下開啟時快照)
# 全自動模式專用 LLM(走 OpenRouter,比 xAI 便宜)
AUTO_LLM_URL    = "https://openrouter.ai/api/v1/chat/completions"
AUTO_LLM_KEY    = os.getenv("OPENROUTER_API_KEY", "")
AUTO_LLM_MODEL  = "openai/gpt-5-nano"             # 快、便宜;可換 google/gemini-2.0-flash-001 等

# ── xAI (Grok) 設定 ───────────────────────────────────────
XAI_API_KEY    = os.getenv("XAI_API_KEY", "")
XAI_URL        = "https://api.x.ai/v1/responses"   # Responses API(支援 web_search 工具)
XAI_MODEL      = "grok-4.20-0309-non-reasoning"     # 非推理模型
AI_WEB_SEARCH  = True            # True = 開啟即時網路搜尋(模型自行判斷需不需要搜)
AI_HISTORY_TURNS = 15            # 逐字保留的輪數上限,超過就觸發壓縮(每輪 = user + assistant)
AI_KEEP_RECENT   = 5             # 壓縮後保留最近幾輪逐字,其餘併入摘要
AI_SUMMARY_CHARS = 500           # 滾動摘要的字數上限
AI_SYSTEM_PROMPT = (
    "你是使用者的聰明好友——機智、有活力、帶一點俏皮幽默,講話自然像在跟朋友聊天,"
    "偶爾可以輕輕吐槽一句,但點到為止、絕不刻薄,讓人覺得親切又可靠。"

    # 工具使用情境說明
    "【你的使用情境】"
    "使用者在用一個語音聽寫 + AI 助理的桌面工具。流程是這樣的:"
    "使用者先把想讓你參考的內容複製到剪貼簿(可能是一篇文章、一段對話、一個網頁、程式碼等),"
    "然後按熱鍵、用語音講出問題或指令。"
    "你收到的每則訊息格式是:"
    "「【剪貼簿內容】...（使用者複製的東西）\n\n【問題】...（語音辨識的問題）」。"
    "剪貼簿內容是使用者刻意提供的參考脈絡,你要優先根據它來回答問題。"
    "如果某次沒有剪貼簿內容(空的),就直接回答問題、或從對話歷史找脈絡。"

    # 回答風格
    "用台灣口語的繁體中文回答(中英夾雜很自然),簡潔有力、不囉嗦。"
    "預設回覆控制在 150 字以內,直接給重點。"
    "但若使用者明確要求「詳細說明」、「仔細分析」、「完整列出」之類,就放寬字數。"
    "回答完就停,不要加反問句(「你覺得呢?」之類)、不要客套收尾(「希望這對你有幫助」之類)。"
    "只有真的需要補充資訊才能回答時,才問問題。"

    # 格式限制(回覆會被念出來+打字輸出)
    "請正常使用標點符號,讓句子有自然停頓。"
    "不要用任何 markdown(不用 **粗體**、# 標題、- 清單、--- 分隔線)。"
    "不要用 emoji 或表情符號。"
    "需要即時資訊時才上網搜尋。"
    "可以在適當地方(不用每句都加)插入語音標記讓回覆更生動:"
    "[laughs](笑)、[sighs](嘆氣)、[whispers](悄悄話)、[giggles](輕笑)。"
    "一段回覆最多一兩個,語氣真的合適才用。"
)

# 對話記憶:逐字最近對話(list of {"role","content"})+ 一份滾動摘要
_chat_history: list = []
_chat_summary: str = ""
_last_sent_clipboard: str = ""   # 上次送 AI 的剪貼簿內容,跟這次一樣就不重複送

# ── AI 語音回覆(TTS)設定 ─────────────────────────────────
AI_TTS        = True                      # True = AI 回覆用語音念出來;False = 不念
AI_TTS_ENGINE = "eleven"                  # "eleven" = ElevenLabs Flash v2.5(雲端,低延遲,需 ELEVENLABS_API_KEY);
                                          # "cosy"   = 本地 CosyVoice 2 server(最自然,需先啟動 server.py);
                                          # "gemini" = Gemini 3.1 Flash TTS(需 GEMINI_API_KEY);
                                          # "edge"   = edge-tts(免費 fallback,較機械)
# ElevenLabs 設定
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVEN_MODEL   = "eleven_v3"              # v3=支援 [laughs][sighs] 等 audio tags(自然);flash=更低延遲但會把 tag 念出來
ELEVEN_RATE    = 24000
ELEVEN_VOICE   = "ht0yrHEoOG42OGi3ERZs"   # 你選的聲音;其他:Sarah=EXAVITQu4vr4xnSDxMaL, Lily=pFZP5JQG7iQjIQuC4Bku
ELEVEN_STABILITY = 0.5
ELEVEN_SIMILARITY = 0.75
# CosyVoice 設定(server 在 cosyvoice/server.py,獨立 conda env)
COSY_HOST     = "127.0.0.1"
COSY_PORT     = 8765
COSY_RATE     = 24000
COSY_VOICE    = "tiffy"     # 對應 cosyvoice/voices/<name>/;新增聲音用 tools/add_voice.py
COSY_SPEED    = 1.0         # 0.5~2.0,1.15 = 快 15%、0.9 = 慢 10%
COSY_INSTRUCT = ""          # 非空 → instruct 模式(語氣指令,例:「用輕鬆的口吻念」),但會慢一點
# Gemini TTS 設定
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
# 每次念都從這個池子隨機挑一個聲音(避免一直聽到同一個人)。
# 留空 list 則永遠用單一聲音 GEMINI_TTS_VOICE。
GEMINI_TTS_VOICES = ["Leda", "Sulafat", "Laomedeia", "Erinome", "Aoede", "Achernar"]
GEMINI_TTS_VOICE  = "Leda"                # 上面 list 空的時候 fallback 用這個
GEMINI_TTS_STYLE = (                      # 語氣指令(放在文字前面)
    "請用台灣人平靜、輕柔的口吻念出以下文字,"
    "音量放輕、力道放鬆、像在耳邊輕聲說話,"
    "語調平穩、不要有太多起伏、不要過度抑揚頓挫、絕對不要用重音強調,"
    "語速偏快、流暢俐落,像在簡潔陳述事情。"
    "偶爾(不是每句都要)可以自然帶入輕微的呼吸聲、輕笑聲、或像 嗯 啊 之類的口頭停頓,"
    "讓聽起來更像真人在說話,但不要刻意誇張:"
)
# edge-tts 設定(fallback)
AI_TTS_VOICE = "zh-TW-HsiaoChenNeural"    # 曉臻(女,台灣腔)
AI_TTS_RATE  = "+20%"
AI_TTS_PITCH = "+18Hz"
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
_SND_START      = os.path.join(_snd_dir, "dictate_start.wav")
_SND_STOP       = os.path.join(_snd_dir, "dictate_stop.wav")
_SND_ERR        = os.path.join(_snd_dir, "dictate_err.wav")
_SND_AI_START   = os.path.join(_snd_dir, "dictate_ai_start.wav")
_SND_AI_DONE    = os.path.join(_snd_dir, "dictate_ai_done.wav")
_SND_AUTO_ON    = os.path.join(_snd_dir, "dictate_auto_on.wav")
_SND_AUTO_OFF   = os.path.join(_snd_dir, "dictate_auto_off.wav")
_SND_AUTO_TICK  = os.path.join(_snd_dir, "dictate_auto_tick.wav")
_TTS_MP3        = os.path.join(_snd_dir, "dictate_ai_tts.mp3")
_TTS_WAV        = os.path.join(_snd_dir, "dictate_ai_tts.wav")

_make_tone(_SND_START,     988,  180)   # B5  清亮  = 普通錄音開始
_make_tone(_SND_STOP,      659,  220)   # E5  沉穩  = 停止/運算
_make_tone(_SND_ERR,       330,  320)   # E4  低    = 沒結果/出錯
_make_tone(_SND_AI_START,  1319, 180)   # E6  高亮  = AI 模式開始
_make_tone(_SND_AI_DONE,   880,  280)   # A5  暖    = AI 回覆完成
_make_tone(_SND_AUTO_ON,   523,  80)    # C5 短促   = 自動模式開啟(第一音)
_make_tone(_SND_AUTO_OFF,  262,  200)   # C4 低     = 自動模式關閉
_make_tone(_SND_AUTO_TICK, 440,  60)    # A4 極短   = 每次 tick 開始處理

def _play_file(path):
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

def beep_start():     _play_file(_SND_START)
def beep_stop():      _play_file(_SND_STOP)
def beep_error():     _play_file(_SND_ERR)
def beep_ai_start():  _play_file(_SND_AI_START)
def beep_ai_done():   _play_file(_SND_AI_DONE)
def beep_auto_on():
    winsound.PlaySound(_SND_AUTO_ON, winsound.SND_FILENAME)   # 同步第一音
    _play_file(_SND_AUTO_ON)                                   # 立刻再播一聲(雙音=開啟)
def beep_auto_off():  _play_file(_SND_AUTO_OFF)
def beep_auto_tick(): _play_file(_SND_AUTO_TICK)

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

# 啟動時偵測 cosy server 是否在跑(只試 TCP 連線,不打 HTTP)
_cosy_alive = False
if AI_TTS_ENGINE == "cosy":
    try:
        _s = socket.socket(); _s.settimeout(1); _s.connect((COSY_HOST, COSY_PORT)); _s.close()
        _cosy_alive = True
    except Exception:
        print(f"⚠ CosyVoice server({COSY_HOST}:{COSY_PORT})沒回應,語音會 fallback 到 Gemini/edge")

if XAI_API_KEY:
    _search_note = "+網路搜尋" if AI_WEB_SEARCH else ""
    if AI_TTS:
        if AI_TTS_ENGINE == "eleven" and ELEVEN_API_KEY:
            _out_note = "語音:ElevenLabs Flash(streaming)"
        elif AI_TTS_ENGINE == "cosy" and _cosy_alive:
            _out_note = "語音:CosyVoice 本地(streaming)"
        elif AI_TTS_ENGINE == "gemini" and GEMINI_API_KEY:
            _out_note = f"語音:Gemini {GEMINI_TTS_VOICE}"
        elif _HAS_TTS:
            _out_note = "語音:edge-tts 曉臻"
        else:
            _out_note = "⚠ 無可用 TTS,改貼文字"
    elif AI_TTS and not _HAS_TTS:
        _out_note = "⚠ 未裝 edge-tts,改貼文字"
    else:
        _out_note = "文字貼上"
    print(f"Grok AI 模式已就緒({XAI_MODEL}{_search_note};{_out_note})。")
else:
    print("⚠ 未設定 XAI_API_KEY,AI 模式停用。")

print(f"按【{HOTKEY}】開始說話,再按一次轉錄。")
print(f"按【{AI_MODIFIER} + {HOTKEY}】進入 AI 模式。")
print(f"按【{AUTO_MODIFIER} + {HOTKEY}】切換全自動模式(每 {AUTO_INTERVAL}s 自動轉錄 + AI 回應)。")
print("結束請按 Ctrl+C。")

# ─────────────────────── 錄音串流 ────────────────────────
_frames    = []
_recording = False
_ai_mode   = False          # True = 本次錄音是 AI 問答模式
_ai_context = ""            # 錄音開始時的剪貼簿快照
_lock      = threading.Lock()
_timeout_timer = None
_session_id = 0             # 每次按熱鍵開始錄音 +1;舊 worker 看到 mismatch 就放棄

# 全自動模式狀態
_auto_mode      = False        # 目前是否在自動模式
_auto_recording = False        # True=正在錄音階段;False=正在 process/念出(暫停收音)
_auto_frames    = []           # 自動模式的音訊緩衝
_auto_context   = ""           # 開啟時快照的剪貼簿
_auto_timer     = None         # 計時器物件

def _audio_callback(indata, frames, time_info, status):
    if _recording:
        _frames.append(indata.copy())
    if _auto_recording:            # 只在錄音階段收音,processing/念出時暫停
        _auto_frames.append(indata.copy())

_stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
    callback=_audio_callback,
)
_stream.start()

# ─────────────────────── 輸出文字 ────────────────────────
# 用 SendInput 直接送 Unicode 字元 = 模擬鍵盤打字,完全不碰剪貼簿(支援中文)。
# 注意:union 必須含 MOUSEINPUT,否則 sizeof(INPUT) 對不上 → SendInput 靜默失敗。
_ULONG_PTR = ctypes.c_size_t   # 指標大小的無號整數(x64=8, x86=4)

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR)]

class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]

class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

_INPUT_KEYBOARD    = 1
_KEYEVENTF_KEYUP   = 0x0002
_KEYEVENTF_UNICODE = 0x0004

_SendInput = ctypes.windll.user32.SendInput
_SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_SendInput.restype  = wintypes.UINT

# IME 相關:中文輸入法會攔截 SendInput Unicode 事件,把標點丟到最後。
# 解法:打字前把 IME 從前景視窗暫時解綁,打完再還原。
_user32 = ctypes.windll.user32
_imm32  = ctypes.windll.imm32
_imm32.ImmAssociateContext.restype  = ctypes.c_void_p
_imm32.ImmAssociateContext.argtypes = (wintypes.HWND, ctypes.c_void_p)
_user32.GetForegroundWindow.restype = wintypes.HWND

def _type_unicode(text: str):
    """逐字以 Unicode 事件送出(不經剪貼簿)。
    打字期間暫時解綁中文 IME,避免標點順序錯亂。處理 BMP 外字元的代理對。"""
    units = []
    for ch in text:
        b = ch.encode("utf-16-le")
        for i in range(0, len(b), 2):
            units.append(b[i] | (b[i + 1] << 8))
    cb   = ctypes.sizeof(_INPUT)
    hwnd = _user32.GetForegroundWindow()
    # 解綁 IME(回傳原本的 HIMC,稍後還原)
    old_himc = _imm32.ImmAssociateContext(hwnd, None) if hwnd else None
    try:
        for unit in units:
            for flags in (_KEYEVENTF_UNICODE, _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP):
                inp = _INPUT()
                inp.type = _INPUT_KEYBOARD
                inp.u.ki = _KEYBDINPUT(0, unit, flags, 0, 0)
                _SendInput(1, ctypes.byref(inp), cb)
    finally:
        # 還原 IME 綁定,使用者下次自己打字時 IME 照常運作
        if hwnd and old_himc:
            _imm32.ImmAssociateContext(hwnd, old_himc)

def _clipboard_paste(text: str, restore: bool):
    """經剪貼簿貼上;restore=True 則貼完還原原本內容。"""
    old = ""
    if restore:
        try:    old = pyperclip.paste()
        except: old = ""
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    if restore:
        # 長文要多等一下讓 Ctrl+V 真的消耗完剪貼簿,再還原
        time.sleep(max(0.3, len(text) * 0.001))
        try:    pyperclip.copy(old)
        except: pass

def _paste_text(text: str):
    """聽寫輸出走 OUTPUT_MODE 設定(預設 type)。"""
    if OUTPUT_MODE == "type":
        _type_unicode(text)
        return
    _clipboard_paste(text, RESTORE_CLIPBOARD)

def _output_ai_text(text: str):
    """AI 回覆輸出走 AI_OUTPUT_MODE(預設 clipboard,避免中文標點被 IME 攔截錯位)。"""
    if AI_OUTPUT_MODE == "type":
        _type_unicode(text)
        return
    _clipboard_paste(text, AI_RESTORE_CLIPBOARD)

# 語音用 audio tags(只有 ElevenLabs v3 看得懂),例:[laughs] [sighs] [whispers]
_AUDIO_TAG_RE = re.compile(
    r"\[(laughs?|laughter|giggles?|chuckles?|sighs?|exhales?|breathes?|"
    r"whispers?|clears throat|gasps?|hmm+|sniffs?)\]",
    re.IGNORECASE,
)

# ─────────────────────── 文字清理 ────────────────────────
def _clean_for_typing(text: str) -> str:
    """打字輸出前:移除 markdown 記號、audio tags、把換行收成空格。
    (多行 + 換行用模擬打字送進輸入框會造成游標亂跳、順序顛倒,攤平成單行最穩。)"""
    text = _AUDIO_TAG_RE.sub("", text)                       # 拿掉 [laughs] 等(不要打進輸入框)
    text = re.sub(r"\[\[\d+\]\]\([^)]*\)", "", text)          # [[1]](url) 引用標記
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # [文字](url) -> 文字
    text = re.sub(r"^[ \t]*#{1,6}\s*", "", text, flags=re.M)  # 標題 #
    text = re.sub(r"^[ \t]*[-*+]\s+", "", text, flags=re.M)   # 項目符號
    text = re.sub(r"^[ \t]*\d+\.\s+", "", text, flags=re.M)   # 編號清單
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = re.sub(r"-{3,}", "", text)                         # --- 分隔線
    text = re.sub(r"\s*\n+\s*", " ", text)                    # 換行 -> 空格
    text = re.sub(r"[ \t]{2,}", " ", text)                    # 多空格收斂
    return text.strip()

# ─────────────────────── 台灣腔語音(TTS)─────────────────
def _clean_for_speech(text: str) -> str:
    """念出來前移除 markdown / 引用網址,避免 TTS 把連結念出來。"""
    # [[1]](http...)、[文字](http...) → 只留文字(citation 標記直接拿掉)
    text = re.sub(r"\[\[\d+\]\]\([^)]*\)", "", text)        # [[1]](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)    # [文字](url)
    text = re.sub(r"https?://\S+", "", text)                # 裸網址
    text = text.replace("**", "").replace("*", "").replace("`", "").replace("#", "")
    # 移除 emoji / 表情符號 / 變體選擇符(念出來會很怪)
    text = re.sub(
        r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
        r"\U0001F1E6-\U0001F1FF\U0000FE00-\U0000FE0F\U0000200D]",
        "", text,
    )
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

_mci = ctypes.windll.winmm.mciSendStringW

def _stop_speech():
    """立刻停止任何播放中的 TTS(打斷用)。安全:沒在播也不會出錯。"""
    try:
        _mci("stop aitts", None, 0, None)
        _mci("close aitts", None, 0, None)
    except Exception:
        pass

_NO_CANCEL = lambda: False     # 預設「不會被打斷」的 cancel 探測器

def _play_wav_interruptible(path: str, file_type: str, cancelled):
    """開檔 + 播放;在 open 前最後一次檢查 cancel,避免「過時的回覆」也被播。
    play 期間如有人按熱鍵,_stop_speech() 會送 MCI stop,本函式就會返回。"""
    with _tts_lock:
        if cancelled():
            return False
        _mci(f'open "{path}" type {file_type} alias aitts', None, 0, None)
        try:
            if cancelled():               # open 完到 play 之間再 check 一次
                return False
            _mci("play aitts wait", None, 0, None)
            return True
        finally:
            _mci("close aitts", None, 0, None)

def _speak_edge(text: str, cancelled=_NO_CANCEL):
    """edge-tts → mp3 → MCI 播放(免費 fallback,音色較機械)。"""
    async def _gen():
        await edge_tts.Communicate(
            text, AI_TTS_VOICE, rate=AI_TTS_RATE, pitch=AI_TTS_PITCH
        ).save(_TTS_MP3)
    asyncio.run(_gen())
    if cancelled():                       # 生成完到播放之間 check
        return
    _play_wav_interruptible(_TTS_MP3, "mpegvideo", cancelled)

def _speak_gemini(text: str, cancelled=_NO_CANCEL):
    """Gemini 3.1 Flash TTS → PCM → 包成 WAV → MCI 播放(較自然)。"""
    voice = random.choice(GEMINI_TTS_VOICES) if GEMINI_TTS_VOICES else GEMINI_TTS_VOICE
    print(f"    (聲音:{voice})")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": GEMINI_TTS_STYLE + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }
    r = requests.post(url, params={"key": GEMINI_API_KEY},
                      headers={"Content-Type": "application/json"},
                      data=json.dumps(body), timeout=60)
    r.raise_for_status()
    if cancelled():                       # HTTP 回來但已被打斷 → 不播
        return
    part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    pcm  = base64.b64decode(part["data"])
    rate = 24000
    for kv in part.get("mimeType", "").split(";"):
        kv = kv.strip()
        if kv.startswith("rate="):
            rate = int(kv.split("=", 1)[1])
    with wave.open(_TTS_WAV, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm)
    if cancelled():
        return
    _play_wav_interruptible(_TTS_WAV, "waveaudio", cancelled)

def _cosy_chunks(text: str):
    """打 cosy server,正確解析 HTTP/1.1 chunked encoding,yield raw PCM bytes。
    用 raw socket 是因為 Python 3.14 的 requests/urllib stream 接收會額外緩衝。"""
    body = json.dumps({
        "text":     text,
        "voice":    COSY_VOICE,
        "speed":    COSY_SPEED,
        "instruct": COSY_INSTRUCT,
    }, ensure_ascii=False).encode("utf-8")
    req = (
        f"POST /tts HTTP/1.1\r\nHost: {COSY_HOST}:{COSY_PORT}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii") + body
    sock = socket.socket()
    sock.settimeout(60)
    sock.connect((COSY_HOST, COSY_PORT))
    sock.sendall(req)
    try:
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError("cosy: connection closed before headers")
            buf += chunk
        hdr_end  = buf.index(b"\r\n\r\n") + 4
        headers  = buf[:hdr_end].decode("latin1")
        if "200" not in headers.split("\r\n", 1)[0]:
            raise RuntimeError(f"cosy: {headers.split(chr(13)+chr(10),1)[0]}")
        local = buf[hdr_end:]
        is_chunked = "transfer-encoding: chunked" in headers.lower()

        def _read_until(sep):
            nonlocal local
            while sep not in local:
                c = sock.recv(8192)
                if not c: return None
                local += c
            idx = local.index(sep); out = local[:idx]
            local = local[idx + len(sep):]
            return out

        def _read_exact(n):
            nonlocal local
            while len(local) < n:
                c = sock.recv(8192)
                if not c: return None
                local += c
            out = local[:n]; local = local[n:]
            return out

        if is_chunked:
            while True:
                size_line = _read_until(b"\r\n")
                if size_line is None: return
                size = int(size_line.split(b";", 1)[0], 16)
                if size == 0: return
                data = _read_exact(size)
                if data is None: return
                _read_exact(2)        # 吃掉 \r\n trailer
                yield data
        else:
            if local: yield local
            while True:
                c = sock.recv(8192)
                if not c: return
                yield c
    finally:
        try: sock.close()
        except: pass

def _speak_cosy(text: str, cancelled=_NO_CANCEL):
    """CosyVoice 本地 server → streaming PCM → sounddevice 即時播放。"""
    stream = sd.OutputStream(samplerate=COSY_RATE, channels=1, dtype="int16")
    stream.start()
    leftover = b""
    try:
        for data in _cosy_chunks(text):
            if cancelled():
                return
            blob = leftover + data
            even = len(blob) - (len(blob) % 2)
            if even:
                stream.write(np.frombuffer(blob[:even], dtype=np.int16))
            leftover = blob[even:]
    finally:
        time.sleep(0.15)              # 讓 buffer 放完
        try: stream.stop()
        except: pass
        try: stream.close()
        except: pass

def _speak_eleven(text: str, cancelled=_NO_CANCEL):
    """ElevenLabs → streaming PCM → sounddevice 即時播放。"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
    params = {"output_format": f"pcm_{ELEVEN_RATE}"}
    # v3 不支援 optimize_streaming_latency;flash 系列才用
    if "flash" in ELEVEN_MODEL or "turbo" in ELEVEN_MODEL:
        params["optimize_streaming_latency"] = 4
    body = {
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {
            "stability": ELEVEN_STABILITY,
            "similarity_boost": ELEVEN_SIMILARITY,
            "use_speaker_boost": True,
        },
    }
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}

    last_err = None
    for attempt in range(2):          # abuse detector 偶發 401,retry 一次
        r = requests.post(url, params=params, headers=headers,
                          data=json.dumps(body), stream=True, timeout=60)
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}: {r.text[:150]}"
            r.close()
            continue
        stream = sd.OutputStream(samplerate=ELEVEN_RATE, channels=1, dtype="int16")
        stream.start()
        leftover = b""
        try:
            for chunk in r.iter_content(chunk_size=None):
                if cancelled():
                    return
                if not chunk:
                    continue
                blob = leftover + chunk
                even = len(blob) - (len(blob) % 2)
                if even:
                    stream.write(np.frombuffer(blob[:even], dtype=np.int16))
                leftover = blob[even:]
        finally:
            time.sleep(0.15)
            try: stream.stop()
            except: pass
            try: stream.close()
            except: pass
        return
    raise RuntimeError(f"ElevenLabs 失敗: {last_err}")

def _speak(text: str, cancelled=_NO_CANCEL):
    """依設定挑引擎;失敗會 fallback 到下一個可用引擎。"""
    if AI_TTS_ENGINE == "eleven" and ELEVEN_API_KEY:
        try:
            _speak_eleven(text, cancelled)
            return
        except Exception as e:
            print(f"  ⚠ ElevenLabs 失敗,fallback 改用 edge-tts: {e}")
        _speak_edge(text, cancelled)
        return
    if AI_TTS_ENGINE == "cosy":
        try:
            _speak_cosy(text, cancelled)
            return
        except Exception as e:
            print(f"  ⚠ CosyVoice 失敗(server 沒開?),fallback: {e}")
    if AI_TTS_ENGINE == "gemini" and GEMINI_API_KEY:
        try:
            _speak_gemini(text, cancelled)
            return
        except Exception as e:
            print(f"  ⚠ Gemini TTS 失敗,fallback 改用 edge-tts: {e}")
    _speak_edge(text, cancelled)

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

def _xai_complete(instructions: str, input_msgs: list, web_search: bool) -> str:
    """呼叫 xAI Responses API,回傳最終文字。"""
    payload = {
        "model":        XAI_MODEL,
        "instructions": instructions,
        "input":        input_msgs,
    }
    if web_search:
        payload["tools"] = [{"type": "web_search"}]   # 模型自行判斷是否搜尋
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    resp = requests.post(XAI_URL, headers=headers,
                         data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    return _extract_answer(resp.json())

def _ask_llm(context: str, question: str) -> tuple[str, str]:
    """把 context + question 送到 Grok(可即時搜尋)。
    回傳 (answer, user_msg) — 不直接改記憶;由 worker 決定是否提交(打斷時就不提交)。"""
    user_msg = ""
    if context.strip():
        user_msg += f"【剪貼簿內容】\n{context.strip()}\n\n"
    user_msg += f"【問題】\n{question.strip()}"

    instructions = AI_SYSTEM_PROMPT
    if _chat_summary:
        instructions += f"\n\n【先前對話摘要(供延續參考)】\n{_chat_summary}"
    input_msgs = list(_chat_history) + [{"role": "user", "content": user_msg}]

    answer = _xai_complete(instructions, input_msgs, AI_WEB_SEARCH)
    return answer, user_msg

def _condense_history():
    """逐字歷史超過上限時,把最舊的對話併入滾動摘要(再 call 一次 LLM 壓成 ≤N 字)。"""
    global _chat_history, _chat_summary
    if len(_chat_history) <= AI_HISTORY_TURNS * 2:
        return
    keep   = AI_KEEP_RECENT * 2
    old    = _chat_history[:-keep]      # 要折疊的舊訊息
    recent = _chat_history[-keep:]      # 保留逐字的最近幾輪

    convo = ""
    for m in old:
        who = "使用者" if m["role"] == "user" else "助理"
        convo += f"{who}: {m['content']}\n"

    body = ""
    if _chat_summary:
        body += f"【先前摘要】\n{_chat_summary}\n\n"
    body += f"【要併入的對話】\n{convo}"

    instr = (
        f"把以下內容濃縮成不超過 {AI_SUMMARY_CHARS} 字的繁體中文摘要,"
        "保留重點、結論、使用者的偏好與個資、待辦事項和提到的人名,"
        "讓之後的對話能無縫延續。只輸出摘要本身,不要客套話。"
    )
    try:
        new_summary = _xai_complete(instr, [{"role": "user", "content": body}],
                                    web_search=False).strip()
        _chat_summary = new_summary
        _chat_history = recent
        print(f"  ⓘ 已壓縮歷史 → 摘要 {len(_chat_summary)} 字,保留最近 {AI_KEEP_RECENT} 輪")
    except Exception as e:
        # 壓縮失敗就退回單純丟棄最舊,避免歷史無限增長
        _chat_history = _chat_history[-AI_HISTORY_TURNS * 2:]
        print(f"  ⚠ 壓縮失敗,改丟棄最舊: {e}")

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

def _dictate_worker(audio: np.ndarray, my_session: int):
    """普通聽寫模式。被打斷(session 變)就放棄,避免打字打到新一輪錄音的視窗。"""
    try:
        t0   = time.time()
        text = _transcribe(audio)
        print(f"  → ({time.time()-t0:.1f}s) {text!r}")
        if _session_id != my_session:
            print("  ⓘ 已被新一輪打斷,丟棄。"); return
        if text:
            _paste_text(text)
        else:
            beep_error()
    except Exception as e:
        print(f"  ✗ 轉錄失敗: {e}")
        beep_error()

def _ai_worker(audio: np.ndarray, context: str, my_session: int):
    """AI 問答模式:ASR → LLM → 打字 + 念。被打斷(session 變)就靜默放棄。"""
    global _last_sent_clipboard
    def cancelled():
        return _session_id != my_session
    try:
        t0       = time.time()
        question = _transcribe(audio)
        print(f"  → ASR ({time.time()-t0:.1f}s) {question!r}")
        if cancelled():
            print("  ⓘ 已被新一輪打斷,丟棄此次回覆。"); return
        if not question:
            beep_error()
            return

        # 脈絡去重:跟上次送過的剪貼簿一樣就不重複送
        snapshot      = context
        effective_ctx = context
        if context and context == _last_sent_clipboard:
            print("  ⓘ 剪貼簿同上一次,不再重複送脈絡。")
            effective_ctx = ""

        turns = len(_chat_history) // 2
        print(f"  → 送 LLM … (脈絡 {len(effective_ctx)} 字 / 歷史 {turns} 輪)")
        t1     = time.time()
        answer, user_msg = _ask_llm(effective_ctx, question)
        print(f"  → LLM ({time.time()-t1:.1f}s) {answer!r}")
        if cancelled():
            print("  ⓘ 已被新一輪打斷,丟棄此次回覆。"); return

        # 提交本輪到記憶(打斷前不會跑到這,所以記憶乾淨)
        _chat_history.append({"role": "user",      "content": user_msg})
        _chat_history.append({"role": "assistant", "content": answer})
        _last_sent_clipboard = snapshot   # 記住這次的剪貼簿,下次比對

        # 永遠先輸出文字(攤平成單行純文字),再念出來
        _output_ai_text(_clean_for_typing(answer))
        print("  ✓ 已輸出文字。")
        # 任一 TTS 引擎可用就念
        _tts_available = (
            (AI_TTS_ENGINE == "eleven" and ELEVEN_API_KEY)
            or AI_TTS_ENGINE == "cosy"
            or (AI_TTS_ENGINE == "gemini" and GEMINI_API_KEY)
            or _HAS_TTS
        )
        if AI_TTS and _tts_available:
            print("  → 念出回覆中…")
            try:
                _speak(_clean_for_speech(answer), cancelled=cancelled)
                print("  ✓ 已念出。" if not cancelled() else "  ⓘ 念到一半被打斷。")
            except Exception as e:
                print(f"  ⚠ 語音失敗: {e}")
        else:
            beep_ai_done()

        # 答案已交付,最後才壓縮歷史(不影響回覆速度)
        if not cancelled():
            _condense_history()
    except Exception as e:
        print(f"  ✗ AI 模式失敗: {e}")
        beep_error()

# ─────────────────────── 熱鍵邏輯 ────────────────────────
def _start_recording(ai: bool):
    global _recording, _frames, _timeout_timer, _ai_mode, _ai_context, _session_id
    # 打斷任何正在播放/排隊的 TTS,並讓任何進行中的 worker 失效
    _stop_speech()
    with _lock:
        if _recording:
            return
        _session_id += 1               # 新一輪:舊 worker 看到 mismatch 會放棄
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
    sid = _session_id
    print(f"■ 停止,長度 {dur:.1f}s,{'AI 問答' if ai else '轉錄'}中…")
    if ai:
        threading.Thread(target=_ai_worker,      args=(audio, ctx, sid), daemon=True).start()
    else:
        threading.Thread(target=_dictate_worker, args=(audio, sid),      daemon=True).start()

def _toggle(ai: bool):
    if _recording:
        _stop_recording()
    else:
        _start_recording(ai)

# ─────────────────────── 全自動模式 ──────────────────────
def _auto_start_recording():
    """開始這一輪的錄音階段,錄滿 AUTO_INTERVAL 秒後自動 tick。"""
    global _auto_recording, _auto_frames, _auto_timer
    if not _auto_mode:
        return
    _auto_frames = []
    _auto_recording = True
    print(f"  [auto] 🎙 開始錄音({AUTO_INTERVAL}s)…")
    _auto_timer = threading.Timer(AUTO_INTERVAL, _auto_tick)
    _auto_timer.daemon = True
    _auto_timer.start()

def _auto_tick():
    """錄音時間到:停止收音 → 處理 → 處理完再開下一輪。"""
    global _auto_recording, _auto_frames
    if not _auto_mode:
        return
    # 停止收音(processing 期間靜默)
    _auto_recording = False
    frames = list(_auto_frames)
    _auto_frames = []
    # 每輪開始 process 時重新快照剪貼簿,讓新複製的內容馬上生效
    try:    _auto_context = pyperclip.paste()
    except: pass

    if not frames:
        _auto_start_recording(); return
    audio = np.concatenate(frames, axis=0).flatten().astype(np.float32)
    dur = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < AUTO_MIN_RMS:
        print(f"  [auto] 靜音跳過 rms={rms:.4f} < {AUTO_MIN_RMS}")
        _auto_start_recording(); return
    beep_auto_tick()
    print(f"  [auto] 處理 {dur:.1f}s rms={rms:.4f}")
    # 在背景 worker 做 ASR+LLM+TTS;worker 結束後再開下一輪
    threading.Thread(target=_auto_worker,
                     args=(audio, _auto_context), daemon=True).start()

def _auto_ask_openrouter(context: str, question: str) -> str:
    """全自動模式用 OpenRouter(便宜模型)回答,不帶完整歷史、只帶摘要。"""
    user_msg = ""
    if context.strip():
        user_msg += f"【剪貼簿內容】\n{context.strip()}\n\n"
    user_msg += f"【問題/說話內容】\n{question.strip()}"
    # system prompt 跟主模式一樣,只是走不同 endpoint
    sys = AI_SYSTEM_PROMPT
    if _chat_summary:
        sys += f"\n\n【先前對話摘要】\n{_chat_summary}"
    messages = [{"role": "system", "content": sys}]
    # 只帶最近 5 輪逐字(不帶全部 15 輪,省 token)
    messages += list(_chat_history)[-10:]
    messages.append({"role": "user", "content": user_msg})
    resp = requests.post(
        AUTO_LLM_URL,
        headers={"Authorization": f"Bearer {AUTO_LLM_KEY}",
                 "Content-Type": "application/json"},
        data=json.dumps({"model": AUTO_LLM_MODEL, "messages": messages}),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip(), user_msg

def _auto_worker(audio: np.ndarray, context: str):
    """全自動模式的 ASR → OpenRouter → TTS worker。
    做完之後(不論成功失敗)自動開下一輪錄音。"""
    try:
        question = _transcribe(audio)
        print(f"  [auto] ASR: {question!r}")
        if not question.strip():
            return
        print(f"  [auto] 送 {AUTO_LLM_MODEL} …")
        answer, user_msg = _auto_ask_openrouter(context, question)
        print(f"  [auto] AI: {answer!r}")
        _chat_history.append({"role": "user",      "content": user_msg})
        _chat_history.append({"role": "assistant", "content": answer})
        _output_ai_text(_clean_for_typing(answer))
        if AI_TTS:
            _speak(_clean_for_speech(answer))   # 念完才繼續
        _condense_history()
    except Exception as e:
        print(f"  [auto] 錯誤: {e}")
        beep_error()
    finally:
        # 無論成功或失敗,都開下一輪(如果還在自動模式)
        if _auto_mode:
            time.sleep(0.3)          # 短暫停頓避免直接接上
            _auto_start_recording()

def _start_auto_mode():
    global _auto_mode, _auto_recording, _auto_context
    _auto_mode = True
    _auto_recording = False
    if AUTO_CONTEXT:
        try:    _auto_context = pyperclip.paste()
        except: _auto_context = ""
    beep_auto_on()
    print(f"★★ 全自動模式開啟(錄 {AUTO_INTERVAL}s → process → 念完 → 再錄)。再按 Left Alt + Copilot 關閉。")
    _auto_start_recording()    # 開第一輪錄音

def _stop_auto_mode():
    global _auto_mode, _auto_recording, _auto_timer
    _auto_mode      = False
    _auto_recording = False
    if _auto_timer:
        _auto_timer.cancel()
        _auto_timer = None
    beep_auto_off()
    print("★★ 全自動模式關閉。")

def _toggle_auto():
    if _auto_mode:
        _stop_auto_mode()
    else:
        _start_auto_mode()

# 自己追蹤左右 Alt 的狀態,不用 is_pressed(Windows AltGr 會讓兩個同時為 True)
_lalt_down = False
_ralt_down = False

def _on_lalt(e):
    global _lalt_down
    _lalt_down = (e.event_type == keyboard.KEY_DOWN)

def _on_ralt(e):
    global _ralt_down
    _ralt_down = (e.event_type == keyboard.KEY_DOWN)

keyboard.hook_key("left alt",  _on_lalt)
keyboard.hook_key("right alt", _on_ralt)

def _on_hotkey(event):
    if event.event_type != "down":
        return
    # 用自追蹤的狀態,右 Alt 優先
    if _ralt_down:
        if not XAI_API_KEY:
            print("⚠ 未設定 XAI_API_KEY,AI 模式無法使用。"); beep_error(); return
        _toggle(ai=True)
    elif _lalt_down:
        if not XAI_API_KEY:
            print("⚠ 未設定 XAI_API_KEY,無法使用自動模式。"); beep_error(); return
        _toggle_auto()
    else:
        _toggle(ai=False)

keyboard.hook_key(HOTKEY, _on_hotkey, suppress=HOTKEY_SUPPRESS)

# ─────────────────────── 主迴圈 ──────────────────────────
try:
    keyboard.wait()
except KeyboardInterrupt:
    pass
finally:
    _stream.stop()
    _stream.close()
    print("\n已結束。")
