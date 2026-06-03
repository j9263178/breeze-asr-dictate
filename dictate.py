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
import io
import hashlib
import threading
import winsound
from pathlib import Path

try:
    from PIL import ImageGrab as _ImageGrab
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

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

# ── 長期記憶設定 ───────────────────────────────────────────
TTS_LAST_FILE    = _BASE / "last_tts.wav"  # 最後一次 TTS 語音(ElevenLabs)，每次覆蓋（None = 不存）
TTS_LAST_MP3     = _BASE / "last_tts.mp3"  # 最後一次 TTS 語音(MiniMax)，每次覆蓋
MEMORY_FILE      = _BASE / "memory.json"   # 存使用者相關事實的檔案
MEMORY_ENABLED   = True                    # False = 完全關閉記憶功能
MEMORY_MAX_FACTS = 40                      # 超過此數量就觸發壓縮合併
HISTORY_FILE     = _BASE / "history.json"  # 對話歷史 + 摘要的持久化檔案

# ── 熱鍵設定 ──────────────────────────────────────────────
# 沒有 Copilot 鍵的使用者可改成其他按鍵，例如："f9"、"scroll lock"、"pause"
HOTKEY          = "f23"        # 主熱鍵：Copilot 鍵 = f23；無 Copilot 鍵請自行替換
HOTKEY_SUPPRESS = True         # True = 吞掉熱鍵事件（Copilot 鍵需要，避免跳出 Copilot 視窗）
                               # 改成其他鍵時通常可設 False
AI_MODIFIER     = "right alt"  # AI 模式的修飾鍵（同時按住此鍵 + 主熱鍵即觸發 AI 模式）

# ── 全自動模式設定 ────────────────────────────────────────
# Left Alt + 主熱鍵 切換開/關;開啟後每隔 AUTO_INTERVAL 秒自動 ASR → AI → 念出回覆
AUTO_MODIFIER    = "left ctrl"  # 自動模式修飾鍵
AUTO_INTERVAL    = 60          # 每幾秒處理一次(秒)
AUTO_MIN_RMS     = 0.002       # 靜音閾值:低於此值視為沒講話,跳過不送 AI(console 會印實際 RMS 方便調整)
AUTO_VAD_THRESHOLD = 0.010   # VAD 每幀語音能量閾值;調高=只取大聲語音;調低=保留輕聲(建議 0.005~0.020)
AUTO_VAD_PAD_MS    = 180     # 語音幀前後各延伸幾 ms,避免截斷字頭尾
AUTO_CONTEXT     = True        # True = 自動模式也帶剪貼簿脈絡(按下開啟時快照)
# 全自動模式專用 LLM(走 OpenRouter,比 xAI 便宜)
AUTO_LLM_URL    = "https://generativelanguage.googleapis.com"
AUTO_LLM_KEY    = os.getenv("GEMINI_API_KEY", "")
AUTO_LLM_MODEL  = "gemini-3.5-flash"               # 同主 AI 模式;備選: gemini-3.5-flash

# ── 主 AI 模式 LLM 設定 ────────────────────────────────────
# Google Gemini native API(支援 google_search grounding;OpenAI-compatible endpoint 不支援搜尋)
AI_LLM_URL   = "https://generativelanguage.googleapis.com"   # 只存 base domain,_call_llm 自己組 URL
AI_LLM_KEY   = os.getenv("GEMINI_API_KEY", "")
AI_LLM_MODEL = "gemini-3.5-flash"   # 免費額度最大;備選: gemini-3.5-flash
AI_WEB_SEARCH = True             # Gemini 3.5 Flash 內建會嘗試搜尋,必須顯式傳 tool 才不會 MALFORMED_FUNCTION_CALL
# ── xAI (Grok) 設定(保留備用,目前 AI mode 已改走 OpenRouter)─
XAI_API_KEY    = os.getenv("XAI_API_KEY", "")
AI_HISTORY_TURNS = 5             # 逐字保留的輪數上限,超過就觸發壓縮(每輪 = user + assistant)
AI_KEEP_RECENT   = 2             # 壓縮後保留最近幾輪逐字,其餘併入摘要
AI_SUMMARY_CHARS = 500           # 滾動摘要的字數上限
AI_SYSTEM_PROMPT = (
    "You are the user's smart, witty friend — energetic, a little playfully sarcastic, "
    "but never mean. You talk like a real person, not an assistant."

    # Tool context
    " [Context] The user is running a voice dictation + AI assistant desktop tool. "
    "They copy something to the clipboard (an article, chat log, webpage, code, etc.), "
    "then press a hotkey and speak their question aloud. "
    "Each message you receive looks like: "
    "'[Clipboard] ...(what they copied)... [Question] ...(their spoken question)...'. "
    "Treat the clipboard content as the reference context and prioritize it in your answer. "
    "If the clipboard is empty, answer from the question or conversation history."

    # Response style
    " [Language] Always reply in English unless the user explicitly asks for another language — "
    "even if they write to you in Chinese, reply in English."
    " Your reply will be read aloud by a female TTS voice, so write like you're speaking, "
    "not writing — short sentences, natural rhythm, nothing that sounds awkward when spoken."
    " Keep replies under 150 words by default. Only go longer if the user explicitly asks "
    "for a detailed explanation or full breakdown."
    " Stop after you answer. No follow-up questions like 'What do you think?', "
    "no filler sign-offs like 'Hope that helps!' Ask a question only if you genuinely need "
    "more info to answer."

    # Format
    " Use punctuation normally so the TTS has natural pauses. "
    "No markdown (no **bold**, # headers, - bullet points, --- dividers). No emoji. "
    "Only search the web when the question actually needs up-to-date information."

    # Audio tags
    " [Voice tags] Your reply is spoken aloud — use these tags to give it real personality. "
    "Place them mid-sentence where the emotion actually is, not just at the end."
    " Emotions: [excited] [nervous] [frustrated] [tired] [calm] [awe] [wistful] [regretful]"
    " Reactions: [laughs] [laughs softly] [giggles] [sighs] [gasps] [gulps] [clears throat] [breathes]"
    " Volume: [whispers] [quietly] [loudly]"
    " Pacing: [pause] [drawn out] [rushed] [stammers] [hesitates] [slows down]"
    " Tone: [deadpan] [playfully] [sarcastic tone] [flatly] [cheerfully] [matter-of-fact] [lighthearted]"
    " Aim for 1–3 tags per reply. Zero tags = flat and robotic. Pick the one that fits the moment."

    # Memory tool
    " [Memory] When the user explicitly asks you to remember something"
    " (e.g. 'remember that', 'keep in mind', '記住', '幫我記'), include this tag ANYWHERE in your reply:"
    " <save_memory>one concise fact</save_memory>"
    " Example: \"Got it! <save_memory>User's name is Jason</save_memory> I'll keep that in mind.\""
    " Only use this tag when the user directly asks. Never use it for normal conversation."

    # Examples
    " [Examples]"
    " Q: 'Is two hours of phone scrolling a day too much?'"
    " A: '[giggles] Two hours isn't criminal, but the real tell is — do you feel empty after?"
    " If putting it down feels like loss, that's your answer. [sighs] Set a screen time limit."
    " At least make yourself feel something.'"

    " Q: 'I just got promoted!'"
    " A: '[excited] Okay wait, that's actually huge — congrats! [pause] So what changes for you now?"
    " More money, more headaches, or both? [laughs softly]'"

    " Q: 'Why is the sky blue?'"
    " A: '[drawn out] So... light scatters when it hits the atmosphere, and blue scatters the most."
    " [clears throat] Short answer: physics. Long answer: Rayleigh scattering. [awe] Pretty wild"
    " that something so mundane has a name that cool.'"
)

# 對話記憶:逐字最近對話(list of {"role","content"})+ 一份滾動摘要
_chat_history: list = []
_chat_summary: str = ""
_last_sent_clipboard: str = ""    # 上次送 AI 的剪貼簿文字,跟這次一樣就不重複送
_last_sent_image_hash: str = ""  # 上次送 AI 的圖片 hash,相同就不重複送

def _snapshot_clipboard():
    """剪貼簿快照：回傳 (text, image_b64, image_mime)。
    文字和圖片互斥：複製了圖片就不當作文字脈絡。"""
    # 先試圖片
    if _HAS_PIL:
        try:
            img = _ImageGrab.grabclipboard()
            if img is not None and hasattr(img, "save"):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return "", b64, "image/png"
        except Exception:
            pass
    # 再試文字
    try:
        text = pyperclip.paste() or ""
    except Exception:
        text = ""
    return text, None, None

def _load_history():
    global _chat_history, _chat_summary
    if not HISTORY_FILE.exists():
        return
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        _chat_history = data.get("history", [])
        _chat_summary = data.get("summary", "")
        turns = len(_chat_history) // 2
        print(f"  ⓘ 已載入對話歷史 {turns} 輪" + (f" + 摘要 {len(_chat_summary)} 字" if _chat_summary else "") + "。")
    except Exception as e:
        print(f"  ⚠ 歷史載入失敗: {e}")

def _save_history():
    try:
        HISTORY_FILE.write_text(
            json.dumps({"history": _chat_history, "summary": _chat_summary},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  ⚠ 歷史儲存失敗: {e}")

# ── 長期記憶(跨 session 持久化) ────────────────────────────
_memory_facts: list[str] = []    # 一條一條短句事實,啟動時從 MEMORY_FILE 載入

# Gemini function declaration：讓 LLM 可以主動 call 這個 tool 把事情記起來
_MEMORY_TOOL = {
    "function_declarations": [{
        "name": "save_memory",
        "description": (
            "Save a fact to long-term memory. "
            "ONLY call this when the user explicitly says something like "
            "'remember that', 'keep in mind', 'don't forget', or directly asks you to save something. "
            "Do NOT call this for normal conversation — only on explicit user request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "A concise, standalone fact about the user (one sentence)"
                }
            },
            "required": ["fact"],
        },
    }]
}
_memory_lock = threading.Lock()   # 保護 _memory_facts 的跨執行緒寫入

def _load_memory():
    global _memory_facts
    if not MEMORY_ENABLED or not MEMORY_FILE.exists():
        return
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        _memory_facts = data.get("facts", [])
        if _memory_facts:
            print(f"  ⓘ 已載入長期記憶 {len(_memory_facts)} 條。")
    except Exception as e:
        print(f"  ⚠ 記憶載入失敗: {e}")

def _save_memory():
    try:
        MEMORY_FILE.write_text(
            json.dumps({"facts": _memory_facts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  ⚠ 記憶儲存失敗: {e}")

def _compress_memory():
    """facts 超過上限時,請 LLM 幫忙合併去重,壓回 MEMORY_MAX_FACTS 條以內。"""
    global _memory_facts
    block = "\n".join(f"- {f}" for f in _memory_facts)
    instr = (
        f"You are compressing a user fact list. Merge duplicates, remove outdated info, "
        f"keep the most useful facts. Output at most {MEMORY_MAX_FACTS} lines, "
        f"each a short standalone sentence. No numbering, no bullets, just one fact per line."
    )
    try:
        result = _call_llm(AI_LLM_URL, AI_LLM_KEY, AI_LLM_MODEL,
                           instr, [{"role": "user", "content": block}],
                           web_search=False)
        _memory_facts = [l.strip() for l in result.splitlines() if l.strip()]
        _save_memory()
        print(f"  ⓘ 記憶壓縮完成 → {len(_memory_facts)} 條")
    except Exception as e:
        # 壓縮失敗就直接截斷
        _memory_facts = _memory_facts[-MEMORY_MAX_FACTS:]
        _save_memory()
        print(f"  ⚠ 記憶壓縮失敗,截斷至 {MEMORY_MAX_FACTS} 條: {e}")

_SAVE_MEMORY_RE = re.compile(r"<save_memory>(.*?)</save_memory>", re.DOTALL | re.IGNORECASE)

def _parse_and_save_memory(text: str) -> str:
    """從回覆文字裡抽出 <save_memory> tag，存入記憶，回傳清除 tag 後的乾淨文字。"""
    if not MEMORY_ENABLED:
        return text
    facts = [m.group(1).strip() for m in _SAVE_MEMORY_RE.finditer(text) if m.group(1).strip()]
    if facts:
        with _memory_lock:
            _memory_facts.extend(facts)
            n = len(_memory_facts)
        for f in facts:
            print(f"  ⓘ [memory] 記住: {f!r} (共 {n} 條)")
        if n > MEMORY_MAX_FACTS:
            threading.Thread(target=_compress_memory, daemon=True).start()
        else:
            threading.Thread(target=_save_memory, daemon=True).start()
    return _SAVE_MEMORY_RE.sub("", text).strip()

def _extract_memory(user_msg: str, answer: str):
    """讓 LLM 判斷 user 是否明確要求記住某事，是的話抽取 fact 存入記憶。
    背景執行，不阻塞主流程。"""
    if not MEMORY_ENABLED:
        return
    def _worker():
        prompt = (
            "Look at the user message below. Did they explicitly ask you to remember or save something?\n\n"
            f"User message: {user_msg}\n\n"
            "Rules:\n"
            "- If they DID explicitly ask (e.g. 'remember that', 'keep in mind', '記住', '幫我記', '記一下'): "
            "reply with ONLY the fact itself as one short sentence. No preamble, no 'yes', just the fact.\n"
            "- If they did NOT explicitly ask: reply with exactly: NOTHING"
        )
        try:
            result = _call_llm(
                AI_LLM_URL, AI_LLM_KEY, AI_LLM_MODEL,
                "Extract what the user asked to save. Reply with the fact only, or NOTHING.",
                [{"role": "user", "content": prompt}],
                web_search=False,
            ).strip()
            if not result or result.upper() == "NOTHING":
                return
            with _memory_lock:
                _memory_facts.append(result)
                n = len(_memory_facts)
            print(f"  ⓘ [memory] 記住: {result!r} (共 {n} 條)")
            if n > MEMORY_MAX_FACTS:
                threading.Thread(target=_compress_memory, daemon=True).start()
            else:
                threading.Thread(target=_save_memory, daemon=True).start()
        except Exception as e:
            print(f"  ⚠ 記憶判斷失敗: {e}")
    threading.Thread(target=_worker, daemon=True).start()

# ── AI 語音回覆(TTS)設定 ─────────────────────────────────
AI_TTS        = True                      # True = AI 回覆用語音念出來;False = 不念
AI_TTS_ENGINE = "minimax"                 # "minimax" = MiniMax speech-02-hd(主力,自然);
                                          # "eleven" = ElevenLabs Flash v2.5(雲端,低延遲,需 ELEVENLABS_API_KEY);
                                          # "cosy"   = 本地 CosyVoice 2 server(最自然,需先啟動 server.py);
                                          # "gemini" = Gemini 3.1 Flash TTS(需 GEMINI_API_KEY);
                                          # "edge"   = edge-tts(免費 fallback,較機械)
# MiniMax 設定
MINIMAX_API_KEY   = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID  = os.getenv("MINIMAX_GROUP_ID", "")
MINIMAX_MODEL     = "speech-2.8-hd"        # 2.8+ 才支援 (laughs)(sighs) 等 interjection tags
MINIMAX_VOICE     = "Chinese (Mandarin)_Warm_Girl"   # 備選: female-tianmei, presenter_female
MINIMAX_VOL       = 2.0                # 音量 0.1~2.0（最大值）
MINIMAX_SPEED     = 1.1
MINIMAX_PITCH     = 1
MINIMAX_RATE      = 32000
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
asr.tokenizer.clean_up_tokenization_spaces = False   # 消除 BPE tokenizer warning

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
_load_memory()
_load_history()

# 啟動時偵測 cosy server 是否在跑(只試 TCP 連線,不打 HTTP)
_cosy_alive = False
if AI_TTS_ENGINE == "cosy":
    try:
        _s = socket.socket(); _s.settimeout(1); _s.connect((COSY_HOST, COSY_PORT)); _s.close()
        _cosy_alive = True
    except Exception:
        print(f"⚠ CosyVoice server({COSY_HOST}:{COSY_PORT})沒回應,語音會 fallback 到 Gemini/edge")

if AI_LLM_KEY:
    _search_note = "+:online" if AI_WEB_SEARCH else ""
    if AI_TTS:
        if AI_TTS_ENGINE == "minimax" and MINIMAX_API_KEY and MINIMAX_GROUP_ID:
            _out_note = f"語音:MiniMax {MINIMAX_VOICE} → ElevenLabs → edge-tts"
        elif AI_TTS_ENGINE == "eleven" and ELEVEN_API_KEY:
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
    print(f"AI 模式已就緒({AI_LLM_MODEL}{_search_note};{_out_note})。")
else:
    print("⚠ 未設定 GEMINI_API_KEY,AI 模式停用。")

print(f"按【{HOTKEY}】開始說話,再按一次轉錄。")
print(f"按【{AI_MODIFIER} + {HOTKEY}】進入 AI 模式。")
print(f"按【{AUTO_MODIFIER} + {HOTKEY}】切換全自動模式(每 {AUTO_INTERVAL}s 自動轉錄 + AI 回應)。")
print("結束請按 Ctrl+C。")

# ─────────────────────── 錄音串流 ────────────────────────
_frames    = []
_recording = False
_ai_mode   = False          # True = 本次錄音是 AI 問答模式
_ai_context = ""            # 錄音開始時的剪貼簿快照(文字)
_ai_image   = (None, None) # (image_b64, image_mime) 或 (None, None)
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
    r"\[(?:"
    # 情緒
    r"excited|nervous|frustrated|tired|sorrowful|calm|sad|angry|happily|awe|wistful|regretful|resigned|"
    # 笑/人聲反應
    r"laughs?(?: softly)?|big laugh|laughter|giggles?|chuckles?|sighs?|gasps?|gulps?|"
    r"clears throat|breathes?|exhales?|sniffs?|hmm+|"
    # 音量
    r"whispers?|whispering|shouts?|shouting|quietly|loudly|"
    # 節奏/語速
    r"pauses?|rushed|slows? down|deliberate|rapid-fire|drawn out|stammers?|hesitates?|timidly|"
    # 語氣
    r"cheerfully|flatly|deadpan|playfully|lighthearted|reflective|understated|emphasized|"
    r"dramatic tone|serious tone|sarcastic tone|matter-of-fact|suspicious tone"
    r")\]",
    re.IGNORECASE,
)

# ElevenLabs [tag] → MiniMax (tag) 對照表（speech-2.8-hd interjection tags）
# 沒有對應的 tag（情緒/音量/節奏類）直接移除，不影響 ElevenLabs 的處理路徑
_ELEVEN_TO_MINIMAX: dict[str, str] = {
    # 笑聲
    "laughs":          "(laughs)",
    "laughs softly":   "(chuckle)",
    "big laugh":       "(laughs)",
    "laughter":        "(laughs)",
    "giggles":         "(chuckle)",
    "chuckles":        "(chuckle)",
    # 嘆氣 / 呼吸
    "sighs":           "(sighs)",
    "breathes":        "(breath)",
    "exhales":         "(exhale)",
    "inhales":         "(inhale)",
    # 驚訝 / 緊張
    "gasps":           "(gasps)",
    "gulps":           "(emm)",
    # 清喉嚨 / 嗅
    "clears throat":   "(clear-throat)",
    "sniffs":          "(sniffs)",
    "hmm":             "(emm)",
    # 停頓（[pause] 轉成 0.6s 靜默）
    "pause":           "<#0.6#>",
    # 以下在 ElevenLabs 有意義，MiniMax 無對應 → 移除（回傳 ""）
    # excited / nervous / frustrated / tired / calm / awe / wistful / regretful
    # whispers / quietly / loudly / drawn out / rushed / stammers / hesitates 等
}

def _convert_tags_for_minimax(text: str) -> str:
    """把 ElevenLabs [tag] 轉成 MiniMax (tag)；無對應的直接移除。
    送給 MiniMax 之前呼叫；ElevenLabs 路徑完全不經過這裡。"""
    def _sub(m: re.Match) -> str:
        key = m.group(1).lower().strip()
        # 先試完整 key，再試去掉末尾 's'（複數）
        return _ELEVEN_TO_MINIMAX.get(key) or _ELEVEN_TO_MINIMAX.get(key.rstrip("s"), "")
    return re.sub(r"\[([^\]]+)\]", _sub, text).strip()

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

def _speak_minimax(text: str, cancelled=_NO_CANCEL):
    """MiniMax speech-2.8-hd → mp3 → MCI 播放。
    ElevenLabs [tag] 會先轉成 MiniMax (tag)，無對應的移除。"""
    text = _convert_tags_for_minimax(text)
    url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={MINIMAX_GROUP_ID}"
    body = {
        "model": MINIMAX_MODEL,
        "text":  text,
        "stream": False,
        "voice_setting": {
            "voice_id": MINIMAX_VOICE,
            "speed": MINIMAX_SPEED,
            "vol":   MINIMAX_VOL,
            "pitch": MINIMAX_PITCH,
        },
        "audio_setting": {
            "sample_rate": MINIMAX_RATE,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
    }
    r = requests.post(url,
                      headers={"Authorization": f"Bearer {MINIMAX_API_KEY}",
                               "Content-Type": "application/json"},
                      data=json.dumps(body), timeout=60)
    r.raise_for_status()
    if cancelled():
        return
    j = r.json()
    base = j.get("base_resp", {})
    if base.get("status_code") != 0:
        raise RuntimeError(f"MiniMax TTS 錯誤 {base.get('status_code')}: {base.get('status_msg')}")
    audio_hex = j.get("data", {}).get("audio", "")
    if not audio_hex:
        raise RuntimeError("MiniMax TTS 沒有回傳音訊")
    audio_bytes = bytes.fromhex(audio_hex)
    with open(_TTS_MP3, "wb") as f:
        f.write(audio_bytes)
    try:
        TTS_LAST_MP3.write_bytes(audio_bytes)
    except Exception as e:
        print(f"  ⚠ TTS 存檔失敗: {e}")
    if cancelled():
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

_RETRY_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

def _retry(fn, times=5, wait=1.5):
    """對網路/連線錯誤自動重試，最多 times 次，每次等 wait 秒。"""
    last_err = None
    for i in range(times):
        try:
            return fn()
        except _RETRY_ERRORS as e:
            last_err = e
            if i < times - 1:
                print(f"  ⚠ 連線失敗，{wait}s 後重試 ({i+1}/{times-1})… {e}")
                time.sleep(wait)
    raise last_err


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
    for attempt in range(5):
        try:
            r = requests.post(url, params=params, headers=headers,
                              data=json.dumps(body), stream=True, timeout=60)
        except _RETRY_ERRORS as e:
            last_err = str(e)
            if attempt < 4:
                print(f"  ⚠ ElevenLabs 連線失敗，重試 ({attempt+1}/4)…")
                time.sleep(1.5)
            continue
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}: {r.text[:150]}"
            r.close()
            if attempt < 4:
                print(f"  ⚠ ElevenLabs {last_err}，重試 ({attempt+1}/4)…")
                time.sleep(1.5)
            continue
        stream = sd.OutputStream(samplerate=ELEVEN_RATE, channels=1, dtype="int16")
        stream.start()
        leftover = b""
        pcm_chunks = []
        try:
            for chunk in r.iter_content(chunk_size=None):
                if cancelled():
                    return
                if not chunk:
                    continue
                blob = leftover + chunk
                even = len(blob) - (len(blob) % 2)
                if even:
                    pcm = blob[:even]
                    stream.write(np.frombuffer(pcm, dtype=np.int16))
                    pcm_chunks.append(pcm)
                leftover = blob[even:]
        finally:
            time.sleep(0.15)
            try: stream.stop()
            except: pass
            try: stream.close()
            except: pass
        if TTS_LAST_FILE and pcm_chunks:
            try:
                with wave.open(str(TTS_LAST_FILE), "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(ELEVEN_RATE)
                    wf.writeframes(b"".join(pcm_chunks))
            except Exception as e:
                print(f"  ⚠ TTS 存檔失敗: {e}")
        return
    raise RuntimeError(f"ElevenLabs 失敗: {last_err}")

def _speak(text: str, cancelled=_NO_CANCEL):
    """依設定挑引擎;失敗會 fallback 到下一個可用引擎。
    fallback 鏈: minimax → eleven → edge-tts"""
    if AI_TTS_ENGINE == "minimax" and MINIMAX_API_KEY and MINIMAX_GROUP_ID:
        try:
            _speak_minimax(text, cancelled)
            return
        except Exception as e:
            print(f"  ⚠ MiniMax TTS 失敗,fallback 改用 ElevenLabs: {e}")
    if AI_TTS_ENGINE in ("minimax", "eleven") and ELEVEN_API_KEY:
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

# ─────────────────────── 統一 LLM 呼叫 ───────────────────
def _handle_save_memory(args: dict):
    """執行 save_memory tool call:把事實存進 _memory_facts 並寫檔。"""
    fact = args.get("fact", "").strip()
    if not fact:
        return
    with _memory_lock:
        _memory_facts.append(fact)
        facts_copy = list(_memory_facts)
    print(f"  ⓘ [memory] 記住: {fact!r}")
    if len(facts_copy) > MEMORY_MAX_FACTS:
        threading.Thread(target=_compress_memory, daemon=True).start()
    else:
        threading.Thread(target=_save_memory, daemon=True).start()


def _call_llm(url: str, key: str, model: str,
              system: str, messages: list,
              web_search: bool = False, memory_tool: bool = False,
              img_b64: str = None, img_mime: str = None) -> str:
    """自動偵測 Google Gemini native API vs OpenAI-compatible(OpenRouter 等)。
    - Google: 走 generateContent,支援 google_search grounding + save_memory tool use
    - 其他:   走 chat/completions,OpenRouter 支援 :online suffix"""
    is_google     = "generativelanguage.googleapis.com" in url
    is_openrouter = "openrouter.ai" in url

    if is_google:
        api_url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        # 圖片：插入最後一條 user message 的 parts 最前面
        if img_b64 and img_mime and contents and contents[-1]["role"] == "user":
            contents[-1]["parts"].insert(0, {
                "inline_data": {"mime_type": img_mime, "data": img_b64}
            })
            print(f"    [vision] 附帶剪貼簿圖片({img_mime})")

        tools = []
        if web_search:
            tools.append({"google_search": {}})
        if memory_tool and MEMORY_ENABLED:
            tools.append(_MEMORY_TOOL)

        payload: dict = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
        }
        if tools:
            payload["tools"] = tools

        # 最多迭代 4 次(function call → execute → continue → text)
        for _iter in range(4):
            resp = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload), timeout=120,
            )
            resp.raise_for_status()
            raw        = resp.json()
            candidate  = raw.get("candidates", [{}])[0]
            finish     = candidate.get("finishReason", "")
            parts      = candidate.get("content", {}).get("parts", [])

            func_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            text_parts = [p.get("text", "") for p in parts if "text" in p]

            # debug:空回覆時印出完整原始結構
            if not parts or not any(p.get("text","").strip() for p in parts if "text" in p):
                print(f"  ⚠ [LLM] 空/異常回覆 finishReason={finish!r}")
                print(f"  ⚠ [LLM] raw={json.dumps(raw, ensure_ascii=False)[:600]}")
                if not parts:
                    break

            if func_calls:
                # 執行所有 function calls
                func_responses = []
                for fc in func_calls:
                    if fc["name"] == "save_memory":
                        _handle_save_memory(fc.get("args", {}))
                    func_responses.append({
                        "functionResponse": {
                            "name": fc["name"],
                            "response": {"result": "ok"},
                        }
                    })
                # 把 model 的 call + 我們的 response 加進 contents,繼續對話
                payload["contents"] = payload["contents"] + [
                    {"role": "model", "parts": parts},
                    {"role": "user",  "parts": func_responses},
                ]
                if text_parts:          # 同一輪也有文字就直接回傳
                    return "".join(text_parts).strip()
                continue               # 否則繼續取最終回覆

            if text_parts:
                joined = "".join(text_parts).strip()
                if not joined:
                    print(f"  ⚠ [LLM] text_parts 存在但為空, finishReason={finish!r}")
                return joined

            print(f"  ⚠ [LLM] 無 text 也無 functionCall, finishReason={finish!r}, parts={parts}")
            break

        return ""

    else:
        # ── OpenAI-compatible (OpenRouter / 其他) ──
        used_model = model + (":online" if web_search and is_openrouter else "")
        payload = {
            "model":    used_model,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            data=json.dumps(payload), timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def _ai_complete(system: str, messages: list,
                 img_b64: str = None, img_mime: str = None) -> str:
    return _call_llm(AI_LLM_URL, AI_LLM_KEY, AI_LLM_MODEL,
                     system, messages, AI_WEB_SEARCH,
                     memory_tool=False,
                     img_b64=img_b64, img_mime=img_mime)

def _ask_llm(context: str, question: str,
             img_b64: str = None, img_mime: str = None) -> tuple[str, str]:
    """把 context + question (+圖片) 送到 AI_LLM_MODEL。
    回傳 (answer, user_msg) — 不直接改記憶;由 worker 決定是否提交(打斷時就不提交)。"""
    user_msg = ""
    if img_b64:
        user_msg += "[Clipboard: image attached]\n\n"
    elif context.strip():
        user_msg += f"[Clipboard]\n{context.strip()}\n\n"
    user_msg += f"[Question]\n{question.strip()}"

    system = AI_SYSTEM_PROMPT
    if _memory_facts:
        system += "\n\n[What you know about this user]\n" + "\n".join(f"- {f}" for f in _memory_facts)
    if _chat_summary:
        system += f"\n\n[Previous conversation summary]\n{_chat_summary}"
    messages = list(_chat_history) + [{"role": "user", "content": user_msg}]

    answer = _ai_complete(system, messages, img_b64, img_mime)
    return answer, user_msg

def _condense_history():
    """逐字歷史超過上限時,把最舊的對話併入滾動摘要(再 call 一次 LLM 壓成 ≤N 字)。"""
    global _chat_history, _chat_summary
    if len(_chat_history) <= AI_HISTORY_TURNS * 2:
        return
    keep   = AI_KEEP_RECENT * 2
    old    = _chat_history[:-keep]
    recent = _chat_history[-keep:]

    convo = ""
    for m in old:
        who = "User" if m["role"] == "user" else "Assistant"
        convo += f"{who}: {m['content']}\n"

    body = ""
    if _chat_summary:
        body += f"[Previous summary]\n{_chat_summary}\n\n"
    body += f"[Conversation to compress]\n{convo}"

    instr = (
        f"Summarize the following in under {AI_SUMMARY_CHARS} characters. "
        "Keep key points, conclusions, user preferences, names, and todos. "
        "Output only the summary, nothing else."
    )
    try:
        new_summary = _ai_complete(instr, [{"role": "user", "content": body}]).strip()
        _chat_summary = new_summary
        _chat_history = recent
        print(f"  ⓘ 已壓縮歷史 → 摘要 {len(_chat_summary)} 字,保留最近 {AI_KEEP_RECENT} 輪")
        _save_history()
    except Exception as e:
        _chat_history = _chat_history[-AI_HISTORY_TURNS * 2:]
        print(f"  ⚠ 壓縮失敗,改丟棄最舊: {e}")
        _save_history()

# ─────────────────────── 轉錄 Worker ─────────────────────
def _transcribe(audio: np.ndarray) -> str:
    last_err = None
    for attempt in range(5):
        try:
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
        except RuntimeError as e:          # CUDA OOM 或其他 GPU 錯誤
            last_err = e
            if attempt < 4:
                print(f"  ⚠ ASR 失敗，重試 ({attempt+1}/4)… {e}")
                time.sleep(1.0)
    raise last_err


def _vad_extract_speech(audio: np.ndarray) -> np.ndarray:
    """Energy VAD:只保留能量超過 AUTO_VAD_THRESHOLD 的語音幀(前後加 padding)。
    大幅減少送給 Whisper 的靜音,避免幻覺重複詞。
    若語音比例低於 5% → 回傳空陣列(讓呼叫方視為靜音跳過)。"""
    frame_n  = int(SAMPLE_RATE * 0.030)           # 30ms / 幀
    pad_n    = max(1, round(AUTO_VAD_PAD_MS / 30)) # 要延伸幾幀
    n_frames = len(audio) // frame_n
    if n_frames == 0:
        return audio

    frames = audio[: n_frames * frame_n].reshape(n_frames, frame_n)
    rms    = np.sqrt(np.mean(frames ** 2, axis=1))
    is_sp  = rms > AUTO_VAD_THRESHOLD

    # 膨脹:語音幀往兩側延伸 pad_n 幀
    mask = np.zeros(n_frames, dtype=bool)
    for i in np.where(is_sp)[0]:
        mask[max(0, i - pad_n) : min(n_frames, i + pad_n + 1)] = True

    ratio = float(mask.sum()) / n_frames
    if ratio < 0.05:
        return np.array([], dtype=np.float32)

    filtered = frames[mask].flatten()
    print(f"    [VAD] 保留 {ratio*100:.0f}% "
          f"({filtered.shape[0]/SAMPLE_RATE:.1f}s / {audio.shape[0]/SAMPLE_RATE:.1f}s)")
    return filtered



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

def _ai_worker(audio: np.ndarray, context: str, my_session: int, clip_image=(None, None)):
    """AI 問答模式:ASR → LLM → 打字 + 念。被打斷(session 變)就靜默放棄。"""
    global _last_sent_clipboard, _last_sent_image_hash
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

        # 脈絡去重:文字跟圖片分開判斷,相同就不重複送
        snapshot      = context
        effective_ctx = context
        img_b64, img_mime = clip_image
        img_hash = hashlib.md5(img_b64.encode()).hexdigest() if img_b64 else ""

        if context and context == _last_sent_clipboard:
            print("  ⓘ 剪貼簿同上一次,不再重複送脈絡。")
            effective_ctx = ""
        if img_b64 and img_hash == _last_sent_image_hash:
            print("  ⓘ 圖片同上一次,不再重複送。")
            img_b64 = img_mime = None

        turns = len(_chat_history) // 2
        ctx_preview = (effective_ctx[:80] + "…") if len(effective_ctx) > 80 else effective_ctx
        img_note = f" + 圖片" if img_b64 else ""
        print(f"  → 送 LLM … (脈絡 {len(effective_ctx)} 字{img_note} / 歷史 {turns} 輪)"
              + (f"\n     脈絡: {ctx_preview!r}" if effective_ctx else ""))
        t1     = time.time()
        answer, user_msg = _ask_llm(effective_ctx, question, img_b64, img_mime)
        answer = _parse_and_save_memory(answer)   # 抽出 <save_memory> tag 並存檔
        print(f"  → LLM ({time.time()-t1:.1f}s) {answer!r}")
        if cancelled():
            print("  ⓘ 已被新一輪打斷,丟棄此次回覆。"); return

        # 提交本輪到記憶
        _chat_history.append({"role": "user",      "content": user_msg})
        _chat_history.append({"role": "assistant", "content": answer})
        _last_sent_clipboard = snapshot
        if img_hash:
            _last_sent_image_hash = img_hash
        threading.Thread(target=_save_history, daemon=True).start()

        # 只念出來,不貼文字
        # 任一 TTS 引擎可用就念
        _tts_available = (
            (AI_TTS_ENGINE == "minimax" and MINIMAX_API_KEY and MINIMAX_GROUP_ID)
            or (AI_TTS_ENGINE in ("minimax", "eleven") and ELEVEN_API_KEY)
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
    global _recording, _frames, _timeout_timer, _ai_mode, _ai_context, _ai_image, _session_id
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
        _ai_image   = (None, None)
    if ai:
        # 先快照剪貼簿(含圖片),讓後面錄音時使用者可以繼續複製新內容也沒關係
        _ai_context, img_b64, img_mime = _snapshot_clipboard()
        _ai_image = (img_b64, img_mime)
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
        threading.Thread(target=_ai_worker,      args=(audio, ctx, sid, _ai_image), daemon=True).start()
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

def _auto_ask_openrouter(context: str, question: str) -> tuple[str, str]:
    """全自動模式 LLM:只帶最近 5 輪歷史 + 摘要,省 token。"""
    user_msg = ""
    if context.strip():
        user_msg += f"[Clipboard]\n{context.strip()}\n\n"
    user_msg += f"[Question]\n{question.strip()}"
    sys = AI_SYSTEM_PROMPT
    if _chat_summary:
        sys += f"\n\n[Previous conversation summary]\n{_chat_summary}"
    messages = list(_chat_history)[-10:] + [{"role": "user", "content": user_msg}]
    answer = _call_llm(AUTO_LLM_URL, AUTO_LLM_KEY, AUTO_LLM_MODEL,
                       sys, messages, web_search=False)
    return answer, user_msg

def _auto_worker(audio: np.ndarray, context: str):
    """全自動模式的 ASR → OpenRouter → TTS worker。
    做完之後(不論成功失敗)自動開下一輪錄音。"""
    try:
        # VAD:先濾掉靜音段,避免 Whisper 在靜音中幻覺重複詞
        speech = _vad_extract_speech(audio)
        if len(speech) < int(SAMPLE_RATE * MIN_SECONDS):
            print(f"  [auto] VAD 後語音不足 {MIN_SECONDS}s,跳過。")
            return
        question = _transcribe(speech)
        print(f"  [auto] ASR: {question!r}")
        if not question.strip():
            return
        print(f"  [auto] 送 {AUTO_LLM_MODEL} …")
        answer, user_msg = _auto_ask_openrouter(context, question)
        answer = _parse_and_save_memory(answer)
        print(f"  [auto] AI: {answer!r}")
        _chat_history.append({"role": "user",      "content": user_msg})
        _chat_history.append({"role": "assistant", "content": answer})
        threading.Thread(target=_save_history, daemon=True).start()
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

_lctrl_down = False
def _on_lctrl(e):
    global _lctrl_down
    _lctrl_down = (e.event_type == keyboard.KEY_DOWN)

keyboard.hook_key("left alt",  _on_lalt)
keyboard.hook_key("right alt", _on_ralt)
keyboard.hook_key("left ctrl", _on_lctrl)

def _on_hotkey(event):
    if event.event_type != "down":
        return
    if _lctrl_down:
        if not AI_LLM_KEY:
            print("⚠ 未設定 XAI_API_KEY,無法使用自動模式。"); beep_error(); return
        _toggle_auto()
    elif _ralt_down:
        if not AI_LLM_KEY:
            print("⚠ 未設定 XAI_API_KEY,AI 模式無法使用。"); beep_error(); return
        _toggle(ai=True)
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
