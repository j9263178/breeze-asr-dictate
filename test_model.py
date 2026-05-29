"""下載 Breeze-ASR-25 並用一段假音訊驗證 GPU 推論可跑通。"""
import time
from pathlib import Path
import numpy as np
import torch
from transformers import pipeline

MODEL_ID = Path(__file__).parent / "models" / "Breeze-ASR-25"  # 自動抓程式所在資料夾
SR = 16000

print("下載 / 載入模型中…")
t0 = time.time()
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    torch_dtype=torch.float16,
    device=0,
    chunk_length_s=30,
)
print(f"模型就緒,耗時 {time.time()-t0:.1f}s")

# 1 秒 440Hz 正弦波當測試音訊(不期待有意義文字,只驗證流程不爆)
audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
t0 = time.time()
out = asr({"raw": audio, "sampling_rate": SR},
          generate_kwargs={"language": "chinese", "task": "transcribe"})
print(f"推論完成,耗時 {time.time()-t0:.1f}s")
print("輸出:", repr(out["text"]))
print("VRAM 已配置:", f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")
