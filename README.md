# breeze-asr-dictate

Windows 全域熱鍵語音聽寫工具，使用 Breeze-ASR-25（台灣中文語音辨識）+ Grok AI 問答（即時搜尋）。

## 功能

| 熱鍵 | 功能 |
|------|------|
| **Copilot 鍵** | 切換錄音；辨識完直接模擬鍵盤打字輸出（不動剪貼簿） |
| **右 Alt + Copilot 鍵** | AI 模式：語音問題 + 剪貼簿脈絡 → Grok AI → 打字輸出 + 台灣腔語音播報 |

- 模型常駐 VRAM，回應延遲低
- AI 具對話記憶（15 輪逐字 + 滾動摘要壓縮）
- AI 可即時上網搜尋

---

## 安裝步驟

### 環境需求

- Windows 10/11 (64-bit)
- Python 3.11–3.14
- NVIDIA GPU（建議 8GB VRAM 以上，CUDA 12.1+）
- 管理員身份執行終端機（`keyboard` 全域鍵盤監聽需要）

### 1. 下載專案

```bat
git clone https://github.com/j9263178/breeze-asr-dictate.git
cd breeze-asr-dictate
```

### 2. 建立虛擬環境

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安裝 PyTorch（依你的 CUDA 版本選一個）

```bat
:: CUDA 12.6（RTX 40 系列）
pip install torch --index-url https://download.pytorch.org/whl/cu126

:: CUDA 12.1（RTX 30 系列等）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

不確定版本就跑 `nvidia-smi`，看右上角的 CUDA Version。

### 4. 安裝其他套件

```bat
pip install -r requirements.txt
```

### 5. 下載 Breeze-ASR-25 模型

從 [HuggingFace](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) 下載所有檔案，放到：

```
breeze-asr-dictate/
└── models/
    └── Breeze-ASR-25/
        ├── model.safetensors   ← 最大的那個，約 3GB
        ├── config.json
        ├── tokenizer.json
        └── ...（其他小檔）
```

`model.safetensors`（~3GB）建議用瀏覽器直接下載，其他小檔可用 `huggingface-cli`：

```bat
pip install huggingface_hub
huggingface-cli download MediaTek-Research/Breeze-ASR-25 --local-dir models/Breeze-ASR-25 --exclude "*.safetensors"
```

### 6. 設定 API 金鑰

建立 `.env` 檔（不會被 git 追蹤）：

```
XAI_API_KEY=你的_xAI_API_金鑰
```

- xAI API 金鑰申請：https://console.x.ai/
- 沒有金鑰也能用，AI 模式（右 Alt + Copilot）會停用，普通聽寫不受影響

### 7. 啟動

```bat
:: 需以管理員身份開啟終端機
.venv\Scripts\python.exe dictate.py
```

第一次啟動約需 20–30 秒載入模型，看到「模型就緒」即可開始使用。

---

## 選用設定

`dictate.py` 最上面可以調：

```python
MAX_SECONDS      = 60       # 最長錄音秒數
AI_TTS           = True     # AI 回覆是否用台灣腔語音念出來
AI_HISTORY_TURNS = 15       # AI 對話保留幾輪逐字
AI_SUMMARY_CHARS = 500      # 滾動摘要字數上限
OUTPUT_MODE      = "type"   # "type"=直接打字(不動剪貼簿) / "clipboard"=剪貼簿貼上
```

### 自訂詞彙

把你常說、但模型常認錯的詞加到 `vocab.txt`（一行一個），重啟生效：

```
Claude
台積電
Kubernetes
```

### 開機自動啟動

在 `shell:startup` 資料夾建立捷徑，目標設為：

```
pythonw.exe C:\完整路徑\dictate.py
```

或用工作排程器（Task Scheduler）設「使用者登入時」執行，勾選「以最高權限執行」。

---

## 驗證 GPU 推論

```bat
.venv\Scripts\python.exe test_model.py
```

---

## 致謝

本專案使用 [Breeze-ASR-25](https://github.com/mtkresearch/Breeze-ASR-25)，由 MediaTek Research 開發，採 [MIT License](https://github.com/mtkresearch/Breeze-ASR-25/blob/main/LICENSE)。
