# DeepSeek-OCR_DockerDeploy
利用 Docker image 離線部屬與測試DeepSeek-OCR。

For offline testing and deployment of DeepSeek-OCR by building a Docker image.

掛載的serve_ocr.py可以調整為符合自己需求的格式，所有程式更新後，直接restart container即可使用更新後的服務。

The mounted serve_ocr.py file can be modified to suit your desired output format. After updating all the code, you can simply restart the container.

# DeepSeek-OCR API 服務完整指南

## 📋 目錄
- [專案結構](#專案結構)
- [功能總覽](#功能總覽)
- [安裝部署](#安裝部署)
- [API 端點](#api-端點)
- [使用範例](#使用範例)
- [測試指南](#測試指南)

---

## 📁 專案結構

```
your-project/
├── serve_ocr.py          # 主服務入口
├── ocr_routes.py         # 進階圖片 OCR 路由
├── pdf_routes.py         # PDF OCR 路由
├── batch_routes.py       # 批量評估路由
├── ocr_utils.py          # 工具函數庫
├── test_ocr.py           # 完整測試程式
├── config.py             # 配置文件
├── process/              # 官方依賴
│   ├── ngram_norepeat.py
│   └── image_process.py
├── deepseek_ocr.py       # 模型定義
├── requirements.txt      # Python 依賴
├── docker-compose.yml    # Docker 服務配置
└── Dockerfile            # Docker 配置
```

---

## 🎯 功能總覽

### 1️⃣ 基礎 OCR（原有功能）
- ✅ 純文字推論
- ✅ Base64 圖片輸入
- ✅ 檔案上傳（單張圖片）

### 2️⃣ 進階圖片 OCR 🆕
- ✅ 邊界框檢測與繪製
- ✅ 圖片區域自動裁剪
- ✅ 多種裁剪模式（singlepage/tiling）
- ✅ 返回帶標註的圖片（base64）

### 3️⃣ PDF 文件 OCR 🆕
- ✅ 多頁 PDF 批次處理
- ✅ 自動提取文件中的圖片
- ✅ 生成帶佈局標註的 PDF
- ✅ 輸出 Markdown 格式
- ✅ 可調整 DPI 品質

### 4️⃣ 批量評估 OCR 🆕
- ✅ 一次處理多張圖片
- ✅ 平行預處理 + 批次推論
- ✅ 自動清理數學公式
- ✅ 下載結果 ZIP 檔案
- ✅ 適合 OmniDocBench 等數據集

---

## 🚀 安裝部署

### 方法 1: Docker 部署（推薦）

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 安裝 Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# 複製專案文件
COPY . /app
WORKDIR /app

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 下載模型（或掛載）
ENV MODEL_PATH=/app/models/deepseek-ocr

# 啟動服務
CMD ["uvicorn", "serve_ocr:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 建置映像
docker build -t deepseek-ocr-api .

# 啟動容器
docker run -d \
  -p 8063:8000 \
  --gpus all \
  -v /path/to/models:/app/models \
  -e GPU_MEM_UTILIZATION=0.75 \
  deepseek-ocr-api
```

### 方法 2: 本地部署

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
export MODEL_PATH=/path/to/deepseek-ocr
export GPU_MEM_UTILIZATION=0.75

# 啟動服務
uvicorn serve_ocr:app --host 0.0.0.0 --port 8000
```

### requirements.txt

```txt
# 核心依賴
fastapi==0.104.1
uvicorn[standard]==0.24.0
pillow==10.1.0
numpy==1.24.3

# OCR 依賴
vllm==0.5.0
transformers==4.38.0
torch==2.1.0

# PDF 處理
pymupdf==1.23.8
img2pdf==0.5.1

# 工具庫
tqdm==4.66.1
requests==2.31.0
```

---

## 📡 API 端點

### 基礎端點

| 端點 | 方法 | 功能 | 超時 |
|------|------|------|------|
| `/` | GET | 服務資訊 | 10s |
| `/health` | GET | 健康檢查 | 10s |
| `/generate` | POST | 基礎推論（支援 base64 圖片）| 60s |
| `/generate_with_file` | POST | 圖片檔案上傳 | 60s |

### 進階圖片 OCR

| 端點 | 方法 | 功能 | 超時 |
|------|------|------|------|
| `/ocr/image/advanced` | POST | 完整圖片 OCR | 60s |
| `/ocr/image/simple` | POST | 快速文字提取 | 60s |
| `/ocr/status` | GET | 功能狀態檢查 | 10s |
| `/ocr/info` | GET | 使用說明 | 10s |

### PDF OCR

| 端點 | 方法 | 功能 | 超時 |
|------|------|------|------|
| `/pdf/ocr` | POST | 完整 PDF OCR | 300s |
| `/pdf/ocr/simple` | POST | 快速 PDF 文字提取 | 300s |
| `/pdf/status` | GET | 功能狀態檢查 | 10s |
| `/pdf/info` | GET | 使用說明 | 10s |

### 批量 OCR

| 端點 | 方法 | 功能 | 超時 |
|------|------|------|------|
| `/batch/ocr` | POST | 批量圖片處理（JSON）| 600s |
| `/batch/ocr/download` | POST | 批量處理並下載 ZIP | 600s |
| `/batch/ocr/simple` | POST | 快速批量處理 | 600s |
| `/batch/status` | GET | 功能狀態檢查 | 10s |
| `/batch/info` | GET | 使用說明 | 10s |

---

## 💡 使用範例

### 1. 基礎圖片 OCR

```bash
curl -X POST "http://localhost:8000/generate_with_file" \
     -F "file=@invoice.png" \
     -F "prompt=<image>Extract all text" \
     -F "max_tokens=512"
```

```python
import requests

response = requests.post(
    "http://localhost:8000/generate_with_file",
    files={"file": open("document.jpg", "rb")},
    data={
        "prompt": "<image>Extract text",
        "max_tokens": 512,
        "temperature": 0.1
    }
)
print(response.json())
```

### 2. 進階圖片 OCR（帶邊界框）

```bash
curl -X POST "http://localhost:8000/ocr/image/advanced" \
     -F "file=@document.png" \
     -F "prompt=<image>Extract all text and structure" \
     -F "draw_boxes=true" \
     -F "crop_regions=true" \
     -F "crop_mode=singlepage"
```

### 3. PDF 完整處理

```bash
curl -X POST "http://localhost:8000/pdf/ocr" \
     -F "file=@report.pdf" \
     -F "prompt=<image>\n<|grounding|>Convert to markdown" \
     -F "draw_layouts=true" \
     -F "extract_images=true" \
     -F "dpi=144"
```

### 4. 批量處理並下載

```bash
curl -X POST "http://localhost:8000/batch/ocr/download" \
     -F "files=@img1.jpg" \
     -F "files=@img2.jpg" \
     -F "files=@img3.jpg" \
     -o results.zip
```

```python
import requests

files = [
    ('files', ('img1.jpg', open('img1.jpg', 'rb'), 'image/jpeg')),
    ('files', ('img2.jpg', open('img2.jpg', 'rb'), 'image/jpeg')),
]

response = requests.post(
    "http://localhost:8000/batch/ocr",
    files=files,
    data={"prompt": "<image>Extract text"}
)

results = response.json()
for result in results['results']:
    print(f"{result['filename']}: {result['markdown'][:100]}...")
```

---

## 🧪 測試指南

### 快速測試

```bash
# 1. 健康檢查
curl http://localhost:8000/health

# 2. 檢查功能狀態
curl http://localhost:8000/ocr/status
curl http://localhost:8000/pdf/status
curl http://localhost:8000/batch/status

# 3. 執行測試程式
python test_ocr.py
```

### 測試程式選單

```
1. 快速測試 (健康檢查 + 單一推論)
2. 圖片上傳測試 (互動式)
3. 批量圖片處理 (自動搜尋當前目錄)
4. 同一圖片多種 Prompt 測試
5. PDF OCR 測試 (完整功能)
6. PDF OCR 測試 (簡單模式)
7. 批量 OCR 測試 (多圖片)
8. 批量 OCR 下載 ZIP
9. 完整測試 (所有功能)
```

### 自訂測試

```python
import requests

# 測試單張圖片
response = requests.post(
    "http://localhost:8000/generate_with_file",
    files={"file": open("test.png", "rb")},
    data={"prompt": "<image>Extract text"}
)
print(response.json()["result"])

# 測試 PDF
response = requests.post(
    "http://localhost:8000/pdf/ocr/simple",
    files={"file": open("test.pdf", "rb")}
)
print(response.json()["text"])

# 測試批量
files = [
    ('files', open('img1.jpg', 'rb')),
    ('files', open('img2.jpg', 'rb'))
]
response = requests.post(
    "http://localhost:8000/batch/ocr/simple",
    files=files
)
print(response.json())
```

---

## ⚙️ 配置說明

### 環境變數

```bash
# 模型路徑（必填）
MODEL_PATH=/app/models/deepseek-ocr

# GPU 記憶體使用率
GPU_MEM_UTILIZATION=0.75

# 圖片處理模式
CROP_MODE=true
BASE_SIZE=1024
IMAGE_SIZE=640

# 並發設定
MAX_CONCURRENCY=100
NUM_WORKERS=64
```

### Prompt 範本

```python
# 文件轉 Markdown
"<image>\n<|grounding|>Convert the document to markdown."

# 自由 OCR
"<image>\nFree OCR."

# 帶定位的 OCR
"<image>\n<|grounding|>OCR this image."

# 圖表解析
"<image>\nParse the figure."

# 詳細描述
"<image>\nDescribe this image in detail."
```

---

## 🐛 故障排除

### 1. 服務無法啟動

```bash
# 檢查依賴
pip list | grep -E "vllm|torch|transformers"

# 檢查 GPU
nvidia-smi

# 檢查模型路徑
ls -la $MODEL_PATH
```

### 2. 記憶體不足

```bash
# 降低 GPU 使用率
export GPU_MEM_UTILIZATION=0.6

# 降低並發數
export MAX_CONCURRENCY=50

# 調整 DPI（PDF）
# 使用 dpi=96 而非 144
```

### 3. 某些功能不可用

```bash
# 檢查功能狀態
curl http://localhost:8000/
curl http://localhost:8000/ocr/status
curl http://localhost:8000/pdf/status
curl http://localhost:8000/batch/status

# 服務會自動降級，其他功能不受影響
```

### 4. 超時錯誤

```python
# 增加超時時間
import requests

response = requests.post(
    url,
    files=files,
    timeout=600  # 10 分鐘
)
```

---

## 📊 效能優化

### 1. 圖片預處理

- 使用 `NUM_WORKERS` 控制平行處理數
- 建議值：CPU 核心數的 2-4 倍

### 2. 批次推論

- 使用 `MAX_CONCURRENCY` 控制並發推論數
- GPU 記憶體充足時可增加至 200+

### 3. PDF 處理

- 小型 PDF：使用完整模式
- 大型 PDF：先用簡單模式預覽
- 調整 DPI：144（高品質）、96（平衡）、72（快速）

### 4. 批量處理

- 單次建議不超過 100 張圖片
- 使用 `/batch/ocr/download` 直接獲取 ZIP
- 大批量分批處理

---

## 📝 架構

- 基於 [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 官方腳本
- 使用 vLLM 加速推論
- FastAPI 框架

---

## 🆘 獲取幫助

```bash
# 查看 API 文件
curl http://localhost:8000/ocr/info
curl http://localhost:8000/pdf/info
curl http://localhost:8000/batch/info

# 或在瀏覽器訪問
http://localhost:8000/docs  # Swagger UI
http://localhost:8000/redoc # ReDoc
```

---

### **1️⃣ 模型整體架構**

DeepSeek‑OCR 是一個 **多模態（Multimodal）OCR 模型**，可以直接把圖像作為輸入，輸出文字或結構化文本（Markdown、HTML-like 標記）。架構主要由三個部分組成：

```
Input Image → DeepEncoder (SAM + CLIP) → Projector → Language Model Decoder → Output Text
```

---

### **2️⃣ DeepEncoder（視覺編碼器）**

- **作用**：將圖片轉換為「視覺特徵向量」，供語言模型解碼器使用。
- **組成**：
    1. **SAM 模型 (build_sam_vit_b)**
        - 主要用於局部 patch 的圖像特徵提取
        - 輸入：局部裁切的圖像 patch 或全局圖像
        - 輸出：patch embedding
    2. **CLIP 模型 (build_clip_l)**
        - 對 SAM 特徵進行進一步語意向量化
        - 支援圖像和文本的對齊
    3. **Projector（MlpProjector）**
        - 將 SAM + CLIP 融合特徵映射到與語言模型 embedding 相同維度（通常 n_embed=1280）
        - 可以理解為「將視覺特徵對齊到語言模型 token 空間」
- **特性**：
    - 支援 **2D tile cropping**，將大圖分塊處理，提高細節捕捉能力
    - 支援全局和局部視角特徵融合（global + local features）
    - 使用特殊 token (`<image>`, `view_separator`) 來標記圖像 token 在語言序列的位置

---

### **3️⃣ 多模態融合**

- **方法**：將視覺特徵嵌入到語言模型的 token embedding 空間
- **流程**：
    1. 將圖像轉換成固定數量的「image tokens」
        - 使用 `_parse_and_validate_image_input` 和 `_pixel_values_to_embedding`
    2. 將 image tokens 插入到語言模型 input sequence 中，對應 `<image>` token
        - 這部分在 `get_input_embeddings` 中完成
        - 語言模型可以直接把 image tokens 當作「文字 token」一起處理

---

### **4️⃣ 語言模型解碼器（Text Decoder）**

- **核心**：自回歸 Transformer LM
- **模型選擇**：
    - DeepSeekV2 / DeepSeekV3 / DeepSeek（依據 text_config 設定）
    - 參數量大約 **0.5B – 3B**，MoE 模型時每次推理啟用部分專家（sparse activations）
- **輸入**：
    - 語言 token embeddings + 視覺 token embeddings
- **輸出**：
    - 文本序列（支援 Markdown、HTML-like 結構、公式等）
- **特性**：
    - 支援 logits_processors（如 `NoRepeatNGramLogitsProcessor`）控制生成重複、標籤等
    - 可做 **batch 推理**，每張圖片生成對應文本文件

---

### **5️⃣ 特殊功能**

1. **公式清理（LaTeX）**
    - 用 `clean_formula()` 過濾 `\quad(...)` 等冗餘符號
2. **結構化匹配**
    - 使用正則處理 `<|ref|>...</|ref|>` 或 `<|det|>...</|det|>`
3. **裁切模式**
    - `CROP_MODE=True` 時會自動分塊（tiles）提取局部特徵
4. **多線程預處理**
    - `ThreadPoolExecutor` 支援 batch 處理多張圖片

---

### **6️⃣ 模型流程總結**

1. **輸入圖片** → PIL Image → 多線程裁切與預處理
2. **DeepEncoder** → SAM + CLIP → projector → image token embeddings
3. **語言模型** → input_ids + image token embeddings → Transformer → 文本生成
4. **後處理** → clean_formula + 正則匹配 → 輸出 Markdown 文件

---

📌 **核心亮點**

- **可直接讀取圖像，不需要先做 OCR 再用文本生成**
- **局部 + 全局視覺特徵融合**
- **多模態 token 對齊語言模型空間**
- **支援結構化文本輸出（Markdown、公式、表格）**
- **MoE 技術降低活躍參數量，節省推理資源**

# **DeepSeek-OCR（開源模型）完整功能統整**

根據程式碼顯示，DeepSeek-OCR 不是一個單純文字 OCR。

它是一個 **多視角、多 Tile、多階段特徵融合的 Vision–Language 模型**，整體功能可分成 4 大類：

---

# 1. **圖片處理能力**

## 1.1 支援多種圖片大小、長圖、極不規則比例圖片

程式碼會：

1. 判斷是否大於 640×640
2. 若圖片太大 → 自動裁切成 **tile local patches**
3. 若圖片可接受 → 保留 global view

→ 適合多種類型圖片：

✅ 長條票據

✅ A4 論文

✅ 手寫筆記

✅ 漫畫長圖

✅ 手機截圖

✅ 多欄位 PDF 頁面

---

## 1.2 兩種視角特徵（One-shot 通吃）

模型同時跑：

### ● Global View (整張圖片)

經由：

- SAM Encoder
- CLIP-L Encoder
- Projector

### ● Local View (裁切 tile)

每個小 tile 同時跑：

- SAM Encoder
- CLIP-L Encoder
- Projector

並把所有視角線性展平成 token 序列：

```
Local tile features → <newline> → Global features → <view_separator>
```

功能：

✅ 針對極小字也能辨識

✅ 解決 A4 PDF 裁切後失真問題

✅ 多區域表格也能抓到

✅ 提高低解析度文字辨識成功率

---

## 1.3 圖片會被轉成「視覺 token 序列」

也就是：

✅ `<image>` 會被展開成 **上千個視覺 Token**

✅ 視覺 Token 會融入語言模型 seq input

✅ 可當成類似 LLaVA、Gemini 的方式使用

---

# 2. **OCR 文字辨識能力**

程式碼顯示 DeepSeek-OCR 的 **核心是 DeepseekV3 / DeepseekV2 / DeepseekForCausalLM**。

具備：

✅ 高階序列建模能力（非傳統 CNN OCR）

✅ 支援任意語言（中英最強）

✅ 支援多行、多欄位

✅ 支援 prompt 指示（可控制輸出格式）

✅ 支援 caption + OCR + 解析（非純文字辨識）

---

# 3. **多模態能力（真正重點）**

從 `DeepseekOCRMultiModalProcessor` 與 embeddings merge 看得出：

✅ 模型並非純 OCR

✅ 是完整 Vision-Language Multimodal 模型

✅ 可以接收：

- prompt
- `<image>`
- 視覺特徵
- 多視角 Tile
- 文字序列

### → 因此支援：

✅ 文字 OCR

✅ 圖片問答

✅ 表格解析

✅ 圖片分類語意描述

✅ 從圖片中定位資訊

✅ 從票據/收據自動抓欄位

✅ 文本區塊理解

✅ 圖片格式化輸出（如 JSON、表格）

注意：

**這些功能全看 prompt 要求，程式碼並沒有限制。**

---

# 4. **輸出特性**

### ✅ 4.1 完整因果語言模型輸出（Causal LM）

模型最終輸出是 language_model.generate** 的 logits。

所以只要 prompt 要，會輸出：

✅ 純文字 OCR

✅ 含座標 OCR（若使用特定 prompt）

✅ 表格 JSON

✅ key-value 抽取

✅ 多欄位 OCR

✅ 圖片摘要

---

# 5. **模型不需要額外 LLM**

程式碼顯示：

```
language_model = DeepseekV3 / DeepseekV2
```

✅ 已內建 DeepSeek 的大型模型

✅ 不需要額外外掛 ChatGPT、LLama、Gemini

✅ 完全端到端

---

# 6. **支援 HuggingFace 格式、vLLM、Quantization**

程式碼顯示：

✅ 支援 HuggingFace 權重

✅ 支援 vLLM runtime（高速推論）

✅ 支援量化（Q-LoRA、INT8、INT4）

✅ 支援 Pipeline Parallel（大模型多 GPU 分布式）

---

# 7. **官網沒說但程式碼揭露的「隱藏功能」**

### ✅ （1）SAM + CLIP 雙編碼器（非常罕見）

同時使用：

- SAM（擅長 segmentation 邊界）
- CLIP-L（擅長語意）

→ 這意味著模型可以更精準處理：

✅ 小字（邊界 segmentation）

✅ 文字邏輯（語意 CLIP）

---

### ✅（2）自動偵測 tile（可擴展大量長圖）

因為有：

```
count_tiles()
```

表示模型可：

✅ 自動切割長條/超大圖

✅ 並拼回 token 序列

✅ 保留相對位置資訊

這是一般 OCR 模型無法做到的。

---

### ✅（3）Prompt 可完全自定義（PROMPT 常量）

程式碼讀取 `PROMPT`

→ 你可以做客製化 OCR 行為：

✅ 像 ChatGPT 那樣解析圖片

✅ 要求輸出 JSON 格式

✅ 要求擷取特定欄位

✅ 要求只回文字內容

---

# **DeepSeek-OCR 功能精華（總結）**

✅ **通用 OCR**

✅ **手寫 / 印刷體** 都能識別

✅ **多欄位 PDF / 表格 / 收據票據**

✅ **圖片問答（VQA）**

✅ **表格結構解析**

✅ **文字資訊抽取（IE）**

✅ **圖片摘要 / captioning**

✅ **key-value 表單擷取**

✅ **從複雜排版中獲得語意結構**

✅ **可用 prompt 完全客製化行為**

✅ **可多張圖片（最多 N 張）一起輸入**

✅ **支援長圖與動態裁切 tile**

✅ **支援全中文 OCR**

✅ **支援量化、vLLM 高速執行**
