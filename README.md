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

**🎉 部署完成！開始使用你的 OCR API 服務吧！**
