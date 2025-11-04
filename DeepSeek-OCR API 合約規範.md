# DeepSeek-OCR API 合約規範 (Single Source of Truth)

**版本**: 2.0.0  
**最後更新**: 2025-11-04

---

## 📋 目錄

1. [概述](#概述)
2. [基礎配置](#基礎配置)
3. [API 端點](#api-端點)
4. [資料結構定義](#資料結構定義)
5. [錯誤處理](#錯誤處理)
6. [使用範例](#使用範例)
7. [核心功能實作](#核心功能實作)
8. [最佳實踐](#最佳實踐)

---

## 概述

DeepSeek-OCR API 提供文檔 OCR 識別服務,支持圖片和 PDF 文檔的文字提取、Markdown 轉換及視覺定位(Grounding)功能。

### 核心特性
- ✅ 多種圖片格式支持 (JPG, PNG, JPEG)
- ✅ PDF 多頁批量處理
- ✅ 智能圖片裁切與分塊 (Dynamic Preprocessing)
- ✅ 視覺定位 (Grounding) - 提取邊界框與子圖片
- ✅ 流式輸出 (Streaming)
- ✅ 批量並發處理
- ✅ N-gram 防重複機制
- ✅ 多種預設模式

---

## 基礎配置

### 模型配置模式

| 模式 | BASE_SIZE | IMAGE_SIZE | CROP_MODE | MIN_CROPS | MAX_CROPS | 適用場景 |
|------|-----------|------------|-----------|-----------|-----------|----------|
| Tiny | 512 | 512 | false | 2 | 6 | 快速處理小圖 |
| Small | 640 | 640 | false | 2 | 6 | 標準文檔 |
| Base | 1024 | 1024 | false | 2 | 6 | 高質量文檔 |
| Large | 1280 | 1280 | false | 2 | 6 | 超高解析度 |
| **Gundam** | 1024 | 640 | **true** | 2 | 6 | 大型文檔智能裁切 (默認) |

### 系統配置參數

```typescript
interface SystemConfig {
  // 圖片處理
  BASE_SIZE: 512 | 640 | 1024 | 1280;      // 全局視圖尺寸
  IMAGE_SIZE: 512 | 640 | 1024 | 1280;     // 局部視圖尺寸
  CROP_MODE: boolean;                      // 是否啟用動態裁切
  MIN_CROPS: number;                       // 最小裁切塊數, 默認: 2
  MAX_CROPS: number;                       // 最大裁切塊數, 默認: 6
  
  // 性能配置
  MAX_CONCURRENCY: number;                 // 最大並發數, 默認: 100
  NUM_WORKERS: number;                     // 圖片預處理工作者數, 默認: 64
  
  // 模型配置
  MODEL_PATH: string;                      // 默認: 'deepseek-ai/DeepSeek-OCR'
  GPU_MEMORY_UTILIZATION: number;          // 默認: 0.9
  MAX_MODEL_LEN: number;                   // 默認: 8192
  
  // 推論配置
  TEMPERATURE: number;                     // 默認: 0.0
  MAX_TOKENS: number;                      // 默認: 8192
  SKIP_REPEAT: boolean;                    // 跳過重複頁面, 默認: true
  
  // N-gram 防重複
  NGRAM_SIZE: number;                      // 默認: 20-40
  WINDOW_SIZE: number;                     // 默認: 50-90
  WHITELIST_TOKEN_IDS: Set;        // 白名單 Token (如 , )
}
```

---

## API 端點

### 1. OCR 圖片識別 (同步)

**端點**: `POST /api/v1/ocr/image`

**請求格式**:
```typescript
interface OCRImageRequest {
  // 圖片來源 (三選一)
  image_url?: string;           // 圖片 URL
  image_base64?: string;        // Base64 編碼圖片
  image_path?: string;          // 本地路徑 (僅服務器內部)
  
  // 處理選項
  prompt?: string;              // 默認: '\nConvert the document to markdown.'
  mode?: 'tiny' | 'small' | 'base' | 'large' | 'gundam';
  
  // 高級選項
  crop_mode?: boolean;          // 覆蓋默認裁切模式
  max_crops?: number;           // 最大裁切數量 (2-9)
  skip_repeat?: boolean;        // 跳過重複內容
  
  // Grounding 選項
  extract_bounding_boxes?: boolean;  // 提取邊界框座標
  extract_sub_images?: boolean;      // 提取子圖片
  draw_bounding_boxes?: boolean;     // 繪製邊界框
  
  // 元數據
  request_id?: string;
}
```

**回應格式**:
```typescript
interface OCRImageResponse {
  success: boolean;
  request_id: string;
  data: {
    // 文字內容
    text: string;                    // 提取的文字內容
    markdown?: string;               // Markdown 格式 (移除 grounding 標記)
    text_with_grounding?: string;    // 包含 grounding 標記的原始輸出
    
    // Grounding 結果
    grounding?: {
      bounding_boxes: Array<{
        label: string;               // 標籤類型 (如 'title', 'image', 'table')
        coordinates: number[][];     // [[x1,y1,x2,y2], ...] 歸一化座標 (0-999)
        absolute_coordinates?: number[][];  // 絕對座標 (像素)
      }>;
      sub_images?: Array<{
        index: number;
        label: string;
        base64?: string;             // 子圖片 Base64
        url?: string;                // 子圖片 URL
      }>;
      visualization?: {
        image_with_boxes_base64?: string;  // 繪製邊界框的圖片
        image_with_boxes_url?: string;
      };
    };
    
    // 處理信息
    processing_info: {
      mode: string;
      crop_enabled: boolean;
      num_crops: number;
      num_visual_tokens: number;
      processing_time_ms: number;
      
      // 裁切信息
      crop_ratio?: [number, number];  // [width_tiles, height_tiles]
    };
    
    // 圖片信息
    image_info: {
      width: number;
      height: number;
      format: string;
      size_bytes: number;
    };
  };
  timestamp: string;
}
```

### 2. OCR 圖片識別 (流式)

**端點**: `POST /api/v1/ocr/image/stream`

**請求格式**: 同上

**回應格式**: Server-Sent Events (SSE)
```typescript
// 事件類型
type StreamEvent = 
  | { type: 'start', data: { request_id: string } }
  | { type: 'token', data: { text: string, cumulative_text: string } }
  | { type: 'complete', data: OCRImageResponse }
  | { type: 'error', data: ErrorResponse };

// SSE 格式
// data: {"type":"token","data":{"text":"#","cumulative_text":"#"}}
// data: {"type":"token","data":{"text":" Title","cumulative_text":"# Title"}}
// data: {"type":"complete","data":{...}}
```

### 3. OCR PDF 識別

**端點**: `POST /api/v1/ocr/pdf`

**請求格式**:
```typescript
interface OCRPDFRequest {
  // PDF 來源 (三選一)
  pdf_url?: string;
  pdf_base64?: string;
  pdf_path?: string;
  
  // 處理選項
  prompt?: string;
  mode?: 'tiny' | 'small' | 'base' | 'large' | 'gundam';
  
  // PDF 特定選項
  page_range?: {
    start: number;              // 起始頁碼 (1-based)
    end?: number;               // 結束頁碼
  };
  pages?: number[];             // 指定頁碼列表
  dpi?: number;                 // PDF 轉圖片 DPI, 默認: 144
  
  // 高級選項
  crop_mode?: boolean;
  max_crops?: number;
  skip_repeat?: boolean;        // 跳過重複頁面 (EOS 檢測)
  
  // Grounding 選項
  extract_bounding_boxes?: boolean;
  extract_sub_images?: boolean;
  draw_bounding_boxes?: boolean;
  generate_annotated_pdf?: boolean;  // 生成標註 PDF
  
  // 元數據
  request_id?: string;
}
```

**回應格式**:
```typescript
interface OCRPDFResponse {
  success: boolean;
  request_id: string;
  data: {
    pages: Array<{
      page_number: number;
      text: string;
      markdown?: string;
      skipped?: boolean;         // 是否因重複而跳過
      skip_reason?: string;      // 'no_eos' | 'duplicate'
      
      grounding?: {
        bounding_boxes: Array;
        sub_images?: Array;
      };
      
      processing_info: {
        mode: string;
        num_crops: number;
        num_visual_tokens: number;
        processing_time_ms: number;
      };
    }>;
    
    // 整體信息
    summary: {
      total_pages: number;
      processed_pages: number;
      skipped_pages: number;
      total_processing_time_ms: number;
      total_text_length: number;
    };
    
    // 合併內容
    merged_content: {
      markdown: string;          // 所有頁面合併的 Markdown
      markdown_with_separators: string;  // 帶  分隔符
    };
    
    // PDF 信息
    pdf_info: {
      page_count: number;
      file_size_bytes: number;
    };
    
    // 附件 (如果請求)
    attachments?: {
      annotated_pdf_base64?: string;     // 標註邊界框的 PDF
      annotated_pdf_url?: string;
    };
  };
  timestamp: string;
}
```

### 4. 批量處理

**端點**: `POST /api/v1/ocr/batch`

**請求格式**:
```typescript
interface OCRBatchRequest {
  items: Array<{
    id: string;                 // 項目唯一 ID
    type: 'image' | 'pdf';
    source: string;             // URL 或 Base64
    source_type: 'url' | 'base64' | 'path';
    prompt?: string;
    mode?: string;
    
    // 可覆蓋全局選項
    crop_mode?: boolean;
    max_crops?: number;
    extract_bounding_boxes?: boolean;
  }>;
  
  // 批量選項
  batch_options?: {
    max_concurrent?: number;    // 最大並發數, 默認: 使用系統配置
    fail_fast?: boolean;        // 遇錯即停, 默認: false
    num_workers?: number;       // 預處理工作者數
  };
  
  request_id?: string;
}
```

**回應格式**:
```typescript
interface OCRBatchResponse {
  success: boolean;
  request_id: string;
  data: {
    results: Array;
    
    summary: {
      total: number;
      succeeded: number;
      failed: number;
      total_processing_time_ms: number;
      average_processing_time_ms: number;
    };
  };
  timestamp: string;
}
```

### 5. 健康檢查

**端點**: `GET /api/v1/health`

**回應格式**:
```typescript
interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  model: {
    loaded: boolean;
    path: string;
    mode: string;
    architecture: string;      // 'DeepseekOCRForCausalLM'
  };
  system: {
    gpu_available: boolean;
    gpu_count: number;
    gpu_memory_used_mb?: number;
    gpu_memory_total_mb?: number;
    gpu_utilization?: number;
    current_concurrency: number;
    max_concurrency: number;
  };
  config: {
    base_size: number;
    image_size: number;
    crop_mode: boolean;
    max_crops: number;
  };
  timestamp: string;
}
```

### 6. 配置管理

**端點**: `GET /api/v1/config`

**回應格式**:
```typescript
interface ConfigResponse {
  current_mode: string;
  available_modes: string[];
  config: SystemConfig;
  prompt_templates: {
    document: string;
    image: string;
    figure: string;
    general: string;
    free_ocr: string;
    table: string;
    form: string;
    recognition: string;
  };
  ngram_config: {
    ngram_size: number;
    window_size: number;
    whitelist_token_ids: number[];
  };
}
```

**端點**: `PUT /api/v1/config`

**請求格式**:
```typescript
interface ConfigUpdateRequest {
  mode?: 'tiny' | 'small' | 'base' | 'large' | 'gundam';
  max_concurrency?: number;
  max_crops?: number;
  skip_repeat?: boolean;
  ngram_config?: {
    ngram_size?: number;
    window_size?: number;
  };
}
```

---

## 資料結構定義

### 提示詞模板

```typescript
enum PromptTemplate {
  // 文檔處理
  DOCUMENT = '\nConvert the document to markdown.',
  IMAGE = '\nOCR this image.',
  FREE_OCR = '\nFree OCR.',  // 無 grounding 標記
  
  // 特殊內容
  FIGURE = '\nParse the figure.',
  TABLE = '\nExtract table data in markdown format.',
  FORM = '\nExtract form fields and values.',
  
  // 通用
  GENERAL = '\nDescribe this image in detail.',
  RECOGNITION = '\nLocate {target}<|/ref|> in the image.',
  
  // 科學內容
  CHEMISTRY = '\nExtract the structural formula.',  // 輸出 SMILES
  GEOMETRY = '\nExtract geometric data.',           // 輸出座標
}
```

### Grounding 標記格式

```typescript
// 輸出格式
type GroundingOutput = string;  // 'label<|/ref|>[[x1,y1,x2,y2],...]<|/det|>'

// 解析後格式
interface GroundingData {
  label: string;                // 'image' | 'title' | 'table' | 'figure' | ...
  coordinates: number[][];      // [[x1,y1,x2,y2], ...], 歸一化 0-999
}

// 正則表達式
const GROUNDING_PATTERN = /((.*?)<\|\/ref\|>(.*?)<\|\/det\|>)/g;
```

### 處理模式

```typescript
enum ProcessingMode {
  TINY = 'tiny',
  SMALL = 'small',
  BASE = 'base',
  LARGE = 'large',
  GUNDAM = 'gundam'
}

// 模式配置映射
const MODE_CONFIG: Record = {
  tiny: { base_size: 512, image_size: 512, crop_mode: false },
  small: { base_size: 640, image_size: 640, crop_mode: false },
  base: { base_size: 1024, image_size: 1024, crop_mode: false },
  large: { base_size: 1280, image_size: 1280, crop_mode: false },
  gundam: { base_size: 1024, image_size: 640, crop_mode: true },
};
```

### 圖片預處理

```typescript
interface ImagePreprocessResult {
  // 輸入
  input_ids: torch.LongTensor;           // [seq_len]
  
  // 圖片特徵
  pixel_values: torch.FloatTensor;       // [n_images, 3, base_size, base_size]
  images_crop: torch.FloatTensor;        // [n_images, n_patches, 3, image_size, image_size]
  images_seq_mask: torch.BoolTensor;     // [seq_len], 標記圖片 token 位置
  images_spatial_crop: torch.LongTensor; // [n_images, 2], [width_tiles, height_tiles]
  
  // 元數據
  num_image_tokens: number[];            // 每張圖片的 token 數量
  image_shapes: [number, number][];      // 原始圖片尺寸 (width, height)
}
```

---

## 錯誤處理

### 標準錯誤格式

```typescript
interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: any;
    request_id?: string;
    traceback?: string;  // 僅開發環境
  };
  timestamp: string;
}
```

### 錯誤代碼

| 代碼 | HTTP 狀態 | 說明 | 解決方案 |
|------|-----------|------|----------|
| `INVALID_INPUT` | 400 | 請求參數無效 | 檢查請求格式 |
| `MISSING_IMAGE` | 400 | 未提供圖片來源 | 提供 image_url/base64/path |
| `UNSUPPORTED_FORMAT` | 400 | 不支持的圖片格式 | 使用 JPG/PNG/JPEG |
| `INVALID_PROMPT` | 400 | 提示詞格式錯誤 | 確保包含 `<image>` 標記 |
| `IMAGE_TOO_LARGE` | 413 | 圖片過大 | 壓縮圖片或調整模式 |
| `PDF_CONVERSION_FAILED` | 422 | PDF 轉圖片失敗 | 檢查 PDF 檔案 |
| `MODEL_NOT_LOADED` | 503 | 模型未加載 | 等待模型初始化 |
| `GPU_OOM` | 503 | GPU 記憶體不足 | 降低 MAX_CROPS 或並發數 |
| `PROCESSING_TIMEOUT` | 504 | 處理超時 | 簡化圖片或分批處理 |
| `RATE_LIMIT_EXCEEDED` | 429 | 超過並發限制 | 降低請求頻率 |
| `INFERENCE_ERROR` | 500 | 推論失敗 | 檢查日誌 |

---

## 使用範例

### Python 客戶端完整實作

```python
import requests
import base64
from typing import Optional, List, Dict, Any
from enum import Enum

class PromptTemplate(Enum):
    DOCUMENT = '\\nConvert the document to markdown.'
    IMAGE = '\\nOCR this image.'
    FREE_OCR = '\\nFree OCR.'
    FIGURE = '\\nParse the figure.'
    TABLE = '\\nExtract table data in markdown format.'

class ProcessingMode(Enum):
    TINY = 'tiny'
    SMALL = 'small'
    BASE = 'base'
    LARGE = 'large'
    GUNDAM = 'gundam'

class DeepSeekOCRClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _load_image_as_base64(self, image_path: str) -> str:
        """載入圖片並轉為 Base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def ocr_image(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        prompt: str = PromptTemplate.DOCUMENT.value,
        mode: str = ProcessingMode.GUNDAM.value,
        extract_bounding_boxes: bool = False,
        extract_sub_images: bool = False,
        draw_bounding_boxes: bool = False,
        crop_mode: Optional[bool] = None,
        max_crops: Optional[int] = None,
    ) -> Dict[str, Any]:
        """OCR 圖片識別"""
        
        data = {
            "prompt": prompt,
            "mode": mode,
            "extract_bounding_boxes": extract_bounding_boxes,
            "extract_sub_images": extract_sub_images,
            "draw_bounding_boxes": draw_bounding_boxes,
        }
        
        if crop_mode is not None:
            data["crop_mode"] = crop_mode
        if max_crops is not None:
            data["max_crops"] = max_crops
        
        # 圖片來源
        if image_path:
            data["image_base64"] = self._load_image_as_base64(image_path)
        elif image_url:
            data["image_url"] = image_url
        elif image_base64:
            data["image_base64"] = image_base64
        else:
            raise ValueError("必須提供 image_path, image_url 或 image_base64")
        
        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/image",
            json=data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    def ocr_image_stream(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt: str = PromptTemplate.DOCUMENT.value,
        mode: str = ProcessingMode.GUNDAM.value,
    ):
        """OCR 圖片識別 (流式輸出)"""
        
        data = {"prompt": prompt, "mode": mode}
        
        if image_path:
            data["image_base64"] = self._load_image_as_base64(image_path)
        elif image_url:
            data["image_url"] = image_url
        else:
            raise ValueError("必須提供 image_path 或 image_url")
        
        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/image/stream",
            json=data,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # 解析 SSE
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    import json
                    data = json.loads(line[6:])
                    yield data
    
    def ocr_pdf(
        self,
        pdf_path: Optional[str] = None,
        pdf_url: Optional[str] = None,
        page_range: Optional[Dict[str, int]] = None,
        pages: Optional[List[int]] = None,
        mode: str = ProcessingMode.GUNDAM.value,
        dpi: int = 144,
        skip_repeat: bool = True,
        extract_bounding_boxes: bool = False,
        generate_annotated_pdf: bool = False,
    ) -> Dict[str, Any]:
        """OCR PDF 識別"""
        
        data = {
            "mode": mode,
            "dpi": dpi,
            "skip_repeat": skip_repeat,
            "extract_bounding_boxes": extract_bounding_boxes,
            "generate_annotated_pdf": generate_annotated_pdf,
        }
        
        if page_range:
            data["page_range"] = page_range
        if pages:
            data["pages"] = pages
        
        if pdf_path:
            with open(pdf_path, "rb") as f:
                data["pdf_base64"] = base64.b64encode(f.read()).decode()
        elif pdf_url:
            data["pdf_url"] = pdf_url
        else:
            raise ValueError("必須提供 pdf_path 或 pdf_url")
        
        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/pdf",
            json=data,
            timeout=600  # PDF 處理可能較慢
        )
        response.raise_for_status()
        return response.json()
    
    def batch_ocr(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        fail_fast: bool = False,
    ) -> Dict[str, Any]:
        """批量 OCR"""
        
        data = {
            "items": items,
            "batch_options": {
                "fail_fast": fail_fast,
            }
        }
        
        if max_concurrent:
            data["batch_options"]["max_concurrent"] = max_concurrent
        
        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/batch",
            json=data,
            timeout=1200
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """健康檢查"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        return response.json()
    
    def get_config(self) -> Dict[str, Any]:
        """獲取配置"""
        response = self.session.get(f"{self.base_url}/api/v1/config")
        return response.json()

# ============ 使用範例 ============

client = DeepSeekOCRClient(base_url="http://localhost:8000")

# 1. 基礎圖片 OCR
result = client.ocr_image(
    image_path="document.jpg",
    prompt=PromptTemplate.DOCUMENT.value,
    mode=ProcessingMode.GUNDAM.value
)
print("Markdown:", result["data"]["markdown"])

# 2. 帶 Grounding 的圖片 OCR
result = client.ocr_image(
    image_path="document.jpg",
    prompt=PromptTemplate.DOCUMENT.value,
    extract_bounding_boxes=True,
    extract_sub_images=True,
    draw_bounding_boxes=True,
)

# 提取邊界框
for bbox in result["data"]["grounding"]["bounding_boxes"]:
    print(f"Label: {bbox['label']}, Coords: {bbox['coordinates']}")

# 保存子圖片
for sub_img in result["data"]["grounding"]["sub_images"]:
    img_data = base64.b64decode(sub_img["base64"])
    with open(f"sub_image_{sub_img['index']}.jpg", "wb") as f:
        f.write(img_data)

# 3. 流式輸出
print("Streaming output:")
for event in client.ocr_image_stream(
    image_path="document.jpg",
    prompt=PromptTemplate.DOCUMENT.value
):
    if event["type"] == "token":
        print(event["data"]["text"], end="", flush=True)
    elif event["type"] == "complete":
        print("\n\nComplete!")

# 4. PDF OCR
pdf_result = client.ocr_pdf(
    pdf_path="report.pdf",
    page_range={"start": 1, "end": 5},
    mode=ProcessingMode.BASE.value,
    skip_repeat=True,
    generate_annotated_pdf=True,
)

print(f"Processed {pdf_result['data']['summary']['processed_pages']} pages")
print(f"Skipped {pdf_result['data']['summary']['skipped_pages']} pages")
print("Merged Markdown:", pdf_result["data"]["merged_content"]["markdown"])

# 5. 批量處理
items = [
    {
        "id": "img1",
        "type": "image",
        "source": "https://example.com/doc1.jpg",
        "source_type": "url",
        "mode": "gundam"
    },
    {
        "id": "img2",
        "type": "image",
        "source": "/path/to/local/doc2.jpg",
        "source_type": "path",
        "mode": "base"
    }
]

batch_result = client.batch_ocr(items, max_concurrent=10)
for item in batch_result["data"]["results"]:
    if item["success"]:
        print(f"{item['id']}: {item['result']['data']['markdown'][:100]}...")
    else:
        print(f"{item['id']}: Error - {item['error']['message']}")
```

### cURL 範例

```bash
# 1. 基礎圖片 OCR (使用 URL)
curl -X POST http://localhost:8000/api/v1/ocr/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/document.jpg",
    "prompt": "\\nConvert the document to markdown.",
    "mode": "gundam"
  }'

# 2. 圖片 OCR with Grounding
curl -X POST http://localhost:8000/api/v1/ocr/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANS...",
    "prompt": "\\nConvert the document to markdown.",
    "extract_bounding_boxes": true,
    "extract_sub_images": true,
    "draw_bounding_boxes": true
  }'

# 3. 流式輸出
curl -X POST http://localhost:8000/api/v1/ocr/image/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "image_url": "https://example.com/doc.jpg",
    "prompt": "\\nConvert the document to markdown."
  }'

# 4. PDF OCR
curl -X POST http://localhost:8000/api/v1/ocr/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_url": "https://example.com/report.pdf",
    "page_range": {"start": 1, "end": 5},
    "mode": "base",
    "dpi": 144,
    "skip_repeat": true
  }'

# 5. 健康檢查
curl http://localhost:8000/api/v1/health

# 6. 查詢配置
curl http://localhost:8000/api/v1/config

# 7. 更新配置
curl -X PUT http://localhost:8000/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "gundam",
    "max_concurrency": 50,
    "max_crops": 6
  }'
```

### TypeScript/JavaScript 範例

```typescript
interface OCRImageRequest {
  image_url?: string;
  image_base64?: string;
  prompt?: string;
  mode?: string;
  extract_bounding_boxes?: boolean;
  extract_sub_images?: boolean;
  draw_bounding_boxes?: boolean;
}

class DeepSeekOCRClient {
  constructor(private baseUrl: string = 'http://localhost:8000') {}
  
  async ocrImage(request: OCRImageRequest): Promise {
    const response = await fetch(`${this.baseUrl}/api/v1/ocr/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`OCR failed: ${error.error.message}`);
    }
    
    return response.json();
  }
  
  async *ocrImageStream(request: OCRImageRequest): AsyncGenerator {
    const response = await fetch(`${this.baseUrl}/api/v1/ocr/image/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      throw new Error(`Stream failed: ${response.statusText}`);
    }
    
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const text = decoder.decode(value);
      const lines = text.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          yield data;
        }
      }
    }
  }
  
  async ocrPDF(request: any): Promise {
    const response = await fetch(`${this.baseUrl}/api/v1/ocr/pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`PDF OCR failed: ${error.error.message}`);
    }
    
    return response.json();
  }
  
  async health(): Promise {
    const response = await fetch(`${this.baseUrl}/api/v1/health`);
    return response.json();
  }
}

// 使用範例
const client = new DeepSeekOCRClient();

// 1. 基礎 OCR
const result = await client.ocrImage({
  image_url: 'https://example.com/doc.jpg',
  mode: 'gundam',
  prompt: '\\nConvert the document to markdown.'
});
console.log(result.data.markdown);

// 2. 流式輸出
for await (const event of client.ocrImageStream({
  image_url: 'https://example.com/doc.jpg'
})) {
  if (event.type === 'token') {
    process.stdout.write(event.data.text);
  } else if (event.type === 'complete') {
    console.log('\nDone!');
  }
}

// 3. PDF OCR
const pdfResult = await client.ocrPDF({
  pdf_url: 'https://example.com/report.pdf',
  page_range: { start: 1, end: 10 },
  mode: 'base'
});
console.log(pdfResult.data.merged_content.markdown);
```

---

## 核心功能實作

### 1. 圖片動態裁切 (Dynamic Preprocessing)

**演算法**: `dynamic_preprocess()`

```python
def dynamic_preprocess(
    image: PIL.Image,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 640
) -> Tuple[List[PIL.Image], Tuple[int, int]]:
    """
    根據圖片長寬比動態裁切成多個塊
    
    Args:
        image: 原始圖片
        min_num: 最小裁切塊數
        max_num: 最大裁切塊數
        image_size: 每個塊的尺寸
    
    Returns:
        (裁切後的圖片列表, (width_tiles, height_tiles))
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # 計算所有可能的裁切比例
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # 找到最接近的裁切比例
    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    # 執行裁切
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    
    processed_images = []
    blocks = best_ratio[0] * best_ratio[1]
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    return processed_images, best_ratio
```

**範例**:
- 輸入: 1920×1080 圖片
- IMAGE_SIZE: 640
- 輸出: 3×2 = 6 個 640×640 的塊, crop_ratio=(3, 2)

### 2. N-gram 防重複機制

**演算法**: `NoRepeatNGramLogitsProcessor`

```python
class NoRepeatNGramLogitsProcessor:
    """防止生成重複的 N-gram"""
    
    def __init__(
        self,
        ngram_size: int = 20,
        window_size: int = 50,
        whitelist_token_ids: Set[int] = None
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
    
    def __call__(
        self,
        input_ids: List[int],
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if len(input_ids) < self.ngram_size:
            return scores
        
        # 當前前綴
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])
        
        # 在窗口內搜索重複的 n-gram
        search_start = max(0, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1
        
        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])
        
        # 白名單 token 不禁止 (如 , )
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        # 將禁止的 token 分數設為 -inf
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")
        
        return scores
```

**配置建議**:
- 批量處理: `ngram_size=40, window_size=90`
- 單圖處理: `ngram_size=30, window_size=90`
- PDF 處理: `ngram_size=20, window_size=50`
- 白名單: `{128821, 128822}` (對應 `<td>`, `</td>`)

### 3. Grounding 解析與處理

**解析 Grounding 標記**:

```python
import re
from typing import List, Tuple

def parse_grounding(text: str) -> Tuple[List[dict], List[str], List[str]]:
    """
    解析 grounding 標記
    
    Returns:
        (所有匹配, 圖片類型匹配, 其他類型匹配)
    """
    pattern = r'((.*?)<\|\/ref\|>(.*?)<\|\/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    matches_image = []
    matches_other = []
    
    for match in matches:
        if 'image<|/ref|>' in match[0]:
            matches_image.append(match[0])
        else:
            matches_other.append(match[0])
    
    parsed_matches = []
    for match in matches:
        try:
            label = match[1]
            coordinates = eval(match[2])  # [[x1,y1,x2,y2], ...]
            parsed_matches.append({
                'label': label,
                'coordinates': coordinates,
                'raw': match[0]
            })
        except:
            continue
    
    return parsed_matches, matches_image, matches_other

def remove_grounding_markers(text: str, matches_other: List[str]) -> str:
    """移除 grounding 標記,保留純文字"""
    for match in matches_other:
        text = text.replace(match, '')
    
    # 清理多餘換行
    text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    text = text.replace('', '').replace('', '')
    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    return text

def extract_coordinates(
    match: dict,
    image_width: int,
    image_height: int
) -> Tuple[str, List[List[int]]]:
    """
    將歸一化座標 (0-999) 轉換為絕對座標
    
    Returns:
        (label, [[x1,y1,x2,y2], ...])
    """
    label = match['label']
    coords_normalized = match['coordinates']
    
    coords_absolute = []
    for coord in coords_normalized:
        x1, y1, x2, y2 = coord
        x1 = int(x1 / 999 * image_width)
        y1 = int(y1 / 999 * image_height)
        x2 = int(x2 / 999 * image_width)
        y2 = int(y2 / 999 * image_height)
        coords_absolute.append([x1, y1, x2, y2])
    
    return label, coords_absolute
```

**繪製邊界框**:

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_bounding_boxes(
    image: Image.Image,
    grounding_matches: List[dict],
    output_path: str = None
) -> Image.Image:
    """在圖片上繪製邊界框"""
    
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # 半透明覆蓋層
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()
    
    for match in grounding_matches:
        label, coords = extract_coordinates(match, image_width, image_height)
        
        # 隨機顏色
        color = (
            np.random.randint(0, 200),
            np.random.randint(0, 200),
            np.random.randint(0, 255)
        )
        color_alpha = color + (20,)
        
        for x1, y1, x2, y2 in coords:
            # 繪製邊界框
            if label == 'title':
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                draw2.rectangle([x1, y1, x2, y2], fill=color_alpha, width=1)
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw2.rectangle([x1, y1, x2, y2], fill=color_alpha, width=1)
            
            # 繪製標籤
            text_x, text_y = x1, max(0, y1 - 15)
            draw.text((text_x, text_y), label, font=font, fill=color)
    
    img_draw.paste(overlay, (0, 0), overlay)
    
    if output_path:
        img_draw.save(output_path)
    
    return img_draw
```

**提取子圖片**:

```python
def extract_sub_images(
    image: Image.Image,
    grounding_matches: List[dict],
    output_dir: str = None
) -> List[dict]:
    """提取邊界框內的子圖片"""
    
    image_width, image_height = image.size
    sub_images = []
    img_idx = 0
    
    for match in grounding_matches:
        label = match['label']
        
        if label == 'image':  # 只提取標記為 'image' 的區域
            label, coords = extract_coordinates(match, image_width, image_height)
            
            for x1, y1, x2, y2 in coords:
                try:
                    cropped = image.crop((x1, y1, x2, y2))
                    
                    sub_img_data = {
                        'index': img_idx,
                        'label': label,
                        'coordinates': [x1, y1, x2, y2],
                        'image': cropped
                    }
                    
                    if output_dir:
                        save_path = f"{output_dir}/sub_image_{img_idx}.jpg"
                        cropped.save(save_path)
                        sub_img_data['path'] = save_path
                    
                    sub_images.append(sub_img_data)
                    img_idx += 1
                except Exception as e:
                    print(f"Failed to extract sub-image: {e}")
    
    return sub_images
```

### 4. PDF 處理流程

```python
import fitz  # PyMuPDF
import img2pdf
from PIL import Image
import io

def pdf_to_images(
    pdf_path: str,
    dpi: int = 144
) -> List[Image.Image]:
    """將 PDF 轉換為圖片列表"""
    
    images = []
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    pdf_document.close()
    return images

def images_to_pdf(
    images: List[Image.Image],
    output_path: str
):
    """將圖片列表合併為 PDF"""
    
    image_bytes_list = []
    
    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        image_bytes_list.append(img_buffer.getvalue())
    
    pdf_bytes = img2pdf.convert(image_bytes_list)
    
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
```

**重複頁面檢測**:

```python
def is_page_repeated(output_text: str, eos_token: str = '') -> bool:
    """
    檢測頁面是否因重複而未正常結束
    
    邏輯:
    - 如果輸出包含 EOS token: 正常頁面
    - 如果沒有 EOS token: 可能是重複頁面
    """
    return eos_token not in output_text

def process_pdf_with_skip(
    images: List[Image.Image],
    llm: Any,
    sampling_params: Any,
    skip_repeat: bool = True
) -> List[dict]:
    """處理 PDF 並跳過重複頁面"""
    
    results = []
    
    for idx, (output, img) in enumerate(zip(outputs_list, images)):
        content = output.outputs[0].text
        
        # 檢查是否重複
        if is_page_repeated(content):
            content = content.replace('', '')
            if skip_repeat:
                results.append({
                    'page_number': idx + 1,
                    'skipped': True,
                    'skip_reason': 'no_eos',
                    'text': content
                })
                continue
        
        results.append({
            'page_number': idx + 1,
            'skipped': False,
            'text': content
        })
    
    return results
```

---

## 最佳實踐

### 1. 模式選擇建議

| 場景 | 推薦模式 | 配置 |
|------|---------|------|
| 小型文檔 (< 1MB) | `small` | crop_mode=false |
| 標準文檔 (1-5MB) | `base` 或 `gundam` | crop_mode=true, max_crops=6 |
| 大型文檔 (> 5MB) | `gundam` | crop_mode=true, max_crops=6 |
| 高精度需求 | `large` | crop_mode=false |
| 快速預覽 | `tiny` | crop_mode=false |
| 表格密集 | `gundam` | 使用 TABLE prompt |
| 圖片多的文檔 | `gundam` | extract_sub_images=true |

### 2. 性能優化

**批量處理優化**:

```python
from concurrent.futures import ThreadPoolExecutor

def preprocess_images_parallel(
    images: List[Image.Image],
    processor: DeepseekOCRProcessor,
    num_workers: int = 64
) -> List[dict]:
    """並行預處理圖片"""
    
    def process_single(image):
        return {
            "prompt": PROMPT,
            "multi_modal_data": {
                "image": processor.tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=CROP_MODE
                )
            }
        }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(executor.map(process_single, images))
    
    return batch_inputs
```

**記憶體管理**:

| GPU 記憶體 | MAX_CROPS | MAX_CONCURRENCY | GPU_UTILIZATION |
|-----------|-----------|-----------------|-----------------|
| 16GB | 4 | 50 | 0.85 |
| 24GB | 6 | 100 | 0.9 |
| 32GB+ | 9 | 150 | 0.9 |

**OOM 錯誤處理**:

```python
import torch

def handle_gpu_oom(func):
    """裝飾器: 處理 GPU OOM"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            # 清理快取
            torch.cuda.empty_cache()
            
            # 降低並發數
            if 'max_num_seqs' in kwargs:
                kwargs['max_num_seqs'] = max(1, kwargs['max_num_seqs'] // 2)
            
            # 重試
            return func(*args, **kwargs)
    return wrapper
```

### 3. 錯誤處理範例

```python
import time
from requests.exceptions import RequestException

def ocr_with_retry(
    client: DeepSeekOCRClient,
    image_path: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """帶重試機制的 OCR"""
    
    for attempt in range(max_retries):
        try:
            return client.ocr_image(image_path=image_path)
        
        except RequestException as e:
            error_msg = str(e)
            
            if attempt == max_retries - 1:
                raise
            
            # GPU OOM: 指數退避
            if "GPU_OOM" in error_msg:
                wait_time = backoff_factor ** attempt * 5
                print(f"GPU OOM, waiting {wait_time}s...")
                time.sleep(wait_time)
            
            # Rate limit: 固定等待
            elif "RATE_LIMIT" in error_msg:
                time.sleep(2)
            
            # 其他錯誤: 立即失敗
            else:
                raise

# 使用
try:
    result = ocr_with_retry(client, "large_document.jpg")
except Exception as e:
    print(f"Failed after retries: {e}")
```

### 4. 提示詞優化

**針對不同內容類型**:

```python
PROMPT_TEMPLATES = {
    # 文檔類型
    "document": "\\nConvert the document to markdown.",
    "table": "\\nExtract table data in markdown format.",
    "form": "\\nExtract form fields and values.",
    
    # 無 Grounding
    "free_ocr": "\\nFree OCR.",
    
    # 特殊內容
    "figure": "\\nParse the figure.",
    "chemistry": "\\nExtract the structural formula.",
    "geometry": "\\nExtract geometric data.",
    
    # 通用
    "general": "\\nDescribe this image in detail.",
}

# 使用範例
result = client.ocr_image(
    image_path="invoice.jpg",
    prompt=PROMPT_TEMPLATES["form"]
)
```

### 5. 批量處理最佳實踐

```python
# 大批量處理: 分批提交
def batch_ocr_large_dataset(
    image_paths: List[str],
    batch_size: int = 50
):
    """大規模批量處理"""
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        items = [
            {
                "id": f"img_{i+j}",
                "type": "image",
                "source": path,
                "source_type": "path",
                "mode": "gundam"
            }
            for j, path in enumerate(batch)
        ]
        
        batch_result = client.batch_ocr(items, max_concurrent=20)
        results.extend(batch_result["data"]["results"])
        
        # 進度顯示
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)}")
    
    return results
```

### 6. 特殊場景處理

**處理化學結構式 (SMILES)**:

```python
result = client.ocr_image(
    image_path="molecule.jpg",
    prompt="\\nExtract the structural formula."
)

# 輸出可能包含  標籤
if '' in result["data"]["text"]:
    smiles = result["data"]["text"].split('')[1].split('')[0]
    print(f"SMILES: {smiles}")
```

**處理幾何圖形**:

```python
result = client.ocr_image(
    image_path="geometry.jpg",
    prompt="\\nExtract geometric data."
)

# 解析幾何數據
if 'Line' in result["data"]["text"]:
    geo_data = eval(result["data"]["text"])
    lines = geo_data['Line']['line']
    endpoints = geo_data['Line']['line_endpoint']
    print(f"Lines: {lines}")
```

### 7. 監控與日誌

```python
import logging
from datetime import datetime

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ocr_with_logging(client, image_path, **kwargs):
    """帶日誌的 OCR"""
    start_time = datetime.now()
    
    try:
        result = client.ocr_image(image_path=image_path, **kwargs)
        
        processing_time = result["data"]["processing_info"]["processing_time_ms"]
        
        logger.info(
            f"OCR Success | "
            f"Image: {image_path} | "
            f"Time: {processing_time:.2f}ms | "
            f"Text Length: {len(result['data']['text'])}"
        )
        
        return result
    
    except Exception as e:
        logger.error(
            f"OCR Failed | "
            f"Image: {image_path} | "
            f"Error: {str(e)}"
        )
        raise
```

---

## 附錄

### A. 支持的圖片格式

| 格式 | 支持 | 備註 |
|------|-----|------|
| JPEG/JPG | ✅ | 推薦 |
| PNG | ✅ | 推薦 |
| BMP | ⚠️ | 需轉換為 JPG |
| TIFF | ⚠️ | 需轉換為 JPG |
| WebP | ⚠️ | 需轉換為 JPG |
| PDF | ✅ | 使用 PDF 端點 |

### B. Token 白名單

用於 N-gram 防重複機制:

| Token ID | Token | 用途 |
|----------|-------|------|
| 128821 | `<td>` | 表格單元格開始 |
| 128822 | `</td>` | 表格單元格結束 |

### C. 性能基準

基於 NVIDIA A100 (40GB):

| 模式 | 圖片大小 | 平均處理時間 | GPU 記憶體 | 吞吐量 (imgs/s) |
|------|----------|--------------|-----------|----------------|
| Tiny | 512×512 | ~0.5s | ~2GB | ~2.0 |
| Small | 640×640 | ~0.8s | ~3GB | ~1.2 |
| Base | 1024×1024 | ~1.5s | ~5GB | ~0.7 |
| Large | 1280×1280 | ~2.5s | ~8GB | ~0.4 |
| Gundam | 可變 | ~2-4s | ~6-10GB | ~0.3-0.5 |

**批量處理性能**:
- 並發數 50: ~15-20 imgs/s
- 並發數 100: ~25-30 imgs/s

### D. 常見問題排查

**1. GPU OOM**:
```python
# 降低配置
config_update = {
    "max_crops": 4,  # 從 6 降到 4
    "max_concurrency": 50  # 從 100 降到 50
}
client.session.put(f"{base_url}/api/v1/config", json=config_update)
```

**2. 推論超時**:
```python
# 增加 timeout
client.session = requests.Session()
client.session.request = lambda *args, **kwargs: requests.request(
    *args, **{**kwargs, 'timeout': 300}
)
```

**3. 重複頁面過多**:
```python
# 調整 N-gram 配置
config_update = {
    "ngram_config": {
        "ngram_size": 30,  # 增加 ngram_size
        "window_size": 120  # 增加 window_size
    }
}
```

### E. Docker 部署

**Dockerfile**:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安裝 Python
RUN apt-get update && apt-get install -y \\
    python3.10 python3-pip git

# 安裝依賴
COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install -r requirements.txt

# 複製程式碼
COPY . /app/

# 下載模型 (可選)
# RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR')"

EXPOSE 8000

CMD ["python3", "serve_ocr.py"]
```

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  deepseek-ocr:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MAX_CONCURRENCY=100
      - CROP_MODE=true
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**運行**:

```bash
# 構建
docker-compose build

# 啟動
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 測試
curl http://localhost:8000/api/v1/health
```

---

## 版本變更日誌

### v2.0.0 (2025-11-04)
- ✅ 新增流式輸出支持
- ✅ 新增 Grounding 視覺定位功能
- ✅ 新增 PDF 批量處理
- ✅ 新增 N-gram 防重複機制
- ✅ 優化動態裁切演算法
- ✅ 完整的錯誤處理體系

### v1.0.0 (2025-11-01)
- ✅ 初始版本
- ✅ 基礎圖片 OCR
- ✅ 多模式支持

---

## 維護與支持

**維護者**: Yueh-Chun Hsieh  
**聯絡方式**: ocr-support@example.com  
**文檔倉庫**: https://github.com/your-org/deepseek-ocr-api  
**問題追蹤**: https://github.com/your-org/deepseek-ocr-api/issues  

**技術支持**:
- Slack: #deepseek-ocr
- Email: wilson5711704@gmail.com
- 文檔: https://docs.example.com/deepseek-ocr

---

**最後更新**: 2025-11-04  
**文檔版本**: 2.0.0
    "