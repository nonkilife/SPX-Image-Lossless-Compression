# SPX (Space Express)：高吞吐量無損影像壓縮引擎

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) ![MSE](https://img.shields.io/badge/MSE-0.00000000-red) ![Version](https://img.shields.io/badge/version-8.3.2-orange) ![Speed](https://img.shields.io/badge/Speed-150_MB/s-brightgreen) ![Savings](https://img.shields.io/badge/Savings-28%25-blueviolet)

SPX (Space Express) 是一個採用 **Python/Rust 混合架構** 的無損影像壓縮引擎，具備 **熵分片 (Entropy Sharding)** 與 **Rayon 加速的 4 路交織 rANS (4-way Interleaved rANS)** 技術，旨在壓縮率與速度之間取得最佳平衡。本專案透過上下文分片與原生計算內核，實現了與現代標準相當的壓縮率，並將 Python 的靈活性與 Rust 的原生效能完美結合。

[繁體中文版本 (Traditional Chinese)](./README_zh-TW.md)

---

## 目錄
1. [v8.3.2 效能快照](#v832-效能快照)
2. [技術分析](#v832-技術分析-混合-rust-架構)
3. [與既有格式比較](#2-與既有格式比較)
4. [系統需求與安裝](#3-系統需求與安裝)
5. [快速開始](#4-快速開始)
6. [技術架構與執行流程](#5-技術架構與執行流程)
7. [效能基準測試](#6-效能基準測試-v8x-統一中心)
8. [對比基準測試](#7-對比基準測試-clic--div2k--tecnick--kodak)
9. [限制與發展路線](#8-限制與發展路線)
10. [數據集來源](#9-數據集來源)
11. [專案背景](#10-專案背景)
12. [致謝](#11-致謝)

---

SPX 的核心設計原則：
> **最大化單位算力的壓縮效率。**

不同於不計代價追求極限壓縮率的方案，SPX 專注於：
- **可預測的效能**：執行時間與輸入解析度呈線性關係 ($O(N)$)。
- **單次編碼 (Single-Pass)**：非迭代執行，無須暴力搜索。
- **極簡建模複雜度**：無狀態（Stateless）的單一模型流水線。
- **高吞吐量**：經過 ILP 優化的原生計算內核。

### 設計哲學
| 維度 | SPX 方案 |
| :--- | :--- |
| **壓縮率** | 具競爭力（與現代無損標準一致） |
| **速度** | $O(N)$ 複雜度（非迭代） |
| **複雜度** | 極小化（無狀態流水線） |
| **確定性** | 絕對確定 |
| **多重掃描** | 否 |

### 關鍵特性
*   **單次編碼**：消除迭代優化迴圈。
*   **確定性流水線**：恆定的執行路徑，無須啟發式搜索。
*   **降低模型複雜度**：單一模型的上下文映射，無須切換。
*   **吞吐量中心設計**：針對每個時鐘週期的最大像素吞吐量進行優化。
*   **原生 Rust 後端**：具備可預測運行效能的零成本抽象。
*   **可擴展架構**：分片邊界與 rANS 概率模板的模組化配置，以適應特殊的數據分佈。

---

### v1.0.0 效能快照

以下數據描述了各個標準數據集上的吞吐量與壓縮效率。

| 數據集 | 類型 | SPX BPP | **節省 (vs PNG)** | **節省 (vs PNM)** | **SPX 編碼速度** | WebP (M6) 速度 | JXL (E7) 速度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Kodak** | RGB | **9.79** | **-24.64 %** | **-59.19 %** | **41.24 MB/s** | 0.33 MB/s | 5.65 MB/s |
| **CLIC '25** | RGB | **8.06** | **-28.32 %** | **-66.43 %** | **69.10 MB/s** | 1.18 MB/s | 5.02 MB/s |
| **CLIC '21** | RGB | **8.46** | **-28.03 %** | **-64.74 %** | **72.22 MB/s** | 0.97 MB/s | 5.62 MB/s |
| **DIV2K Val**| 2K | **9.22** | **-27.32 %** | **-61.59 %** | **80.76 MB/s** | 1.28 MB/s | 5.83 MB/s |
| **DIV2K Train**| 2K | **9.35** | **-26.22 %** | **-61.04 %** | **66.97 MB/s** | 1.38 MB/s | 6.15 MB/s |
| **Tecnick** | RGB | **5.18** | **-25.90 %** | **-78.42 %** | **25.12 MB/s** | 0.68 MB/s | 4.75 MB/s |
| **Tecnick** | Gray | **1.68** | **-27.63 %** | **-79.01 %** | **14.20 MB/s** | 0.29 MB/s | 5.53 MB/s |
| **Standard (ICI)** | RGB | **10.51** | **n/a** | **-56.19 %** | **150.81 MB/s** | 2.16 MB/s | 10.84 MB/s |
| **Standard (ICI)** | Gray | **3.44** | **n/a** | **-56.99 %** | **53.06 MB/s** | 0.58 MB/s | 12.16 MB/s |

> [!NOTE]
> **硬體基準測試環境**：
> - **CPU**: AMD Ryzen 5 3500X (6-Core, 3.60 GHz)
> - **RAM**: 32.0 GB
> - **OS**: Windows 11 (64-bit, x64)

#### **技術對比**
- **編碼速度**：在受測硬體上，SPX 的吞吐量比 WebP (Method 6) 高出 **25x–150x**，比 JPEG-XL (Effort 7) 高出 **3x–14x**。
- **完美無損**：所有 1,500+ 張測試影像均實現位元完美重建 (**MSE = 0.00000000**)。
- **核心效率**：原生 Rust 後端利用 **Rayon** 實現內部數據並行，並使用 **4 路交織 rANS** 實現指令級並行 (ILP)。
- **混合效能**：關鍵熱路徑（RCT、MED、分片、rANS）均以 Rust 實作，協調層則保留在 Python。

*完整對比分析請參閱 [Comparative Benchmarks](./technical/BENCHMARK.md)。*

---

### v1.0.0 技術分析 (混合 Rust 架構)
- **核心工作流**：統一流水線，包含 **基礎協議** $\rightarrow$ **Rust 預測內核** $\rightarrow$ **Rust 空間轉換** $\rightarrow$ **Rust 無狀態分片**。
- **預測中心 (Pillar 2)**：解耦的中心化設計，由 `predictor.py` 負責協調，`rans_core.rs` 執行 **無分支邊緣調整 MED (Branchless Edge-Tuned MED)**，大幅提升執行效率。
- **上下文感知路徑架構**：
    - **RGB 路徑 (Pillar 3/4)**：原生 Rust G-sub RCT 轉換配合無狀態分片矩陣。
    - **灰階快速路徑**：專門的單色旁路，利用序列化的綠色通道隔離。
- **無狀態分片中心 (Pillar 4)**：使用 `ShardProfile`「配置即數據」模型進行統一的上下文 ID 推導，由原生 Rust "Gather" 內核執行。
- **BICC (偏差消除)**：對殘差應用上下文驅動的 PDF 中心化，以減少離散度。
- **Bitplane rANS**：層次化熵建模，使用 **2,688 路上下文模型**（42 分片 x 64 空間模式），完全以 Rust 實作。
- **4 路交織 rANS 核心**：利用指令級並行 (ILP) 的向量化原生熵引擎。

---

## 2. 與既有格式比較

- **壓縮率**：比標準 PNG **減少約 25-30%**；在高解析度攝影圖片上與 WebP (m6) 相當。
- **效率**：無狀態分片為高頻噪點和低熵漸層提供穩定的效能表現。
- **解碼**：原生 Rust 解碼吞吐量約為 **20–130 MB/s**（取決於圖像複雜度）。可透過多核批處理擴展。

---

## 3. 系統需求與安裝

- **Python 版本**：3.10+ (建議 3.11+)
- **Python 套件**：
  - `numpy>=1.22.0`, `zstandard>=0.19.0`, `Pillow>=9.0.0`, `pytest>=7.0.0`
- **原生擴充**：
  - `spx_rans` (原生 Rust 後端)
- **開發依賴**：
  - `maturin>=1.0.0` (用於銜接 Rust 與 Python)
  - **Rust 工具鏈**：`cargo`, `rustc` (從源碼構建時需要)
- **系統核心 (Linux)**：
  - 需要 `zlib` 與 `libpng` 標頭檔以供 **Pillow** 與 **Zstd** 使用 (`sudo apt install build-essential zlib1g-dev libpng-dev`)。
- **Windows**：自我包含 (從源碼構建時需要 Visual Studio C++ Build Tools)。

> [!TIP]
> **原生加速**：SPX v8.3.2 使用預編譯的 Rust 後端。與舊版本不同，第一次執行時 **零 JIT 延遲**。
> **多執行緒**：並行處理在 Rust 後端內部使用 Rayon 庫自動處理。

**安裝步驟**：
```bash
# 1. 安裝 Python 依賴
pip install numpy>=1.22.0 zstandard>=0.19.0 Pillow>=9.0.0 pytest>=7.0.0

# 2. 構建/安裝原生擴充
cd native && maturin develop --release
```

---

## 4. 快速開始

### 4.1 命令列介面 (CLI)
```bash
# 壓縮
python main.py compress input.png --optimize

# 解壓
python main.py decompress input.spx --output restored.png

# 基準測試 (僅限 SPX)
python main.py benchmark ./path/to/images -n 20 -w 8

# 基準測試 (對比 SPX vs WebP vs JXL)
python main.py benchmark ./path/to/images --codec bench -n 20
```

### 4.2 Python API
```python
from core import compress_spx, decompress_spx

# 1. 壓縮影像 (RGB/RGBA)
result = compress_spx("input.png", "output.spx", use_bitplane=False)
print(f"壓縮率: {result.ratio:.2%} | 時間: {result.enc_time:.2f}s")

# 2. 解壓影像
with open("output.spx", "rb") as f: payload = f.read()
rgb_arr, dec_time = decompress_spx(payload, "reconstructed.png")
print(f"解壓時間: {dec_time:.2f}s")
```

### 4.3 Windows 批處理工具 (`test.bat`)
為 Windows 用戶提供了方便的封裝工具：
```powershell
# 執行對比基準測試 (SPX vs WebP vs JXL)
.\test bench ./data/local_test_folder
.\test webp ./data/local_test_folder
.\test jxl ./data/local_test_folder

# 執行特定編碼器的測試
.\test spx ./data/local_test_folder

# 傳遞額外參數 (例如限制 10 張影像)
.\test spx ./my_images -n 10
```
> [!NOTE]
> 本倉庫 **不包含** `data/` 目錄與基準測試數據集。要執行基準測試，您可以手動建立 `data/` 資料夾並放入影像（如 Kodak, DIV2K），或直接引用影像資料夾的路徑。該工具支援原始絕對/相對路徑，以及在 **`core/test_suite.py`** 中管理的預定義別名（如 `clic`, `kodak`, `trgb`）。

### 4.4 專案結構
```text
.
├── core/                   # SPX 四柱核心引擎 (Python 協調層)
│   ├── codec.py            # 位元流調度與序列化
│   ├── sharding.py         # 支柱 4: 分片配置與無狀態中心
│   ├── rans.py             # 支柱 4: 4 路交織 rANS 核心
│   ├── predictor.py        # 支柱 2: 無分支 MED 內核
│   ├── transform.py        # 支柱 3: G-sub RCT 與空間運算
│   ├── common.py           # 支柱 1: 協議常量與標誌
│   └── env.py              # 環境與依賴驗證
├── technical/              # 演算法深度規格文檔
├── data/                   # [用戶提供] 基準測試數據集目錄 (不包含在倉庫中)
├── native/                 # [實驗性] Rust 加速後端
├── test.bat                # Windows 基準測試工具
└── main.py                 # CLI 入口點
```

---

## 5. 技術架構與執行流程

SPX 遵循嚴格定義的 **四柱架構 (4-Pillar Architecture)**：

### 5.1 SPX 的四大支柱
1. **支柱 1: 空間轉換 (RCT)**：透過綠色減法 RCT (`transform.py`) 處理色彩去相關。
2. **支柱 2: 空間預測 (MED)**：透過無分支邊緣調整 MED (`predictor.py`) 執行空間去相關。
3. **支柱 3: 無狀態分片**：將殘差映射到統計上下文（分片）以進行優先編碼 (`sharding.py`)。
4. **支柱 4: 熵編碼 (rANS)**：透過 4 路交織 rANS 引擎 (`rans.py`) 執行統計壓縮。

這些支柱由 **基礎協議** (`common.py`) 統一管理，該協議定義了位元流架構與編碼閾值。

```mermaid
graph TD
    A[輸入: 8-bit RGB/RGBA] --> B{協議檢查}
    B -->|灰階| C1[Rust: 融合灰階路徑]
    B -->|彩色| C2[Rust: 支柱 1 & 2 融合內核]
    
    C1 & C2 --> D[Rust: 支柱 3 分片中心]
    
    D --> E[BICC 偏差消除]
    E --> F[Rust: 支柱 4 rANS 引擎]
    F --> G[v8.3.2 SPX 位元流輸出]
    
    subgraph "Rust 擴充 (spx_rans)"
        C1
        C2
        D
        F
    end
```

### 5.2 深度技術系列
詳細演算法規格請參閱 `technical/` 目錄下的文檔：
*   [**01. PREDICTOR.md**](./technical/1.%20PREDICTOR.md)：空間調度器與 MED 變體的詳細邏輯。
*   [**02. SHARD_TEMPLATE.md**](./technical/2.%20SHARD_TEMPLATE.md)：Universal-42 矩陣與上下文推導規格。
*   [**03. RANS_MODE.md**](./technical/3.%20RANS_MODE.md)：4 路交織 rANS 核心架構與概率建模。
*   [**04. DATASET_FINGERPRINT.md**](./technical/4.%20DATASET_FINGERPRINT.md)：工業數據集的統計效能剖析。

### 5.3 雙路徑策略
SPX 實作了 **上下文感知旁路 (Context-Aware Bypass)** 邏輯，以處理不同類型的影像：
* **RGB 路徑**：利用 **G-sub RCT**，先提取綠色基礎（Lead），隨後處理紅/藍殘差（Lag）。使用交錯處理窗口以保持通道間的上下文一致性。
* **灰階路徑**：若檢測到 R=G=B，引擎會激活專門的單色旁路，將計算開銷降低約 65%。此路徑利用 **Bitplane rANS** 將 8-bit 信號分解為層次化圖層，以提高冗餘提取效率。

### 5.4 無狀態分片與配置驅動中心
SPX 的骨幹是 **無狀態分片中心**，根據 V-Tier（梯度強度）、亮度（Intensity）和趨勢（Trend）將像素映射到 42+ 個上下文中。這種「配置即數據」的模型允許在不重新編譯內核的情況下無縫切換 Profile。

在熵編碼方面，引擎使用了 **30 模式模板矩陣**：
- **10 個基礎質心**：源自真實影像分片的數據驅動概率形狀。
- **3 個 Sigma 縮放**：每個質心以 `0.5`, `1.0`, `1.5` 進行縮放，以適應不同噪點水平。
- **零開銷**：這 30 個經驗模式硬編碼在解碼器中，無須傳輸自定義頻率表。

### 5.5 Bitplane rANS 與熵核心
對於高密度影像，SPX 採用 **分片條件 Bitplane rANS**。它將信號分解為 2-bit 圖層，每層使用 **2,688 路上下文模型** ($42 \text{ 分片} \times 64 \text{ 空間模式}$)，允許 rANS 核心將結構化可預測位元與隨機噪點位元隔離。

**交織 rANS** 引擎透過在單個迴圈中管理 4 個獨立狀態變量，利用 ILP 提升 CPU 執行單元的利用率。

### 5.6 記憶體占用與擴展性
SPX 旨在實現低延遲處理，並具備可預測的記憶體占用：
- **峰值 RAM (1080p RGB)**: ~85 MB
- **峰值 RAM (4K RGB)**: ~320 MB
- **峰值 RAM (8K RGB)**: ~1.2 GB
*註：記憶體使用量與像素數呈線性關係。*

---

## 6. 效能基準測試 (v8.x 統一中心)
  
當前引擎使用 **SPX 統一中心** 進行測試，提供與 WebP (Method 6) 和 JPEG-XL (Effort 7) 的對比分析。

### 6.1 綜合指標
詳細的效能數據（包括壓縮節省率、吞吐量 MB/s 和競爭勝率）維護在 [technical/BENCHMARK.md](./technical/BENCHMARK.md)。

### 6.2 基準版本
為確保可重複性，競爭基準鎖定在以下版本：
- **WebP (Method 6)**: `cwebp` v1.3.2 (libwebp v1.3.2)。
- **JPEG-XL (Effort 7)**: `cjxl` v0.8.2 (libjxl v0.8.2)。

### 6.3 使用方法
```bash
# 使用主入口點
python main.py benchmark C:\datasets\my_images -n 50

# Windows 快捷方式 (Batch)
.\test bench ./local_folder -n 50
```

---

## 7. 對比基準測試 (CLIC / DIV2K / TECNICK / KODAK)

詳細的效能指標以及與 WebP、JPEG-XL 在工業數據集上的對比基準測試，請參閱獨立的基準測試文檔：

👉 **[查看對比基準測試 (BENCHMARK.md)](./technical/BENCHMARK.md)**

---

## 8. 限制與發展路線

- **位元深度**：目前限制為每通道 **8-bit**。
- **色彩空間**：針對 **RGB** 優化。不支援 CMYK 或 YCbCr 子採樣。
- **Alpha 通道**：雖然支援 RGBA，但 Alpha 通道目前使用傳統的 Zstd 壓縮（Level 1），未來計畫引入高效 rANS 優化。
- **執行緒**：Python 協調層為單執行緒；但 Rust 後端在熱路徑內核中使用 **Rayon** 實現數據並行。
- **效能上限**：目前的吞吐量是透過無分支算法設計與 LLVM 自動向量化實現的。**目前沒有手寫的 SIMD (AVX2/NEON) 實作**。
- **內容偏向 (Content Bias)**：目前的基準測試主要針對 **「自然攝影影像」**。由於缺乏專門的測試數據集，SPX 在 **「人工合成內容」**（如螢幕截圖、UI 介面或電腦繪圖）上的效能尚未經過大規模驗證。處理具有高頻邊緣的人工圖案時，表現可能與照片有所不同。本專案作為高效能基準，若尋求極限優化，可考慮進一步的手動彙編優化。

---

## 9. 數據集來源

要驗證基準測試或使用標準數據集測試引擎，您可以從以下官方來源下載影像：

- **DIV2K Data Set**: [ETH Zurich CVL](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Kodak Data Set**: [Kaggle - Kodak Dataset](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset/data)
- **Clic Data Set**: [Kaggle - CLIC Dataset](https://www.kaggle.com/datasets/mustafaalkhafaji95/clic-dataset?resource=download)
- **Tecnick Data Set**: [SourceForge - TestImages](https://sourceforge.net/projects/testimages/files/SAMPLING/)
- **Standard Test Images (ICI)**: [Image Compression Info](https://imagecompression.info/test_images/)

---

## 10. 專案背景

SPX 專案是一個透過多階段研究週期開發的無損影像壓縮框架。本專案利用代理 AI **Claude Code** 與 **Antigravity** 來構建技術組件，包括 **四柱架構**、Universal-42 分片以及 4 路交織 rANS 熵編碼引擎。在 v8.3.2 中，核心計算內核從 Python/Numba 遷移到了 **原生 Rust 後端**，實現了顯著更高的吞吐量並減少了運行時 JIT 開銷。

此倡議作為 **AI 輔助工程 (AI-assisted engineering)** 的技術概念驗證，證明了自主代理能夠協助處理複雜的演算法優化與多語言系統整合。

## 11. 致謝
- **熵編碼**：本專案使用了由 Jarosław (Jarek) Duda 博士開發的 rANS 演算法。
- **空間預測**：空間去相關引擎使用了最初在 LOCO-I (JPEG-LS) 標準中引入的邊緣檢測器 (MED) 演算法。

---

**當前版本:** v8.3.2 | **授權:** Apache 2.0 | **MSE 目標:** 0.00000000
