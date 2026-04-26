# SPX: Context-Sensitive Data Engine

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) ![MSE](https://img.shields.io/badge/MSE-0.00000000-red) ![Version](https://img.shields.io/badge/version-8.3.2-orange) ![Speed](https://img.shields.io/badge/Speed-150_MB/s-brightgreen) ![Savings](https://img.shields.io/badge/Savings-28%25-blueviolet)

An implementation of lossless image compression using a **Hybrid Python/Rust Architecture**, featuring **Entropy Sharding** and **Rayon-accelerated 4-way Interleaved rANS**. This project exhibits comparable compression ratios through contextual sharding and native computational kernels, bridging Python's flexibility with Rust's native performance.

---

### v8.3.2 Performance Snapshot

The following data characterizes the throughput and compression efficiency across standard datasets (Baseline v7.5.0).

| Dataset | Type | SPX BPP | **Savings (vs PNG)** | **Savings (vs PNM)** | **SPX Enc Speed** | WebP (M6) Speed | JXL (E7) Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Kodak** | RGB | **9.79** | **-24.64 %** | **-59.19 %** | **41.24 MB/s** | 0.33 MB/s | 5.65 MB/s |
| **CLIC '25** | RGB | **8.06** | **-28.32 %** | **-66.43 %** | **69.10 MB/s** | 1.18 MB/s | 5.02 MB/s |
| **CLIC '21** | RGB | **8.46** | **-28.03 %** | **-64.74 %** | **72.22 MB/s** | 0.97 MB/s | 5.62 MB/s |
| **DIV2K Val**| 2K | **9.22** | **-27.32 %** | **-61.59 %** | **80.76 MB/s** | 1.28 MB/s | 5.83 MB/s |
| **DIV2K Train**| 2K | **9.35** | **-26.22 %** | **-61.04 %** | **66.97 MB/s** | 1.38 MB/s | 6.15 MB/s |
| **Tecnick** | RGB | **5.18** | **-25.90 %** | **-78.42 %** | **25.12 MB/s** | 0.68 MB/s | 4.75 MB/s |
| **Tecnick** | Gray | **1.68** | **-27.63 %** | **-79.01 %** | **14.20 MB/s** | 0.29 MB/s | 5.53 MB/s |
| **Waterloo** | RGB | **10.51** | **n/a** | **-56.19 %** | **150.81 MB/s** | 2.16 MB/s | 10.84 MB/s |
| **Waterloo** | Gray | **3.44** | **n/a** | **-56.99 %** | **53.06 MB/s** | 0.58 MB/s | 12.16 MB/s |

> [!NOTE]
> **Hardware Benchmark Environment**:
> - **CPU**: AMD Ryzen 5 3500X (6-Core, 3.60 GHz)
> - **RAM**: 32.0 GB
> - **OS**: Windows 11 (64-bit, x64)

#### **Technical Comparison**
- **Encoding Speed**: SPX exhibits **25x–150x higher throughput** than WebP (Method 6) and **5x–7x higher throughput** than JPEG-XL (Effort 7) on the tested hardware.
- **Quality Assurance**: Bit-perfect reconstruction across all 1,500+ test images (**MSE = 0.00000000**).
- **Core Efficiency**: Rust-native backend utilizes **Rayon** for internal data parallelism and **4-way interleaved rANS** for instruction-level parallelism (ILP).
- **Hybrid Performance**: Critical hot-paths (RCT, MED, Sharding, rANS) are implemented in Rust, while orchestration remains in Python.

*Full comparative analysis vs. standard formats is available in [Official Benchmarks](./technical/BENCHMARK.md).*

---

### v8.3.2 Technical Analysis (Hybrid Rust Architecture)
- **Core Workflow**: Unified pipeline featuring **Foundational Protocol** $\rightarrow$ **Rust Prediction Kernels** $\rightarrow$ **Rust Spatial Transforms** $\rightarrow$ **Rust Stateless Sharding**.
- **Predictor Hub (Pillar 2)**: Decoupled hub in `predictor.py` (orchestration) and `rans_core.rs` (execution) featuring **Branchless Edge-Tuned MED** for increased execution efficiency.
- **Context-Aware Path Architecture**: 
    - **RGB Path (Pillar 3/4)**: Rust-native G-sub RCT transform coupled with a stateless sharding matrix.
    - **Grayscale Fast-Path**: Specialized monochrome bypass utilizing serialized Green-channel isolation.
- **Stateless Sharding Hub (Pillar 4)**: Unified profile-driven context ID derivation using the `ShardProfile` configuration-as-data model, executed via Rust-native "Gather" kernels.
- **BICC (Bias Cancellation)**: Context-driven PDF centering applied to residuals to reduce dispersion.
- **Bitplane rANS**: Hierarchical entropy modeling using a **2,688-way context model** (42 Shards x 64 Spatial Patterns), fully implemented in Rust.
- **4-Way Interleaved rANS Core**: Vectorized native entropy engine utilizing instruction-level parallelism (ILP).

---

## 2. Comparison with Existing Formats

- **Compression**: **~25-30% reduction** compared to standard PNG; comparable with WebP (m6) on high-resolution photography.
- **Efficiency**: Stateless sharding provides a stable performance profile for both high-frequency noise and low-entropy gradients.
- **Decoding**: Rust-native decompression throughput (~60–130 MB/s). Scalable via multi-core batching.

---

## 3. System Requirements & Installation

- **Python Version**: 3.10+ (3.11+ recommended)
- **Python Packages**:
  - `numpy>=1.22.0`, `zstandard>=0.19.0`, `Pillow>=9.0.0`, `pytest>=7.0.0`
- **Native Extension**:
  - `spx_rans` (Rust-native backend)
- **Development Dependencies**:
  - `maturin>=1.0.0` (for bridging Rust and Python)
  - **Rust Toolchain**: `cargo`, `rustc` (required to build the native extension from source)
- **System Core (Linux)**:
  - Requires `zlib` and `libpng` headers for **Pillow** and **Zstd** I/O (`sudo apt install build-essential zlib1g-dev libpng-dev`).
- **Windows**: Self-contained (requires Visual Studio C++ Build Tools if building from source).

> [!TIP]
> **Native Acceleration**: SPX v8.3.2 utilizes a pre-compiled Rust backend. Unlike previous versions, there is **zero JIT latency** during the first run.
> **Multithreading**: Parallelism is handled internally by the Rust backend using the Rayon library.

**Installation**:
```bash
# 1. Install Python dependencies
pip install numpy>=1.22.0 zstandard>=0.19.0 Pillow>=9.0.0 pytest>=7.0.0

# 2. Build/Install native extension
cd native && maturin develop --release
```

---

## 4. Quick Start

### 4.1 Command Line Interface (CLI)
```bash
# Compress
python main.py compress input.png --optimize

# Decompress
python main.py decompress input.spx --output restored.png

# Benchmark (SPX only)
python main.py benchmark ./path/to/images -n 20 -w 8

# Benchmark (Compare SPX vs WebP vs JXL)
python main.py benchmark ./path/to/images --codec bench -n 20
```

### 4.2 Python API
```python
from core import compress_spx, decompress_spx

# 1. Compress Image (RGB/RGBA)
result = compress_spx("input.png", "output.spx", use_bitplane=False)
print(f"Ratio: {result.ratio:.2%} | Time: {result.enc_time:.2f}s")

# 2. Decompress Image
with open("output.spx", "rb") as f: payload = f.read()
rgb_arr, dec_time = decompress_spx(payload, "reconstructed.png")
print(f"Dec Time: {dec_time:.2f}s")
```

### 4.3 Windows Batch Utility (`test.bat`)
For Windows users, a convenient batch wrapper is provided for benchmarking:
```powershell
# Run comparative benchmark (SPX vs WebP vs JXL)
.\test bench ./data/kodak
.\test webp ./data/local_test
.\test jxl ./data/local_test

# Run solo tests for specific codecs
.\test spx ./data/local_test

# Pass additional arguments (e.g., limit to 10 images)
.\test bench ./my_images -n 10
```
> [!NOTE]
> The `data/` directory and benchmark datasets are **not included** in this repository. To run benchmarks, you must create a `data/` folder and populate it with your own images (e.g., Kodak, DIV2K). The utility supports both raw absolute/relative paths and pre-defined aliases if the corresponding data is present.

### 4.4 Project Structure
```text
.
├── core/                   # SPX 4-Pillar Core Engine (Python Orchestration)
│   ├── codec.py            # Bitstream orchestration & serialization
│   ├── sharding.py         # Pillar 4: Shard profiles & Stateless Hub
│   ├── rans.py             # Pillar 4: 4-way interleaved rANS core
│   ├── predictor.py        # Pillar 2: Branchless MED kernels
│   ├── transform.py        # Pillar 3: G-sub RCT & Spatial ops
│   ├── common.py           # Pillar 1: Protocol constants & Flags
│   └── env.py              # Environment & Dependency validator
├── technical/              # Deep-dive algorithmic specifications
├── data/                   # [User-provided] Directory for benchmark datasets (not in repo)
├── native/                 # [Experimental] Rust-accelerated backend
├── test.bat                # Windows benchmark utility
└── main.py                 # CLI entry point
```

---

## 5. Technical Architecture & Execution Flow

SPX follows a strictly defined **4-Pillar Architecture** to transform raw pixels into a bit-perfect compressed stream:

### 5.1 The 4 Pillars of SPX
1. **Pillar 1: Spatial Transforms (RCT)**: Handles color decorrelation via the Green-Subtract RCT (`transform.py`).
2. **Pillar 2: Spatial Prediction (MED)**: Performs spatial decorrelation via Branchless Edge-Tuned MED (`predictor.py`).
3. **Pillar 3: Stateless Sharding**: Maps residuals into statistical contexts (shards) for prioritized coding (`sharding.py`).
4. **Pillar 4: Entropy Coding (rANS)**: Executes statistical compression via the 4-way interleaved rANS engine (`rans.py`).

These pillars are governed by the **Foundational Protocol** (`common.py`), which defines the bitstream schema and coding thresholds.

```mermaid
graph TD
    A[Input: 8-bit RGB/RGBA] --> B{Protocol Gate: Flag Check}
    B -->|Grayscale| C1[Rust: Fused Gray Pass]
    B -->|Color| C2[Rust: Pillar 1 & 2 Fused Kernel]
    
    C1 & C2 --> D[Rust: Pillar 3 Sharding Hub]
    
    D --> E[BICC Bias Cancellation]
    E --> F[Rust: Pillar 4 rANS Engine]
    F --> G[v8.3.2 SPX Bitstream Output]
    
    subgraph "Rust Extension (spx_rans)"
        C1
        C2
        D
        F
    end
```

### 5.2 Deep-Dive Technical Series

For detailed algorithmic specifications, refer to the following documentation in the `technical/` directory:

*   [**01. PREDICTOR.md**](./technical/1.%20PREDICTOR.md): Detailed logic of the Spatial Dispatcher and MED variations.
*   [**02. SHARD_TEMPLATE.md**](./technical/2.%20SHARD_TEMPLATE.md): Specification of the Universal-42 matrix and context derivation.
*   [**03. RANS_MODE.md**](./technical/3.%20RANS_MODE.md): Architecture of the 4-way Interleaved rANS core and probability modeling.
*   [**04. DATASET_FINGERPRINT.md**](./technical/4.%20DATASET_FINGERPRINT.md): Statistical performance profiling across industrial datasets.

### 5.3 Dual-Path Strategy
SPX implements a **Context-Aware Bypass** logic to handle different image types with optimal efficiency:

* **RGB Route**: Utilizing the **G-sub RCT**, it extracts a Green foundation (Lead) followed by RD/BD residuals (Lag). It uses a staggered processing window to maintain context consistency across channels.
* **Grayscale Route**: If R=G=B is detected, the engine activates a specialized monochrome bypass, reducing computational overhead by ~65%. This path utilizes **Bitplane rANS**, which decomposes the 8-bit signal into hierarchical layers to increase redundancy extraction efficiency.

### 5.4 Stateless Sharding & Profile-Driven Hub
The backbone of SPX is the **Stateless Sharding Hub**, mapping pixels into 42+ contexts based on V-Tier (gradient strength), Intensity, and Trend. This configuration-as-data model allows for seamless profile switching without kernel recompilation.

For entropy coding, the engine utilizes a **30-Mode Template Matrix**:
- **6 Base Centroids**: Data-driven probability shapes derived from 6,000+ real-world image shards.
- **5 Sigma Scales**: Each centroid is scaled from `0.5` to `1.5` to adapt to different noise levels.
- **Zero-Overhead**: These 30 empirical modes are hardcoded in the decoder, allowing optimal PDF matching without the "Header Tax" of custom frequency tables.

### 5.5 Bitplane rANS & Entropy Core
For high-density images, SPX employs **Shard-Conditioned Bitplane rANS**. Instead of treating the residual as a single 256-symbol alphabet, it decomposes the signal into 2-bit layers. Each layer uses a **2,688-way context model** ($42 \text{ Shards} \times 64 \text{ Spatial Patterns}$), allowing the rANS core to isolate structural predictable bits from stochastic noise bits.

The **Interleaved rANS** engine increases throughput by managing 4 independent state variables in a single loop, increasing CPU execution port utilization via ILP.

### 5.6 Memory Profile & Scalability
SPX is designed for low-latency processing with a predictable memory footprint.
- **Peak RAM (1080p RGB)**: ~85 MB
- **Peak RAM (4K RGB)**: ~320 MB
- **Peak RAM (8K RGB)**: ~1.2 GB
*Note: Memory usage scales linearly with pixel count. Peak values include native Rust buffers and Python object overhead.*

### 5.7 Cache Residency & Hot-Path Efficiency
SPX is optimized for L1/L2 cache residency to avoid memory stalls:
- **L1 (Hot Path)**: rANS Models, ZigZag LUTs, Branchless Predictors.
- **L2 (Context Hub)**: Stateless Spatial LUT (256 KB), Shard Histograms.
- **L3 (Bulk Data)**: Image Channel Buffers and Bitstream Payloads.
*Full technical audit available in [technical/CACHE.md](./technical/CACHE.md).*

### 5.8 Extensibility: Custom Shard Profiles
Pillar 4 (Stateless Sharding) allows adding new segmentation strategies without logic changes:
1.  **Define Boundaries**: Create new `V_BOUND` and `INTENSITY_SEG` numpy arrays in `sharding.py`.
2.  **Map Shards**: Implement a `build_shard_map_custom()` function to assign Context IDs.
3.  **Precompute LUTs**: Call `precompute_luts()` to generate the O(1) dispatch tables.
4.  **Register Profile**: Instantiate a new `ShardProfile` and update the `PROFILE_RGB` alias.

### 5.9 Developer Setup & Debugging
- **Regression Testing**: Run `pytest` to verify bit-perfect parity and logic integrity.
- **Log Verbosity**: Set `SPX_LOG_LEVEL=DEBUG` for detailed pipeline tracing.
- **Diagnostic Dumps**: Use `SPX_DUMP_SHARDS=1` to audit raw residual distributions.
- **Thread Safety**: Core modules utilize `threading.local()` for scratch-buffer management; ensure `clear_spx_workspaces()` is called in long-running server workers.

---

## 6. Performance Benchmarking (v8.x Unified Hub)
 
 The current engine is benchmarked using the **SPX Unified Hub**, providing comparative analysis against WebP (Method 6) and JPEG-XL (Effort 7).

### 6.1 Comprehensive Metrics
  Detailed performance data, including compression savings, throughput (MB/s), and competitive win rates, are maintained in [technical/BENCHMARK.md](./technical/BENCHMARK.md).

### 6.2 Benchmark Baseline Versions
To ensure reproducibility, the competitive baselines are locked to the following versions:
- **WebP (Method 6)**: `cwebp` v1.3.2 (libwebp v1.3.2).
- **JPEG-XL (Effort 7)**: `cjxl` v0.8.2 (libjxl v0.8.2).
- **PNG (Optimize)**: `zopflipng` v1.0.3.

### 6.3 Usage
```bash
# Use the main entry point
python main.py benchmark C:\datasets\my_images -n 50

# Windows Shortcut (Batch)
.\test bench ./local_folder -n 50
```

### 6.4 Technical Control Arguments
Common arguments supported by the benchmarking suite:

| Argument | Full Name | Description | Example |
| :--- | :--- | :--- | :--- |
| **-n** | `--num_tests` | Limits the number of processed files. | `-n 50` |
| **--offset** | `--offset` | Skips the first N images in the set. | `--offset 100` |
| **-w** | `--workers` | Manually sets the number of CPU cores. | `-w 8` |
| **--codec** | `--codec` | Selects codec: `spx`, `webp`, `jxl`, `bench`. | `--codec bench` |
| **--reclassify**| `--reclassify`| Categorize images into Easy/Hard/Hell folders. | `--reclassify` |
| **--bitplane** | `--bitplane` | Force the Bitplane engine for the benchmark. | `--bitplane` |
| **--build** | `--build` | Assemble dataset: `PATH E H HELL`. | `--build my_set 10 10 5` |

#### **Advanced CLI Examples**
```powershell
# 1. Analyze a specific slice of a large dataset
python main.py benchmark C:\data -n 100 --offset 500

# 2. Extract specific difficulty levels from a results run
# This copies processed files into ./data/DIV2K_Easy, etc.
python main.py benchmark ./my_images --reclassify

# 3. Create a synthetic dataset (10 Easy, 10 Hard, 5 Hell images)
# Resulting images are saved to './data/balanced_set'
python main.py benchmark --build balanced_set 10 10 5
```

---

## 7. Comparative Benchmarks (CLIC / DIV2K / TECNICK / KODAK)

Detailed performance metrics and comparative benchmarks comparing SPX against WebP and JPEG-XL across industrial datasets are available in the independent benchmark document:

👉 **[View Comparative Benchmarks (BENCHMARK.md)](./technical/BENCHMARK.md)**

---

## 8. Limitations & Roadmap

- **Bit Depth**: Currently limited to **8-bit** per channel.
- **Color Spaces**: Optimized for **RGB**. No support for CMYK or YCbCr subsampling (G-sub transform is native to RGB).
- **Alpha Channel**: While RGBA is supported, the Alpha channel currently utilizes traditional Zstd compression (Level 1) rather than the high-performance rANS sharding engine used for RGB. Optimization for constant alpha (solidity detection) is a planned future improvement.
- **Threading**: The Python orchestration layer is single-threaded; however, the Rust-native backend utilizes internal data-parallelism via **Rayon** for hot-path kernels (RCT, Sharding, rANS).

## 9. Dataset Sources

To verify the benchmarks or test the engine with standard datasets, you can download the images from the following official sources:

- **DIV2K Data Set - Train & Validation**: [ETH Zurich CVL](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- **Kodak Data Set**: [Kaggle - Kodak Dataset](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset/data)

- **Clic Data Set**: [Kaggle - CLIC Dataset](https://www.kaggle.com/datasets/mustafaalkhafaji95/clic-dataset?resource=download)

- **Tecnick Data Set**: [SourceForge - TestImages](https://sourceforge.net/projects/testimages/files/SAMPLING/)

- **Waterloo Data Set**: [Image Compression Info](https://imagecompression.info/test_images/)

---

## 10. Project Background

The SPX project is a lossless image compression framework developed through a multi-phase research cycle. The project utilized the agentic AI **Antigravity** to architect technical components, including the **Four-Pillar Architecture**, Universal-42 Sharding, and a 4-way interleaved rANS entropy engine. In v8.3.2, the core computational kernels were migrated from Python/Numba to a **Rust-native backend**, achieving significantly higher throughput and reducing runtime JIT overhead. This initiative serves as a technical proof-of-concept for AI-assisted engineering, demonstrating that autonomous agents can assist in complex algorithmic optimization and multi-language systems integration.

## 11. Acknowledgments
- **Entropy Coding**: This project utilizes the rANS algorithm developed by Dr. Jarosław (Jarek) Duda. His work on Asymmetric Numeral Systems (ANS) provided the mathematical foundation for the entropy core.
- **Spatial Prediction**: The spatial decorrelation engine utilizes the Median Edge Detector (MED) algorithm, originally introduced in the LOCO-I (JPEG-LS) standard.

---

**Current Version:** v8.3.2 | **License:** Apache 2.0 | **MSE Target:** 0.00000000
