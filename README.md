[中文版](./README_ZH.md)

# ZPNG: Python-based Lossless Image Compression

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-GPL--3.0-green) ![MSE](https://img.shields.io/badge/MSE-0.000-red)

A high-performance lossless image compression pipeline implemented in Python with Numba JIT acceleration, utilizing the **Flexible-Shard Architecture** for hierarchical decorrelation and adaptive entropy sharding.

![ZPNG G-sub Visualization](./zpng_layers.png)
*Visualizing the G-sub Pipeline: Source -> Green Foundation (Grn) -> Modular G-sub Residuals (RD/BD)*

---

This approach manages data representation for lossless compression.

> [!IMPORTANT]
> **Focused on 8-bit RGB/RGBA**: ZPNG is architected specifically for the universal standard of 8-bit digital imaging used in web, game assets, and professional digital photography workflows.

### v6.2.0 Highlights (Flexible-Shard Architecture)
- **Universal Sharding Hub (USH)**: Unified matrix-driven context ID derivation using the Universal-42 architecture.
- **Context-Aware Path Architecture**: 
    - **RGB Path**: Universal-42 sharding matrix optimized for cross-channel chrominance residuals.
    - **Grayscale Fast-Path**: Hardware-accelerated monochrome bypass utilizing serialized Green-channel isolation.
- **Bitplane rANS (Experimental)**: Hierarchical entropy modeling that treats 8-bit planes as distinct context layers, maximizing redundancy extraction in smooth monochrome gradients.
- **BICC v2 (Bias Cancellation)**: Intelligent PDF centering applied to the Green foundation to minimize dispersion.
- **4-Way Interleaved rANS Core**: Branchless decoding achieving >85 MB/s throughput in JIT environments.

---

## 2. Comparison with Existing Formats

- **Compression**: **~25-30% smaller** than standard PNG; competitive with WebP (m6) on high-resolution photography.
- **Efficiency**: Universal-42 sharding provides a balanced profile for both high-frequency noise and low-entropy gradients.
- **Decoding**: Throughput (~25–35 MB/s) on a single thread; scalable up to 100+ MB/s via multi-core batching.

---

## 3. System Requirements & Installation

- **Python Version**: 3.10+ (3.11+ recommended)
- **Python Packages**:
  - `numpy>=1.22.0`, `numba>=0.57.0`, `zstandard>=0.19.0`, `Pillow>=9.0.0`, `pytest>=7.0.0`
- **System Core (Linux)**:
  - Requires `zlib` and `libpng` headers for **Pillow** and **Zstd** I/O (`sudo apt install build-essential zlib1g-dev libpng-dev`).
- **Windows**: Self-contained.

> [!WARNING]
> **JIT Latency**: Initial execution involves a ~30s JIT compilation delay (Numba). Subsequent runs are near-instant.
> **Pro Tip**: Set the environment variable `NUMBA_CACHE_DIR` to a local directory to persist compiled kernels across sessions.

**Installation**:
```bash
pip install numpy>=1.22.0 numba>=0.57.0 zstandard>=0.19.0 Pillow>=9.0.0 pytest>=7.0.0
```

---

## 4. Quick Start

### 4.1 CLI Usage (Recommended)
```bash
# Compress
python main.py compress input.png --optimize

# Decompress
python main.py decompress input.zpng --output restored.png

# Benchmark (Parallel)
python main.py benchmark ./data/gold -n 20 -w 8
```

### 4.2 API Usage
```python
from core import compress_csde, decompress_csde

# 1. Compress Image (RGB/RGBA)
result = compress_csde("input.png", "output.zpng", use_bitplane=False)
print(f"Ratio: {result.ratio:.2%} | Time: {result.enc_time:.2f}s")

# 2. Decompress Image
with open("output.zpng", "rb") as f: payload = f.read()
rgb_arr, dec_time = decompress_csde(payload, "reconstructed.png")
print(f"Dec Time: {dec_time:.2f}s")
```

---

## 5. Technical Architecture

ZPNG follows the **Flexible-Shard Architecture**:
- **Pillar 1: G-sub RCT**: Median-based reversible color transform extracting a stable Green channel foundation.
- **Pillar 2: MED Predictor**: Spatial decorrelation using the Median Edge Detector.
- **Pillar 3: Flexible Sharding Hub (FSH)**: Dynamic 42-shard context mapping based on local intensity and gradient strength (V-Tier).
- **Pillar 4: Interleaved rANS**: 4-way interleaved entropy core for high-instruction-level parallelism.

```mermaid
graph TD
    A[Input: 8-bit RGB/RGBA] --> B{Path Detection}
    B -->|Grayscale| C1[Grayscale Bypass: Pure Green Path]
    B -->|Color| C2[G-sub RCT: RGB Path]
    
    C1 --> D[Universal-42 Sharding Matrix]
    C2 --> D
    
    D --> E[BICC Bias Cancellation]
    E --> F[Hierarchical Bitplane rANS Engine]
    F --> G[ZPNG Bitstream]
```

### 5.1 Dual-Path Strategy
ZPNG implements a **Context-Aware Bypass**. If the encoder detects R=G=B (true monochrome), it activates the Grayscale Path, pruning 66% of computational overhead by bypassing chrominance residual calculations. This path is further optimized for **Bitplane rANS**, which isolates high-order bitplanes (MSB) from low-order noise planes (LSB).

### 5.2 Universal-42 Sharding Hub
The backbone of ZPNG is the **Universal-42 matrix**. It maps local pixel environments into one of 42 high-fidelity contexts based on:
1.  **V-Tier (Strength)**: 8 levels of local gradient magnitude.
2.  **Intensity Logic**: 3 exposure segments (Dark/Normal/Light).
3.  **Trend Awareness**: Straight vs. Diagonal edge detection.

### 5.3 Bitplane rANS Concepts
For extreme monochrome density, ZPNG explores **Bitplane-Level Entropy Isolation**. Instead of treating an 8-bit residual as a single symbol, the engine can decompose the residual into 2D bit contexts. This allows the rANS core to apply distinct probability models to predictable structural bits vs. unpredictable stochastic noise.

---

## 6. Performance Benchmarking (v6.x Unified Hub)
 
 The current engine is benchmarked using the **ZPNG Unified Hub**, providing objective head-to-head comparisons against WebP (Method 6) and JPEG-XL (Effort 7).

### 6.1 Comprehensive Metrics
 Detailed performance data, including compression savings, throughput (MB/s), and competitive win rates, are maintained in **Section 7: Official Benchmarks**.

### 6.2 Usage
```bash
# Use the main entry point
python main.py benchmark ./data/gold -n 50
```

### 6.3 Technical Control Arguments
Common arguments supported by the benchmarking suite:

| Argument | Full Name | Description | Example |
| :--- | :--- | :--- | :--- |
| **-n** | `--num_tests` | Limits the number of processed files. | `python zpng_imgtest.py ./data -n 50` |
| **--offset** | `--offset` | Skips the first N images in the set. | `python zpng_imgtest.py ./data --offset 100` |
| **-w** | `--workers` | Manually sets the number of CPU cores. | `python zpng_cwebp.py ./data -w 8` |
| **-v** | `--verbose` | (imgtest only) Displays detailed logs for each file. | `python zpng_imgtest.py ./data -v` |
| **-rc** | `--recategorize`| (imgtest only) Categorize images by difficulty.| `python zpng_imgtest.py ./data -rc` |
| **--csv** | `--csv` | (imgtest only) Custom path for the results CSV. | `python zpng_imgtest.py ./data --csv test.csv` |
| **--research** | `--research` | (imgtest only) Show G-sub channel redundancy analysis. | `python zpng_imgtest.py ./data --research` |

---

## 7. Official Benchmarks (CLIC / VAL / TRAIN)

| CLIC (60 imgs)       | ZPNG               | WebP(M6)           | JXL(E7)            |   
|----------------------|--------------------|--------------------|--------------------|   
| Original Size        |     204.01 MB      |     204.01 MB      |     204.01 MB      |   
| Compressed Size      |     139.01 MB      |     144.84 MB      |     132.81 MB      |   
| Savings (%)          |      31.86 %       |      29.00 %       |      34.90 %       |   
| BPP                  |        7.4004      |        7.7108      |        7.0707      |   
| Mean Ratio (%)       |      68.14 %       |      71.00 %       |      65.10 %       |   
| Median Ratio (%)     |      67.04 %       |      70.26 %       |      63.78 %       |   
| Ratio Range (%)      |  56.2-90.9 %       |  61.2-84.1 %       |  55.0-83.9 %       |   
| Avg Enc Time         |     1646.9 ms      |    20795.2 ms      |     3722.8 ms      |   
| Avg Dec Time         |      132.8 ms      |       90.5 ms      |      546.8 ms      |   
| TP: Compress         |       9.74 MB/s    |       0.94 MB/s    |       5.06 MB/s    |   
| TP: Decompress       |     120.71 MB/s    |     215.04 MB/s    |      34.43 MB/s    |   
| Core Eff (C)         |       2.06 MB/s    |       0.16 MB/s    |       0.91 MB/s    |   
| Core Eff (D)         |      25.60 MB/s    |      37.56 MB/s    |       6.22 MB/s    |   
| Wins: Space          |             5      |             0      |            55      |   
| Wins: Encode         |            58      |             0      |             2      |   
| Wins: Decode         |             0      |            60      |             0      |   
| Wall-clock           |      22.64 s       |     218.87 s       |      46.26 s       |   
| MSE (Quality)        |    0.00000000      |    0.00000000      |    0.00000000      |  

<br>

| DIV2K VAL (100 imgs) | ZPNG               | WebP(M6)           | JXL(E7)            |   
|----------------------|--------------------|--------------------|--------------------|   
| Original Size        |     428.68 MB      |     428.68 MB      |     428.68 MB      |   
| Compressed Size      |     311.58 MB      |     314.98 MB      |     292.36 MB      |   
| Savings (%)          |      27.32 %       |      26.52 %       |      31.80 %       |   
| BPP                  |        9.2194      |        9.3200      |        8.6507      |   
| Mean Ratio (%)       |      72.68 %       |      73.48 %       |      68.20 %       |   
| Median Ratio (%)     |      72.63 %       |      73.44 %       |      67.59 %       |   
| Ratio Range (%)      |  59.4-93.4 %       |  63.5-89.1 %       |  53.7-85.8 %       |   
| Avg Enc Time         |     1761.9 ms      |    17500.8 ms      |     4290.2 ms      |   
| Avg Dec Time         |      146.8 ms      |      102.3 ms      |      599.2 ms      |   
| TP: Compress         |      11.61 MB/s    |       1.32 MB/s    |       5.76 MB/s    |   
| TP: Decompress       |     139.43 MB/s    |     226.07 MB/s    |      41.27 MB/s    |   
| Core Eff (C)         |       2.43 MB/s    |       0.24 MB/s    |       1.00 MB/s    |   
| Core Eff (D)         |      29.21 MB/s    |      41.92 MB/s    |       7.15 MB/s    |   
| Wins: Space          |             3      |             2      |            95      |   
| Wins: Encode         |            99      |             0      |             1      |   
| Wins: Decode         |             0      |           100      |             0      |   
| Wall-clock           |      39.98 s       |     326.44 s       |      84.75 s       |   
| MSE (Quality)        |    0.00000000      |    0.00000000      |    0.00000000      |  

<br>

|DIV2K TRAIN (800 imgs)| ZPNG               | WebP(M6)           | JXL(E7)            |
|----------------------|--------------------|--------------------|--------------------|
| Original Size        |    3370.06 MB      |    3370.06 MB      |    3370.06 MB      |
| Compressed Size      |    2494.22 MB      |    2499.93 MB      |    2333.46 MB      |
| Savings (%)          |      25.99 %       |      25.82 %       |      30.76 %       |
| BPP                  |        9.3808      |        9.4022      |        8.7761      |
| Mean Ratio (%)       |      74.01 %       |      74.18 %       |      69.24 %       |
| Median Ratio (%)     |      73.81 %       |      74.13 %       |      69.15 %       |
| Ratio Range (%)      |  49.9-96.6 %       |  49.3-91.4 %       |  46.9-88.3 %       |
| Avg Enc Time         |     2064.9 ms      |    17543.9 ms      |     3836.5 ms      |
| Avg Dec Time         |      150.7 ms      |       93.9 ms      |      540.8 ms      |
| TP: Compress         |      11.09 MB/s    |       1.42 MB/s    |       6.46 MB/s    |
| TP: Decompress       |     151.88 MB/s    |     265.98 MB/s    |      45.83 MB/s    |
| Core Eff (C)         |       2.04 MB/s    |       0.24 MB/s    |       1.10 MB/s    |
| Core Eff (D)         |      27.95 MB/s    |      44.87 MB/s    |       7.79 MB/s    |
| Wins: Space          |            17      |            22      |           761      |
| Wins: Encode         |           794      |             0      |             6      |
| Wins: Decode         |             2      |           798      |             0      |
| Wall-clock           |     326.14 s       |    2380.16 s       |     595.26 s       |
| MSE (Quality)        |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| TECNICK RGB (40 imgs)| ZPNG               | WebP(M6)           | JXL(E7)            |   
|----------------------|--------------------|--------------------|--------------------|   
| Original Size        |     191.98 MB      |     191.98 MB      |     191.98 MB      |   
| Compressed Size      |     148.94 MB      |     148.18 MB      |     131.95 MB      |   
| Savings (%)          |      22.42 %       |      22.82 %       |      31.27 %       |   
| BPP                  |        5.4229      |        5.3950      |        4.8043      |   
| Mean Ratio (%)       |      77.58 %       |      77.18 %       |      68.73 %       |   
| Median Ratio (%)     |      77.61 %       |      77.89 %       |      70.63 %       |   
| Ratio Range (%)      |  69.5-94.3 %       |  69.9-89.2 %       |  59.2-89.8 %       |   
| Avg Enc Time         |     2934.7 ms      |    41539.6 ms      |     5582.3 ms      |   
| Avg Dec Time         |      452.1 ms      |      196.6 ms      |     1143.2 ms      |   
| TP: Compress         |       7.43 MB/s    |       0.66 MB/s    |       4.72 MB/s    |   
| TP: Decompress       |      48.22 MB/s    |     138.90 MB/s    |      23.04 MB/s    |   
| Core Eff (C)         |       1.64 MB/s    |       0.12 MB/s    |       0.86 MB/s    |   
| Core Eff (D)         |      10.62 MB/s    |      24.42 MB/s    |       4.20 MB/s    |   
| Wins: Space          |             0      |             6      |            34      |   
| Wins: Encode         |            40      |             0      |             0      |   
| Wins: Decode         |             0      |            40      |             0      |   
| Wall-clock           |      29.83 s       |     293.50 s       |      49.01 s       |   
| MSE (Quality)        |    0.00000000      |    0.00000000      |    0.00000000      |  

<br>

|TECNICK GRAY (40 imgs)| ZPNG               | WebP(M6)           | JXL(E7)            |
|-------------------------------------------------------------------------------------|
| Original Size        |      63.72 MB      |      63.72 MB      |      63.72 MB      |
| Compressed Size      |      49.00 MB      |      54.57 MB      |      43.65 MB      |
| Savings (%)          |      23.10 %       |      14.36 %       |      31.51 %       |
| BPP                  |        1.7841      |        1.9869      |        1.5891      |
| Mean Ratio (%)       |      76.90 %       |      85.64 %       |      68.49 %       |
| Median Ratio (%)     |      75.13 %       |      85.21 %       |      68.59 %       |
| Ratio Range (%)      |  70.4-85.5 %       |  80.3-96.7 %       |  62.2-84.0 %       |
| Avg Enc Time         |     1586.6 ms      |    36433.6 ms      |     4477.7 ms      |
| Avg Dec Time         |      116.9 ms      |      144.3 ms      |      553.6 ms      |
| Warmup Time          |       0.50 s       |       0.00 s       |       0.00 s       |
| TP: Compress         |       4.58 MB/s    |       0.25 MB/s    |       1.99 MB/s    |
| TP: Decompress       |      62.10 MB/s    |      62.01 MB/s    |      16.12 MB/s    |
| Core Eff (C)         |       1.00 MB/s    |       0.04 MB/s    |       0.36 MB/s    |
| Core Eff (D)         |      13.62 MB/s    |      11.04 MB/s    |       2.88 MB/s    |
| Wins: Space          |             2      |             0      |            38      |
| Wins: Encode         |            40      |             0      |             0      |
| Wins: Decode         |            27      |            13      |             0      |
| Wall-clock           |      14.95 s       |     260.42 s       |      35.92 s       |
| MSE (Quality)        |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| KODAK (24 imgs)      | ZPNG               | WebP(M6)           | JXL(E7)            |   
|----------------------|--------------------|--------------------|--------------------|   
| Original Size        |      14.68 MB      |      14.68 MB      |      14.68 MB      |   
| Compressed Size      |      11.11 MB      |      10.69 MB      |      10.32 MB      |   
| Savings (%)          |      24.34 %       |      27.22 %       |      29.69 %       |   
| BPP                  |        9.8738      |        9.4978      |        9.1759      |   
| Mean Ratio (%)       |      75.66 %       |      72.78 %       |      70.31 %       |   
| Median Ratio (%)     |      76.07 %       |      73.08 %       |      70.56 %       |   
| Ratio Range (%)      |  70.6-82.3 %       |  67.8-77.5 %       |  65.6-77.4 %       |   
| Avg Enc Time         |       61.5 ms      |     9460.6 ms      |      484.7 ms      |   
| Avg Dec Time         |       21.9 ms      |       12.1 ms      |       88.6 ms      |   
| Warmup Time          |       0.50 s       |       0.00 s       |       0.00 s       |   
| TP: Compress         |       6.55 MB/s    |       0.06 MB/s    |       1.24 MB/s    |   
| TP: Decompress       |      18.41 MB/s    |      49.29 MB/s    |       6.79 MB/s    |   
| Core Eff (C)         |       9.95 MB/s    |       0.06 MB/s    |       1.26 MB/s    |   
| Core Eff (D)         |      27.93 MB/s    |      49.34 MB/s    |       6.90 MB/s    |   
| Wins: Space          |             0      |             0      |            24      |   
| Wins: Encode         |            24      |             0      |             0      |   
| Wins: Decode         |             0      |            24      |             0      |   
| Wall-clock           |       3.04 s       |     227.24 s       |      14.00 s       |   
| MSE (Quality)        |    0.00000000      |    0.00000000      |    0.00000000      |  

### 7.1 Technical Observations
- **Encoding Dominance**: ZPNG v7.2 exhibits a **153x parallel encoding lead** over WebP (m=6) and is **7.8x faster** than JXL (Effort 7).
- **Industrial Stability**: Tested on the 60-image CLIC dataset, ZPNG maintains a stable **31.86% saving** with bit-perfect reconstruction (MSE = 0.000000).
- **Core Efficiency**: The architecture provides high per-core density, achieving **18.4 MB/s decompression** in pure-Python/JIT execution environments, making it a viable real-time contender for low-level image tasks.

---

---

## 8. Limitations & Roadmap

- **Bit Depth**: Currently limited to **8-bit** per channel.
- **Color Spaces**: Optimized for **RGB/RGBA**. No support for CMYK or YCbCr subsampling (G-sub transform is native to RGB).
- **Alpha Channel**: While RGBA is supported, the Alpha channel currently utilizes traditional Zstd compression (Level 1) rather than the high-performance rANS sharding engine used for RGB. Optimization for constant alpha (solidity detection) is a planned future improvement.
- **Threading**: Single-threaded core; parallelization achieved via image-level batching.

## 9. Dataset Sources

To verify the benchmarks or test the engine with standard datasets, you can download the images from the following official sources:

- **DIV2K Data Set - Train & Validation**: [ETH Zurich CVL](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Kodak Data Set**: [Kaggle - Kodak Dataset](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset/data)
- **Clic Data Set**: [Kaggle - CLIC Dataset](https://www.kaggle.com/datasets/mustafaalkhafaji95/clic-dataset?resource=download)
- **Tecnick Data Set**: [SourceForge - TestImages](https://sourceforge.net/projects/testimages/files/SAMPLING/)

---

## 10. Project Background

The ZPNG project is a lossless image compression framework developed by a non-technical lead with limited Python experience and no prior background in information theory. Over an extensive development cycle, the project utilized the agentic AI **Antigravity** to architect and execute complex technical components, including the **Four-Pillar Architecture**, Universal-42 Sharding, and a 4-way interleaved rANS entropy engine. This initiative serves as a technical proof-of-concept for AI-augmented engineering, demonstrating that autonomous agents can bridge the gap between high-level conceptual intent and low-level algorithmic optimization, resulting in a bit-perfect codec that maintains industrial-grade compression ratios against established standards like WebP and JPEG-XL.

## 11. Acknowledgments
This project stands on the shoulders of giants. Special thanks to Dr. Jarosław (Jarek) Duda for his groundbreaking work on Asymmetric Numeral Systems (ANS) and his unwavering commitment to keeping the rANS algorithm in the public domain.

His contribution allows independent developers like us to explore the frontiers of data compression without the constraints of patent barriers, enabling ZPNG-CSDE to achieve high-performance, bit-perfect results.

---

**Current Version:** v7.2.0 | **License:** GPL v3 | **MSE Target:** 0.000000
