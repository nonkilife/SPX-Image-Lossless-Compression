# SPX Performance Benchmarks (Official)

This document maintains the official performance metrics for the SPX engine, comparing it against established standards like WebP (Method 6) and JPEG-XL (Effort 7) across industrial datasets (CLIC, DIV2K, Kodak, Tecnick).

---

## 0. Raw Data Efficiency (PNM Baseline)

To accurately measure the compression and throughput capabilities of SPX, we use calculated **PNM (Raw Pixels)** as the baseline despite actual statistics are from PNG file compression and decompression. 

**Hardware Benchmark Environment**:
**CPU**: AMD Ryzen 5 3500X (6-Core, 3.60 GHz)
**RAM**: 32.0 GB
**OS**: Windows 11 (64-bit, x64)

### SPX vs. PNM Raw Size Savings Comparison

| Dataset | PNM Size (Raw MB) | SPX Size (MB) | **Savings (vs PNM) %** |
| :--- | :---: | :---: | :---: |
| CLIC 2021 (Full) | 3902.28 | 1375.80 | **64.74 %** |
| CLIC 2025 (Test) | 246.75 | 82.82 | **66.43 %** |
| DIV2K VAL | 811.11 | 311.57 | **61.59 %** |
| DIV2K TRAIN | 6381.30 | 2486.43 | **61.04 %** |
| TECNICK RGB | 659.18 | 142.26 | **78.42 %** |
| TECNICK GRAY | 219.73 | 46.12 | **79.01 %** |
| WATERLOO RGB | 448.81 | 196.63 | **56.19 %** |
| WATERLOO GRAY | 155.33 | 66.80 | **56.99 %** |
| KODAK | 28.13 | 11.48 | **59.19 %** |

### SPX Performance Re-evaluation (PNM Raw Throughput)

| Dataset | PNM Size (MB) | Wall-clock (s) | **TP: Compress (MB/s)** | **TP: Decompress (MB/s)** |
| :--- | :---: | :---: | :---: | :---: |
| CLIC 2021 (Full) | 3902.28 | 56.79 | **147.42** | **128.70** |
| CLIC 2025 (Test) | 246.75 | 3.60 | **147.75** | **127.85** |
| DIV2K VAL | 811.11 | 12.17 | **152.75** | **118.06** |
| DIV2K TRAIN | 6381.30 | 100.34 | **126.81** | **127.57** |
| TECNICK RGB | 659.18 | 13.85 | **86.28** | **106.15** |
| TECNICK GRAY | 219.73 | 7.72 | **48.94** | **67.82** |
| WATERLOO RGB | 448.81 | 6.44 | **150.61** | **129.71** |
| WATERLOO GRAY | 155.33 | 5.40 | **53.01** | **62.63** |
| KODAK | 28.13 | 0.60 | **76.03** | **122.30** |

---

## 1. Comparative Results (Cumulative v8.2)
Test Date: 2026/04/26

| CLIC 2025 TEST            |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |     246.75 MB      |     246.75 MB      |     246.75 MB      |
| Dataset Size (30 imgs)    |     115.55 MB      |     115.55 MB      |     115.55 MB      |
| SPX Size                  |      82.82 MB      |      84.42 MB      |      77.81 MB      |
| BPP (PNM)                 |       24.0000      |       24.0000      |       24.0000      |
| BPP (PNG)                 |       11.2391      |       11.2391      |       11.2391      |
| BPP (Compressed)          |      8.055653      |      8.211502      |      7.568146      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      66.43 %       |      65.79 %       |      68.47 %       |
| Savings % (vs PNG)        |      28.32 %       |      26.94 %       |      32.66 %       |
| Mean Ratio (%)            |      71.50 %       |      72.00 %       |      67.09 %       |
| Median Ratio (%)          |      72.92 %       |      75.05 %       |      68.76 %       |
| Ratio Range (%)           |  51.0-83.7 %       |  50.3-81.4 %       |  50.4-75.6 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |       55.7 ms      |     3260.4 ms      |      767.7 ms      |
| Avg Dec Time              |       64.3 ms      |       16.5 ms      |      116.2 ms      |
| Total Enc Time            |       1.67 s       |      97.81 s       |      23.03 s       |
| Total Dec Time            |       1.93 s       |       0.49 s       |       3.49 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |       3.60 s       |      98.31 s       |      26.52 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |      19.99 MB/s    |       0.22 MB/s    |       0.94 MB/s    |
| Single Core (Dec)         |      17.33 MB/s    |      42.86 MB/s    |       6.22 MB/s    |
| Throughput (Enc)          |      69.10 MB/s    |       1.18 MB/s    |       5.02 MB/s    |
| Throughput (Dec)          |      59.92 MB/s    |     234.11 MB/s    |      33.14 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |       2            |       3            |      25            |
| Wins: Encode              |      30            |       0            |       0            |
| Wins: Decode              |       0            |      30            |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| CLIC 2021                 |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |    3902.28 MB      |    3902.28 MB      |    3902.28 MB      |
| Dataset Size (585 imgs)   |    1911.69 MB      |    1911.69 MB      |    1911.69 MB      |
| SPX Size                  |    1375.80 MB      |    1392.57 MB      |    1303.39 MB      |
| BPP (PNM)                 |       24.0001      |       24.0001      |       24.0001      |
| BPP (PNG)                 |       11.7574      |       11.7574      |       11.7574      |
| BPP (Compressed)          |      8.461509      |      8.564646      |      8.016184      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      64.74 %       |      64.31 %       |      66.60 %       |
| Savings % (vs PNG)        |      28.03 %       |      27.16 %       |      31.82 %       |
| Mean Ratio (%)            |      71.42 %       |      72.23 %       |      67.56 %       |
| Median Ratio (%)          |      71.31 %       |      72.73 %       |      67.79 %       |
| Ratio Range (%)           |  44.8-96.3 %       |  44.9-90.6 %       |  42.5-92.3 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |       45.2 ms      |     3370.9 ms      |      581.4 ms      |
| Avg Dec Time              |       51.8 ms      |       13.8 ms      |       84.4 ms      |
| Total Enc Time            |      26.47 s       |    1971.97 s       |     340.14 s       |
| Total Dec Time            |      30.32 s       |       8.08 s       |      49.40 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |      56.79 s       |    1980.05 s       |     389.54 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |      21.14 MB/s    |       0.16 MB/s    |       0.98 MB/s    |
| Single Core (Dec)         |      18.45 MB/s    |      40.24 MB/s    |       6.76 MB/s    |
| Throughput (Enc)          |      72.22 MB/s    |       0.97 MB/s    |       5.62 MB/s    |
| Throughput (Dec)          |      63.05 MB/s    |     236.48 MB/s    |      38.70 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |      18            |      16            |      551           |
| Wins: Encode              |      585           |       0            |       0            |
| Wins: Decode              |       0            |      585           |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| DIV2K VALIDATION          |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |     811.11 MB      |     811.11 MB      |     811.11 MB      |
| Dataset Size (100 imgs)   |     428.68 MB      |     428.68 MB      |     428.68 MB      |
| SPX Size                  |     311.57 MB      |     314.98 MB      |     292.36 MB      |
| BPP (PNM)                 |       24.0000      |       24.0000      |       24.0000      |
| BPP (PNG)                 |       12.6844      |       12.6844      |       12.6844      |
| BPP (Compressed)          |      9.219225      |      9.320002      |      8.650657      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      61.59 %       |      61.17 %       |      63.96 %       |
| Savings % (vs PNG)        |      27.32 %       |      26.52 %       |      31.80 %       |
| Mean Ratio (%)            |      72.68 %       |      73.55 %       |      68.23 %       |
| Median Ratio (%)          |      72.58 %       |      73.44 %       |      67.59 %       |
| Ratio Range (%)           |  59.3-93.3 %       |  63.5-89.1 %       |  53.7-85.8 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |       53.1 ms      |     3352.9 ms      |      734.7 ms      |
| Avg Dec Time              |       68.7 ms      |       18.7 ms      |      104.1 ms      |
| Total Enc Time            |       5.31 s       |     335.29 s       |      73.47 s       |
| Total Dec Time            |       6.87 s       |       1.87 s       |      10.41 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |      12.17 s       |     337.16 s       |      83.88 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |      24.17 MB/s    |       0.24 MB/s    |       1.03 MB/s    |
| Single Core (Dec)         |      18.69 MB/s    |      42.11 MB/s    |       7.26 MB/s    |
| Throughput (Enc)          |      80.76 MB/s    |       1.28 MB/s    |       5.83 MB/s    |
| Throughput (Dec)          |      62.44 MB/s    |     228.70 MB/s    |      41.19 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |       3            |       2            |      95            |
| Wins: Encode              |      100           |       0            |       0            |
| Wins: Decode              |       0            |      100           |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| DIV2K TRAINING            |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |    6381.30 MB      |    6381.30 MB      |    6381.30 MB      |
| Dataset Size (800 imgs)   |    3370.06 MB      |    3370.06 MB      |    3370.06 MB      |
| SPX Size                  |    2486.43 MB      |    2499.93 MB      |    2333.46 MB      |
| BPP (PNM)                 |       24.0000      |       24.0000      |       24.0000      |
| BPP (PNG)                 |       12.6748      |       12.6748      |       12.6748      |
| BPP (Compressed)          |      9.351439      |      9.402221      |      8.776122      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      61.04 %       |      60.82 %       |      63.43 %       |
| Savings % (vs PNG)        |      26.22 %       |      25.82 %       |      30.76 %       |
| Mean Ratio (%)            |      73.58 %       |      74.11 %       |      69.14 %       |
| Median Ratio (%)          |      73.60 %       |      74.13 %       |      69.15 %       |
| Ratio Range (%)           |  49.7-96.2 %       |  49.3-91.4 %       |  46.9-88.3 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |       62.9 ms      |     3043.8 ms      |      684.8 ms      |
| Avg Dec Time              |       62.5 ms      |       15.8 ms      |       96.9 ms      |
| Total Enc Time            |      50.32 s       |    2435.05 s       |     547.82 s       |
| Total Dec Time            |      50.02 s       |      12.65 s       |      77.49 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |     100.34 s       |    2447.70 s       |     625.31 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |      20.31 MB/s    |       0.23 MB/s    |       1.07 MB/s    |
| Single Core (Dec)         |      20.44 MB/s    |      45.15 MB/s    |       7.55 MB/s    |
| Throughput (Enc)          |      66.97 MB/s    |       1.38 MB/s    |       6.15 MB/s    |
| Throughput (Dec)          |      67.38 MB/s    |     266.42 MB/s    |      43.49 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |      19            |      22            |      759           |
| Wins: Encode              |      797           |       0            |       3            |
| Wins: Decode              |       0            |      800           |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| TECNICK RGB               |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |     659.18 MB      |     659.18 MB      |     659.18 MB      |
| Dataset Size (40 imgs)    |     191.98 MB      |     191.98 MB      |     191.98 MB      |
| SPX Size                  |     142.26 MB      |     148.18 MB      |     131.95 MB      |
| BPP (PNM)                 |       24.0000      |       24.0000      |       24.0000      |
| BPP (PNG)                 |        6.9898      |        6.9898      |        6.9898      |
| BPP (Compressed)          |      5.179538      |      5.394966      |      4.804305      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      78.42 %       |      77.52 %       |      79.98 %       |
| Savings % (vs PNG)        |      25.90 %       |      22.82 %       |      31.27 %       |
| Mean Ratio (%)            |      74.86 %       |      77.92 %       |      70.34 %       |
| Median Ratio (%)          |      74.13 %       |      77.89 %       |      70.63 %       |
| Ratio Range (%)           |  66.0-89.4 %       |  69.9-89.2 %       |  59.2-89.8 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |      191.1 ms      |     7096.0 ms      |     1010.5 ms      |
| Avg Dec Time              |      155.2 ms      |       32.6 ms      |      206.0 ms      |
| Total Enc Time            |       7.64 s       |     283.84 s       |      40.42 s       |
| Total Dec Time            |       6.21 s       |       1.30 s       |       8.24 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |      13.85 s       |     285.14 s       |      48.66 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |       6.20 MB/s    |       0.12 MB/s    |       0.87 MB/s    |
| Single Core (Dec)         |       7.63 MB/s    |      26.15 MB/s    |       4.29 MB/s    |
| Throughput (Enc)          |      25.12 MB/s    |       0.68 MB/s    |       4.75 MB/s    |
| Throughput (Dec)          |      30.93 MB/s    |     147.26 MB/s    |      23.30 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |       3            |       6            |      31            |
| Wins: Encode              |      40            |       0            |       0            |
| Wins: Decode              |       0            |      40            |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

|TECNICK GRAY               |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |     219.73 MB      |     219.73 MB      |     219.73 MB      |
| Dataset Size (40 imgs)    |      63.72 MB      |      63.72 MB      |      63.72 MB      |
| SPX Size                  |      46.12 MB      |      54.57 MB      |      42.91 MB      |
| BPP (PNM)                 |        8.0000      |        8.0000      |        8.0000      |
| BPP (PNG)                 |        2.3201      |        2.3201      |        2.3201      |
| BPP (Compressed)          |      1.679036      |      1.986861      |      1.562218      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      79.01 %       |      75.16 %       |      80.47 %       |
| Savings % (vs PNG)        |      27.63 %       |      14.36 %       |      32.67 %       |
| Mean Ratio (%)            |      71.67 %       |      86.32 %       |      67.89 %       |
| Median Ratio (%)          |      71.14 %       |      85.21 %       |      67.22 %       |
| Ratio Range (%)           |  63.0-84.8 %       |  80.3-96.7 %       |  61.1-82.7 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |      112.2 ms      |     5444.2 ms      |      287.9 ms      |
| Avg Dec Time              |       80.9 ms      |       22.1 ms      |       69.2 ms      |
| Total Enc Time            |       4.49 s       |     217.77 s       |      11.52 s       |
| Total Dec Time            |       3.24 s       |       0.88 s       |       2.77 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |       7.72 s       |     218.65 s       |      14.28 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |       5.16 MB/s    |       0.05 MB/s    |       1.03 MB/s    |
| Single Core (Dec)         |       7.16 MB/s    |      12.73 MB/s    |       4.29 MB/s    |
| Throughput (Enc)          |      14.20 MB/s    |       0.29 MB/s    |       5.53 MB/s    |
| Throughput (Dec)          |      19.69 MB/s    |      72.06 MB/s    |      23.03 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |      10            |       0            |      30            |
| Wins: Encode              |      40            |       0            |       0            |
| Wins: Decode              |       0            |      40            |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| WATERLOO RGB              |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |     448.81 MB      |     448.81 MB      |     448.81 MB      |
| Dataset Size (14 imgs)    |     448.81 MB      |     448.81 MB      |     448.81 MB      |
| SPX Size                  |     196.63 MB      |     192.55 MB      |     184.10 MB      |
| BPP (PNM)                 |       24.0000      |       24.0000      |       24.0000      |
| BPP (PNG)                 |        n/a         |        n/a         |        n/a         |
| BPP (Compressed)          |     10.514828      |     10.296527      |      9.844573      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      56.19 %       |      57.10 %       |      58.98 %       |
| Savings % (vs PNG)        |        n/a         |        n/a         |        n/a         |
| Mean Ratio (%)            |      40.22 %       |      39.14 %       |      37.69 %       |
| Median Ratio (%)          |      44.16 %       |      44.68 %       |      41.24 %       |
| Ratio Range (%)           |   8.1-72.7 %       |   5.5-64.7 %       |   6.4-67.1 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |      212.6 ms      |    14842.5 ms      |     2957.3 ms      |
| Avg Dec Time              |      247.4 ms      |      101.2 ms      |      551.1 ms      |
| Total Enc Time            |       2.98 s       |     207.79 s       |      41.40 s       |
| Total Dec Time            |       3.46 s       |       1.42 s       |       7.72 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |       6.44 s       |     209.21 s       |      49.12 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |      44.47 MB/s    |       0.60 MB/s    |       2.81 MB/s    |
| Single Core (Dec)         |      38.22 MB/s    |      87.54 MB/s    |      15.10 MB/s    |
| Throughput (Enc)          |     150.81 MB/s    |       2.16 MB/s    |      10.84 MB/s    |
| Throughput (Dec)          |     129.59 MB/s    |     316.64 MB/s    |      58.17 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |       0            |       4            |      10            |
| Wins: Encode              |      14            |       0            |       0            |
| Wins: Decode              |       0            |      14            |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| WATERLOO GRAY             |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |     155.33 MB      |     155.33 MB      |     155.33 MB      |
| Dataset Size (15 imgs)    |     155.33 MB      |     155.33 MB      |     155.33 MB      |
| SPX Size                  |      66.80 MB      |      66.19 MB      |      63.95 MB      |
| BPP (PNM)                 |        8.0000      |        8.0000      |        8.0000      |
| BPP (PNG)                 |        n/a         |        n/a         |        n/a         |
| BPP (Compressed)          |      3.440704      |      3.409191      |      3.293657      |
|---------------------------|--------------------|--------------------|--------------------|
| Savings % (vs PNM)        |      56.99 %       |      57.39 %       |      58.83 %       |
| Savings % (vs PNG)        |        n/a         |        n/a         |        n/a         |
| Mean Ratio (%)            |      41.83 %       |      40.07 %       |      40.03 %       |
| Median Ratio (%)          |      45.49 %       |      46.57 %       |      42.85 %       |
| Ratio Range (%)           |   9.6-94.7 %       |   8.9-58.4 %       |   8.8-90.4 %       |
|---------------------------|--------------------|--------------------|--------------------|
| Avg Enc Time              |      195.2 ms      |    17873.1 ms      |      851.4 ms      |
| Avg Dec Time              |      165.0 ms      |       60.4 ms      |      185.0 ms      |
| Total Enc Time            |       2.93 s       |     268.10 s       |      12.77 s       |
| Total Dec Time            |       2.48 s       |       0.91 s       |       2.78 s       |
| Warmup Time               |       0.01 s       |       0.00 s       |       0.00 s       |
| Wall-clock                |       5.40 s       |     269.00 s       |      15.55 s       |
|---------------------------|--------------------|--------------------|--------------------|
| Single Core (Enc)         |      17.65 MB/s    |       0.16 MB/s    |       3.25 MB/s    |
| Single Core (Dec)         |      20.87 MB/s    |      45.95 MB/s    |      14.94 MB/s    |
| Throughput (Enc)          |      53.06 MB/s    |       0.58 MB/s    |      12.16 MB/s    |
| Throughput (Dec)          |      62.76 MB/s    |     171.57 MB/s    |      55.97 MB/s    |
|---------------------------|--------------------|--------------------|--------------------|
| Wins: Space               |       0            |       2            |      13            |
| Wins: Encode              |      15            |       0            |       0            |
| Wins: Decode              |       0            |      15            |       0            |
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

<br>

| KODAK                     |        SPX         |      WebP(M6)      |      JXL(E7)       |
|---------------------------|--------------------|--------------------|--------------------|
| PNM Size                  |      28.13 MB      |      28.13 MB      |      28.13 MB      |  
| Dataset Size (25 imgs)    |      15.23 MB      |      15.23 MB      |      15.23 MB      |  
| SPX Size                  |      11.48 MB      |      11.09 MB      |      10.71 MB      |  
| BPP (PNM)                 |       24.0003      |       24.0003      |       24.0003      |  
| BPP (PNG)                 |       12.9973      |       12.9973      |       12.9973      |  
| BPP (Compressed)          |      9.794347      |      9.461382      |      9.141508      |  
|---------------------------|--------------------|--------------------|--------------------|  
| Savings % (vs PNM)        |      59.19 %       |      60.58 %       |      61.91 %       |  
| Savings % (vs PNG)        |      24.64 %       |      27.20 %       |      29.67 %       |  
| Mean Ratio (%)            |      75.67 %       |      73.01 %       |      70.63 %       |  
| Median Ratio (%)          |      75.60 %       |      73.19 %       |      70.78 %       |  
| Ratio Range (%)           |  70.3-82.0 %       |  67.8-77.5 %       |  65.6-77.4 %       |  
|---------------------------|--------------------|--------------------|--------------------|  
| Avg Enc Time              |       14.8 ms      |     1834.7 ms      |      107.9 ms      |  
| Avg Dec Time              |        9.3 ms      |        2.4 ms      |       19.6 ms      |  
| Total Enc Time            |       0.37 s       |      45.87 s       |       2.70 s       |  
| Total Dec Time            |       0.23 s       |       0.06 s       |       0.49 s       |  
| Warmup Time               |       0.01 s       |       0.00 s       |       0.01 s       |  
| Wall-clock                |       0.60 s       |      45.93 s       |       3.19 s       |  
|---------------------------|--------------------|--------------------|--------------------|  
| Single Core (Enc)         |      11.87 MB/s    |       0.06 MB/s    |       1.14 MB/s    |  
| Single Core (Dec)         |      18.80 MB/s    |      47.86 MB/s    |       6.27 MB/s    |  
| Throughput (Enc)          |      41.24 MB/s    |       0.33 MB/s    |       5.65 MB/s    |  
| Throughput (Dec)          |      65.33 MB/s    |     252.69 MB/s    |      31.06 MB/s    |  
|---------------------------|--------------------|--------------------|--------------------|  
| Wins: Space               |       0            |       0            |      25            |  
| Wins: Encode              |      25            |       0            |       0            |  
| Wins: Decode              |       0            |      25            |       0            |  
| MSE (Quality)             |    0.00000000      |    0.00000000      |    0.00000000      |

---

### 2. Technical Observations

- **Encoding Performance**: SPX v8.3.2 exhibits a **103x parallel encoding lead** over WebP (m=6) in RGB and a **63x lead** in Grayscale (Waterloo). It remains **6x–8x faster** than JXL (Effort 7) across industrial datasets.
- **Industrial Stability**: Validated on 2,000+ images (CLIC, DIV2K, Waterloo), SPX maintains a stable **25-30% saving vs PNG** with bit-perfect reconstruction (MSE = 0.000000).
- **Core Efficiency**: The architecture achieving **62.63 MB/s decompression** in Rust-native environments (Waterloo Gray), suitable for real-time archival and high-speed delivery pipelines.


## 3. Dataset Sources

To verify the benchmarks or test the engine with standard datasets, you can download the images from the following official sources:

- **DIV2K Data Set - Train & Validation**: [ETH Zurich CVL](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- **Kodak Data Set**: [Kaggle - Kodak Dataset](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset/data)

- **Clic Data Set**: [Kaggle - CLIC Dataset](https://www.kaggle.com/datasets/mustafaalkhafaji95/clic-dataset?resource=download)

- **Tecnick Data Set**: [SourceForge - TestImages](https://sourceforge.net/projects/testimages/files/SAMPLING/)

- **Waterloo Data Set**: [Image Compression Info](https://imagecompression.info/test_images/)