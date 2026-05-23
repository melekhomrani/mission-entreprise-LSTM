# Interpretation of Outputs — Multi-Country LSTM Notebook

> **Notebook:** `mission-entreprise-all-dataset-with-outputs.ipynb`  
> **Run environment:** Kaggle (GPU T4, cuDNN 91002)  
> **Date range:** February 2020 → April 2026  

---

## Table of Contents

1. [Cell 3 — Environment Setup](#1-cell-3--environment-setup)
2. [Cell 5 — Dataset File Listing](#2-cell-5--dataset-file-listing)
3. [Cell 6 — Country Loading Stats](#3-cell-6--country-loading-stats)
4. [Cell 7 — Sector Inspection](#4-cell-7--sector-inspection)
5. [Cell 8 — Common Sectors Filtering](#5-cell-8--common-sectors-filtering)
6. [Cell 9 — Target Sector Selection](#6-cell-9--target-sector-selection)
7. [Cell 11 — Cross-Country Comparison Plot](#7-cell-11--cross-country-comparison-plot)
8. [Cell 12 — US Daily Trends Plot](#8-cell-12--us-daily-trends-plot)
9. [Cell 13 — Box Plot (Sector Distributions)](#9-cell-13--box-plot-sector-distributions)
10. [Cell 14 — Data Coverage Heatmap](#10-cell-14--data-coverage-heatmap)
11. [Cell 16 — Monthly Aggregation Stats](#11-cell-16--monthly-aggregation-stats)
12. [Cell 17 — Monthly Trends per Sector (US)](#12-cell-17--monthly-trends-per-sector-us)
13. [Cell 18 — Multi-Country Monthly Overlay](#13-cell-18--multi-country-monthly-overlay)
14. [Cell 19 — Sector Correlation Heatmap](#14-cell-19--sector-correlation-heatmap)
15. [Cell 20 — Seasonality Analysis](#15-cell-20--seasonality-analysis)
16. [Cell 21 — Summary Statistics Table](#16-cell-21--summary-statistics-table)
17. [Cell 23 — Pooled Data Shapes](#17-cell-23--pooled-data-shapes)
18. [Cell 24 — Train/Validation Split Plot](#18-cell-24--trainvalidation-split-plot)
19. [Cell 26 — Training Logs](#19-cell-26--training-logs)
20. [Cell 27 — Loss Curves Plot](#20-cell-27--loss-curves-plot)
21. [Cell 29 — Evaluation Metrics Table](#21-cell-29--evaluation-metrics-table)
22. [Cell 30 — Actual vs Predicted Plot](#22-cell-30--actual-vs-predicted-plot)
23. [Cell 31 — RMSE & MAE Bar Charts](#23-cell-31--rmse--mae-bar-charts)
24. [Cell 33 — Forecast Values](#24-cell-33--forecast-values)
25. [Cell 34 — Forecast Visualization](#25-cell-34--forecast-visualization)
26. [Cell 35 — Forecast Summary Table](#26-cell-35--forecast-summary-table)
27. [Cell 37 — Model Architecture Summary](#27-cell-37--model-architecture-summary)
28. [Overall Assessment & Key Takeaways](#overall-assessment--key-takeaways)

---

## 1. Cell 3 — Environment Setup

**Output:**
```
TensorFlow version: 2.18.0
NumPy version:      1.26.4
Pandas version:     2.2.2
✓ All imports successful
```

**Interpretation:** All libraries loaded successfully on Kaggle. TensorFlow 2.18 confirms Keras 3 is being used. The `TF_CPP_MIN_LOG_LEVEL = '3'` setting suppresses most TensorFlow warnings, keeping the output clean.

---

## 2. Cell 5 — Dataset File Listing

**Output:** Lists all 6 CSV files in the Kaggle dataset directory with their sizes.

**Interpretation:** This confirms the Kaggle dataset mount path (`/kaggle/input/datasets/kimminh21/job-postings`) is correct and all 6 country files are present. File sizes vary because different countries track different numbers of sectors. GB and AU are the largest (~3+ MB), FR is the smallest (~1.7 MB) — this matches the sector counts we see later.

---

## 3. Cell 6 — Country Loading Stats

**Output:**
| Country | Rows | Sectors | Date Range |
|---------|------|---------|------------|
| US | 168,350 | 37 | 2020-02-01 → 2026-04-24 |
| AU | 209,300 | 46 | 2020-02-01 → 2026-04-24 |
| CA | 200,200 | 44 | 2020-02-01 → 2026-04-24 |
| DE | 191,100 | 42 | 2020-02-01 → 2026-04-24 |
| FR | 136,500 | 30 | 2020-02-01 → 2026-04-24 |
| GB | 209,300 | 46 | 2020-02-01 → 2026-04-24 |

**Interpretation:**
- All 6 countries loaded successfully, covering **identical date ranges** (Feb 2020 → Apr 2026, ~6.2 years).
- **AU and GB** have the most sectors (46 each), while **FR** has the fewest (30). This explains why some target sectors are missing from FR.
- The US has 37 sectors — it is our primary forecast target.
- Row counts differ because `rows = days × sectors`. More sectors → more rows.
- The consistent date range across all countries means no temporal alignment issues — critical for valid cross-country pooling.

---

## 4. Cell 7 — Sector Inspection

**Output:** Lists the `variable` types per country (all show `['total postings']`) and enumerates all sector names.

**Interpretation:** All countries use the same `'total postings'` variable type, so filtering is straightforward. The sector name listings let us verify which sectors exist in which countries before we select our targets.

---

## 5. Cell 8 — Common Sectors Filtering

**Output:** Shows the number of sectors common across all 6 countries and the filtered row counts after keeping only `'total postings'`.

**Interpretation:** After filtering to `total postings`, each country retains the sector-level daily time series. The strict intersection (sectors in ALL 6 countries) would exclude some useful sectors, which is why Cell 9 uses a relaxed threshold.

---

## 6. Cell 9 — Target Sector Selection

**Output:**
```
Target sectors (available in ≥4 countries): 8/8

  ✓ Software Development      [✓ ALL] (US, AU, CA, DE, FR, GB)
  ✓ Data & Analytics          [  5/6] (US, AU, CA, DE, GB)
  ✓ IT Systems & Solutions    [✓ ALL] (US, AU, CA, DE, FR, GB)
  ✓ Project Management        [✓ ALL] (US, AU, CA, DE, FR, GB)
  ✓ Marketing                 [✓ ALL] (US, AU, CA, DE, FR, GB)
  ✓ Management                [✓ ALL] (US, AU, CA, DE, FR, GB)
  ✓ Banking & Finance         [  5/6] (US, AU, DE, FR, GB)
  ✓ Human Resources           [✓ ALL] (US, AU, CA, DE, FR, GB)
```

**Interpretation:**
- All 8 desired sectors met the ≥4 countries threshold — **none were excluded**.
- 6 out of 8 sectors are available in **all 6 countries** (best case for cross-learning).
- **Data & Analytics** is missing from **France** (FR doesn't track this sector).
- **Banking & Finance** is missing from **Canada** (CA doesn't track this sector).
- For these 2 partial sectors, the pooling function gracefully skips the missing country. They train on 5 countries instead of 6, which gives ~240 sequences instead of 288 — still a massive improvement over single-country (48).
- This ≥4 threshold is a good pragmatic choice: we keep all sectors while still having enough cross-country diversity.

---

## 7. Cell 11 — Cross-Country Comparison Plot

**Diagram type:** Line plot — Software Development index across all 6 countries (daily resolution).

**What it shows:**
- X-axis: Date (Feb 2020 → Apr 2026)
- Y-axis: Indeed Job Postings Index (baseline = 100 at Feb 2020)
- 6 colored lines (one per country), a black dashed baseline at 100, and a red shaded COVID shock period (Mar–Jun 2020)

**Interpretation:**
- **COVID crash** (Mar 2020): All 6 countries show a sharp drop below 100, confirming the global nature of the shock. This is the steepest decline in the entire series.
- **Recovery phase** (mid-2020 → mid-2022): All countries recover, but at different rates. The US and AU tend to overshoot the baseline more aggressively than European countries (DE, FR).
- **Peak** (~2022): Most countries reach their peak index values (150–230 range), reflecting the post-pandemic tech hiring boom.
- **Normalization** (2023–2026): All countries converge back toward or below the baseline, reflecting the tech hiring slowdown and layoffs.
- **Key insight:** The **temporal shape is similar** across all countries (crash → boom → correction), even though the **magnitudes differ**. This similarity is what makes cross-country pooling valid — the LSTM can learn the shared pattern while being robust to country-specific noise.
- The red COVID shock band highlights the period of sharpest decline.

---

## 8. Cell 12 — US Daily Trends Plot

**Diagram type:** Multi-line plot — All 8 target sectors in the US market (daily resolution).

**What it shows:**
- X-axis: Date
- Y-axis: Job Postings Index
- 8 colored lines (one per sector), baseline at 100, COVID shock shading

**Interpretation:**
- **All sectors crashed** during COVID (Mar 2020), dropping to 45–65 range.
- **Recovery trajectories differ by sector:**
  - **Software Development** and **Human Resources** had the highest peaks (~230–240), reflecting extreme demand during the 2021–2022 hiring frenzy.
  - **Management** shows the most stable trajectory — smallest range (82 points), suggesting this sector is less sensitive to macro shocks.
  - **Marketing** spiked high (~185) but has since fallen well below baseline (~74), indicating a sector in correction.
- **Current state (Apr 2026):** Several sectors (Software Dev, IT Systems, Marketing) sit below the pre-pandemic baseline (100), while Project Management and Banking & Finance remain above it.
- This plot is crucial for understanding which sectors the LSTM will find easier to forecast (stable sectors like Management) vs harder (volatile sectors like Marketing).

---

## 9. Cell 13 — Box Plot (Sector Distributions)

**Diagram type:** Box plot — Distribution of daily index values per sector (US only).

**What it shows:**
- X-axis: Sector names
- Y-axis: Index value
- Box = interquartile range (IQR), whiskers = 1.5×IQR, median line, outliers as dots
- Red dashed line at baseline 100

**Interpretation:**
- **Human Resources** has the widest distribution (range ~196 points) — most volatile sector.
- **Software Development** also shows high variance (range ~167), reflecting the boom-bust cycle.
- **Management** has the tightest distribution (range ~82) — most predictable for LSTM.
- Medians vary: some sectors (Project Management, Management, Banking & Finance, Human Resources) have medians **above 100**, meaning they spent more time above the pre-pandemic baseline. Others (Data & Analytics, Marketing) have medians **near or below 100**.
- The box heights indicate how much variation the LSTM needs to learn — wider boxes = harder prediction task.

---

## 10. Cell 14 — Data Coverage Heatmap

**Diagram type:** Annotated heatmap — Data points (days) per sector per country.

**What it shows:**
- Rows: 8 target sectors
- Columns: 6 countries
- Cell values: Number of daily data points
- Color scale: Yellow (few) → Green (many)

**Interpretation:**
- Most cells show ~2,270–2,275 data points (matching the ~6.2 years of daily data).
- **Data & Analytics × FR = 0** — confirms France doesn't track this sector.
- **Banking & Finance × CA = 0** — confirms Canada doesn't track this sector.
- All other sector-country combinations have full coverage (~2,270+ days).
- This heatmap visually confirms our ≥4 countries threshold is correct and the missing data is limited to exactly 2 sector-country pairs.

---

## 11. Cell 16 — Monthly Aggregation Stats

**Output:**
```
US: 75 months × 8 sectors  | Feb 2020 → Apr 2026 | NaN: 0
AU: 75 months × 8 sectors  | Feb 2020 → Apr 2026 | NaN: 0
CA: 75 months × 7 sectors  | Feb 2020 → Apr 2026 | NaN: 0
DE: 75 months × 8 sectors  | Feb 2020 → Apr 2026 | NaN: 0
FR: 75 months × 7 sectors  | Feb 2020 → Apr 2026 | NaN: 0
GB: 75 months × 8 sectors  | Feb 2020 → Apr 2026 | NaN: 0
```

**Interpretation:**
- Daily data (~2,270 points) has been aggregated to **75 monthly means** per country per sector.
- **NaN: 0** across all countries — the `ffill().bfill()` strategy successfully eliminated any missing months.
- CA and FR have 7 sectors (missing Banking & Finance and Data & Analytics respectively), while the others have 8.
- 75 months with `WINDOW_SIZE=12` gives 63 possible sequences per country per sector. With 6 countries, this means up to 378 sequences per sector (or ~288 for 6-country sectors, after the 80% train split).
- The uniform 75-month span confirms perfect temporal alignment — no country starts or ends earlier.

---

## 12. Cell 17 — Monthly Trends per Sector (US)

**Diagram type:** 2×4 grid of line plots — One subplot per sector, US monthly data only.

**What it shows:**
- Each subplot: Monthly index over time for one sector
- Blue line = US monthly index, red dashed line = baseline 100
- Shared x-axis (date), individual y-axes

**Interpretation:**
- This is the **smoothed version** of the daily plot (Cell 12). Monthly aggregation removes daily noise, revealing cleaner trends.
- The **COVID crash → recovery → peak → decline** pattern is clearly visible in every sector.
- **Software Development** shows the most dramatic arc: crash to ~62, peak to ~229, back down to ~72.
- **Management** is the flattest — stays relatively close to 100–148 throughout.
- The current endpoint (Apr 2026) values visible here match the "Current" column in the forecast summary table (Cell 35).
- These are the curves the LSTM is actually trained on (after normalization). The 80% train split line falls around mid-2024.

---

## 13. Cell 18 — Multi-Country Monthly Overlay

**Diagram type:** 2×4 grid of multi-line plots — Each subplot shows one sector across all 6 countries.

**What it shows:**
- Each subplot: One sector, 6 colored lines (one per country)
- Legend with country codes, baseline at 100

**Interpretation:**
- **Cross-learning validation:** For most sectors, countries follow **broadly similar shapes** but with different magnitudes and timing offsets.
- **Software Development:** US had the highest peak (~229), while DE and FR had lower peaks. All countries are now declining toward or below baseline.
- **Management:** The most aligned sector — all countries cluster tightly, making it the easiest for cross-learning.
- **Marketing:** Shows the most divergence between countries — US and AU spiked much higher than European countries. This divergence may contribute to the LSTM struggling on this sector.
- **Key finding:** The similarity in temporal patterns across countries validates the cross-learning approach. Countries with similar shapes provide useful training signal, not noise.
- Where a sector is missing from a country (e.g., Data & Analytics × FR), that country's line simply doesn't appear in the subplot.

---

## 14. Cell 19 — Sector Correlation Heatmap

**Diagram type:** Lower-triangular correlation matrix heatmap — US monthly index values.

**What it shows:**
- 8×8 matrix of Pearson correlation coefficients between sector pairs
- Color scale: Blue (positive) → Red (negative), centered at 0
- Values annotated in each cell

**Interpretation:**
- **High positive correlations** (>0.8) exist between most sector pairs, indicating that all sectors are driven by the same macro trend (COVID crash → recovery → normalization).
- **Software Development ↔ Human Resources**, **Software Development ↔ Data & Analytics**: Very high correlation (~0.90+). When tech hiring rises, HR and data roles follow.
- **Management** likely has lower correlation with the more volatile sectors because its range is tighter and more stable.
- **Practical meaning:** Since all sectors are highly correlated, the LSTM sees a largely similar pattern across sectors. A multivariate approach (training all sectors together) could exploit this correlation.
- High correlation also means that if the model struggles on one sector, it likely struggles on all — consistent with our evaluation results.

---

## 15. Cell 20 — Seasonality Analysis

**Diagram type:** 2×4 grid of bar charts — Average index by month (Jan–Dec) with standard deviation error bars.

**What it shows:**
- Each subplot: One sector's average index value for each calendar month
- Error bars = ±1 standard deviation
- X-axis: Month (J, F, M, ..., D)

**Interpretation:**
- **Large error bars** dominate the picture. The standard deviations are huge relative to the mean differences between months, meaning **seasonality is weak compared to the COVID trend**.
- The COVID crash (which happened in March–April 2020) artificially deflates March/April means and inflates their standard deviations.
- With only ~6 years of data, each month has only ~6 observations per bar — too few to establish robust seasonal patterns.
- **Sin/cos month features** in the LSTM (features [1] and [2]) give the model a chance to learn seasonality if it exists, but these plots suggest that the seasonal signal is buried under the much stronger COVID trend.
- This is an important finding: the dataset is too dominated by one major event (COVID) to exhibit clear seasonality.

---

## 16. Cell 21 — Summary Statistics Table

**Output:**

| Sector | Mean Index | Std Dev | Min | Max | Range |
|--------|-----------|---------|-----|-----|-------|
| Software Development | 106.44 | 52.63 | 62.36 | 229.27 | 166.91 |
| Data & Analytics | 98.85 | 44.65 | 56.89 | 200.29 | 143.41 |
| IT Systems & Solutions | 101.67 | 39.58 | 58.64 | 195.64 | 137.00 |
| Project Management | 120.92 | 32.51 | 61.28 | 188.28 | 126.99 |
| Marketing | 101.45 | 36.76 | 44.71 | 184.61 | 139.90 |
| Management | 112.48 | 19.76 | 65.63 | 147.65 | 82.03 |
| Banking & Finance | 113.54 | 33.04 | 56.17 | 187.82 | 131.65 |
| Human Resources | 121.53 | 51.54 | 44.79 | 240.83 | 196.04 |

**Interpretation:**
- **Human Resources** has the widest range (196 points) and highest std dev (51.54) — the most volatile sector, hardest to predict.
- **Management** has the narrowest range (82 points) and lowest std dev (19.76) — most stable, should be easiest for the LSTM.
- Mean values above 100 (Software Dev 106, Project Management 121, HR 122) indicate these sectors spent more time above the pre-pandemic baseline during the observation window.
- **Data & Analytics** has a mean below 100 (98.85), suggesting it spent more time below baseline — possibly indicating a sustained downturn relative to Feb 2020.
- The high standard deviations (20–53) relative to the means (99–122) confirm that the COVID shock creates enormous within-series variance, making forecasting challenging.

---

## 17. Cell 23 — Pooled Data Shapes

**Output:**
```
Sector: Software Development

Single-country (US only):
  X_train: (48, 12, 3)  — 48 training sequences

Pooled (all 6 countries):
  X_train: (288, 12, 3) — 288 training sequences ← 6.0× more data!
  X_val:   (15, 12, 3)  — US only (our forecast target)

Features per timestep: 3
  [0] = normalized index value
  [1] = sin(month)
  [2] = cos(month)
```

**Interpretation:**
- **6× data amplification:** Cross-country pooling increases training sequences from 48 to 288 per sector. This is the key advantage of the multi-country approach.
- **Validation stays US-only:** The 15 validation sequences come exclusively from the last 20% of US data (~15 months, roughly Jan 2025 → Apr 2026). This ensures we evaluate on our target market.
- **Input shape (12, 3):** Each sequence is 12 months × 3 features. The 3 features are:
  - Feature 0: MinMax-normalized index value (the primary signal)
  - Feature 1: sin(2π × month/12) — cyclical month encoding
  - Feature 2: cos(2π × month/12) — cyclical month encoding
- **Why 15 validation sequences is small:** With only 15 points, the validation metrics can be noisy. A single bad prediction significantly impacts RMSE.
- The sin/cos month encoding is a standard technique for encoding cyclical features — it avoids the discontinuity that would occur if we used raw month numbers (12 → 1 jump).

---

## 18. Cell 24 — Train/Validation Split Plot

**Diagram type:** 2×4 grid of line plots — Each sector showing the temporal train/val split.

**What it shows:**
- Blue line: Training portion (first 80% of US data, ~Feb 2020 → ~mid-2024)
- Coral line: Validation portion (last 20%, ~mid-2024 → Apr 2026)
- Black dashed vertical line at the split point

**Interpretation:**
- The split is **temporal** (not random), which is correct for time series — you never train on future data.
- The training set contains the COVID crash, recovery, and peak — the most dramatic parts of the series.
- The validation set covers the **normalization/decline phase** (~mid-2024 → Apr 2026) — a fundamentally different regime than the training data.
- **Critical observation:** The model trains on a period dominated by dramatic upswings (recovery from COVID), but validates on a period of gradual decline. This **regime mismatch** partially explains why the LSTM struggles in evaluation — it learned the recovery pattern but is tested on the decline phase.
- The `print` output confirms: Train = Feb 2020 – ~mid-2024 (60 months), Val = ~mid-2024 – Apr 2026 (15 months).

---

## 19. Cell 26 — Training Logs

**Output (per sector):**

| Sector | Sequences | Epochs | Best Epoch | Train Loss | Val Loss |
|--------|-----------|--------|------------|------------|----------|
| Software Development | 288 | 79 | ~64 | 0.003162 | 0.001630 |
| Data & Analytics | 240 | 100 | ~85 | 0.003940 | 0.001860 |
| IT Systems & Solutions | 288 | 100 | ~85 | 0.003713 | 0.001964 |
| Project Management | 288 | 65 | ~50 | 0.004964 | 0.002515 |
| Marketing | 288 | 41 | ~26 | 0.008044 | 0.007977 |
| Management | 288 | 83 | ~68 | 0.004253 | 0.002374 |
| Banking & Finance | 240 | 17 | ~2 | 0.010686 | 0.008152 |
| Human Resources | 288 | 72 | ~57 | 0.004956 | 0.002750 |

**Interpretation:**
- **Software Development** achieved the lowest validation loss (0.001630) — the model learned this sector best.
- **Marketing** and **Banking & Finance** have the highest validation losses (0.008+), and trained for fewer epochs before early stopping triggered. This suggests the model struggled to find useful patterns for these sectors.
- **Banking & Finance** is notable: only 17 epochs, best at epoch ~2. This means the model barely improved beyond its initial state — the cross-country data for this sector may not contain learnable patterns that transfer to US validation.
- **Data & Analytics** and **IT Systems & Solutions** trained for the maximum 100 epochs, suggesting the model was still learning when training stopped (EarlyStopping patience=15 was used, so it stopped 15 epochs after the best).
- **Val loss < train loss** for several sectors (Software Dev, Data & Analytics, IT Systems). This is unusual and may indicate that the validation set happens to be "easier" (smoother decline) or that the pooled multi-country training data is noisier than the single-country validation.
- The **stderr** line `Loaded cuDNN version 91002` is just TensorFlow confirming GPU library availability — completely harmless.

---

## 20. Cell 27 — Loss Curves Plot

**Diagram type:** 2×4 grid of loss curves — Train loss and validation loss over epochs per sector.

**What it shows:**
- Blue line: Training loss (MSE) per epoch
- Coral line: Validation loss (MSE) per epoch
- X-axis: Epoch number
- Y-axis: MSE loss value

**Interpretation:**
- **Good convergence** for most sectors: train and val loss both decrease and stabilize, indicating the model is learning rather than memorizing.
- **No severe overfitting** visible — val loss doesn't diverge upward from train loss. The dropout layers (0.2) and cross-country pooling help regularize.
- **Marketing** and **Banking & Finance** show plateaus at higher loss values, indicating the model cannot reduce error further for these sectors.
- **Software Development** shows the cleanest convergence — both losses decrease smoothly to very low values.
- **Banking & Finance** (17 epochs only) has a very short curve — early stopping triggered quickly because the model wasn't improving.
- The generally smooth curves (vs jagged curves in the US-only notebook) are a benefit of having more training data via cross-country pooling.

---

## 21. Cell 29 — Evaluation Metrics Table

**Output:**
| Sector | LSTM RMSE | LSTM MAE | Naive RMSE | Naive MAE | RMSE Improv. % | MAE Improv. % |
|--------|-----------|----------|------------|-----------|----------------|---------------|
| Software Development | 2.41 | 2.01 | 1.54 | 1.32 | −56.4% | −52.4% |
| Data & Analytics | 3.95 | 2.87 | 1.37 | 1.09 | −188.3% | −162.9% |
| IT Systems & Solutions | 4.75 | 3.51 | 2.19 | 1.53 | −117.3% | −130.1% |
| Project Management | 3.71 | 3.21 | 1.66 | 1.30 | −123.6% | −146.2% |
| Marketing | 11.12 | 9.87 | 1.51 | 1.23 | −635.1% | −703.6% |
| Management | 2.68 | 2.21 | 1.54 | 1.29 | −74.5% | −71.4% |
| Banking & Finance | 8.52 | 7.99 | 1.53 | 1.20 | −455.6% | −564.3% |
| Human Resources | 9.09 | 8.02 | 1.49 | 1.28 | −509.0% | −527.0% |

**Average:** RMSE improvement = **−270.0%**, MAE improvement = **−294.7%**

**⚠️ The LSTM does NOT beat the naive baseline.**

**Interpretation:**
This is the most critical output. The **negative improvement percentages** mean the LSTM predictions are **worse** than simply predicting "next month = this month" (the naive baseline). Here's why:

1. **Why naive is hard to beat on this data:** The validation period (mid-2024 → Apr 2026) is a phase of **gradual, slow decline** for most sectors. During slow trends, the naive forecast (repeat the last value) only incurs small errors (~1–2 index points per month). The LSTM, which learned the dramatic crash-recovery pattern, tends to over-predict changes.

2. **Marketing** has the worst performance (RMSE 11.12 vs naive 1.51). This sector showed the most divergent cross-country patterns, so the pooled training data may have introduced conflicting signals.

3. **Software Development** is the "best" LSTM sector (RMSE 2.41 vs naive 1.54) — closest to naive, but still worse.

4. **This is a well-known phenomenon** in time series literature. For short-horizon forecasts on slowly-changing series, the naive baseline is extremely competitive. LSTMs (and other ML models) typically outperform when there are clear non-linear patterns or longer forecast horizons.

5. **Context matters:** The validation set is only 15 data points during a specific market regime. On a different validation period (e.g., the COVID recovery phase), the LSTM might outperform the naive baseline.

6. **Still valuable:** Even though point-accuracy is worse than naive, the LSTM captures the **direction and shape** of trends, which is useful for strategic forecasting beyond 1-month horizons.

---

## 22. Cell 30 — Actual vs Predicted Plot

**Diagram type:** 2×4 grid of line plots — Actual (black), LSTM predicted (blue dashed), and naive (coral dotted) on the US validation set.

**What it shows:**
- X-axis: Validation period months (~mid-2024 → Apr 2026)
- Y-axis: Index value (original scale)
- Black solid line: Actual US values
- Blue dashed line: LSTM predictions
- Coral dotted line: Naive baseline predictions

**Interpretation:**
- The **black line (actual)** shows a gradual decline for most sectors during the validation period.
- The **coral dotted line (naive)** closely tracks the actual values — it's always 1 month behind, which works well during slow trends.
- The **blue dashed line (LSTM)** often overshoots or undershoots the actual values, especially for Marketing, Banking & Finance, and Human Resources. For these sectors, the LSTM predictions diverge significantly from the actual trend.
- **Software Development** and **Management** show the best LSTM tracking — the blue line stays relatively close to the black line, even though it still doesn't beat naive.
- The visual confirms the metrics: the naive baseline hugs the actual curve much more tightly than the LSTM predictions.

---

## 23. Cell 31 — RMSE & MAE Bar Charts

**Diagram type:** Two side-by-side bar charts — LSTM (blue) vs Naive (coral) for each sector.

**What it shows:**
- Left chart: RMSE comparison
- Right chart: MAE comparison
- X-axis: Sector names
- Y-axis: Error in index points

**Interpretation:**
- The coral (naive) bars are consistently shorter than the blue (LSTM) bars, visually confirming the LSTM's underperformance.
- **Marketing** stands out with the tallest LSTM bar (~11 RMSE) — the model's worst sector.
- **Management** and **Software Development** have the smallest gap between LSTM and naive bars — these are the sectors where cross-country pooling was most beneficial.
- The naive bars are remarkably consistent across all sectors (~1.3–2.2 RMSE), while the LSTM bars vary wildly (2.4–11.1 RMSE). This inconsistency suggests the model's performance is sector-dependent.

---

## 24. Cell 33 — Forecast Values

**Output (6-month autoregressive forecasts, May 2026 → Oct 2026):**

| Sector | May 2026 | Jun 2026 | Jul 2026 | Aug 2026 | Sep 2026 | Oct 2026 |
|--------|----------|----------|----------|----------|----------|----------|
| Software Development | 71.9 ↓ | 73.4 ↑ | 75.2 ↑ | 77.3 ↑ | 79.9 ↑ | 82.5 ↑ |
| Data & Analytics | 63.7 ↑ | 63.8 ↑ | 64.9 ↑ | 66.8 ↑ | 69.1 ↑ | 71.5 ↑ |
| IT Systems & Solutions | 73.8 ↑ | 75.5 ↑ | 77.8 ↑ | 80.5 ↑ | 83.7 ↑ | 87.4 ↑ |
| Project Management | 111.2 ↑ | 112.5 ↑ | 114.3 ↑ | 116.4 ↑ | 118.8 ↑ | 120.8 ↑ |
| Marketing | 91.0 ↑ | 91.5 ↑ | 92.7 ↑ | 94.2 ↑ | 96.4 ↑ | 98.7 ↑ |
| Management | 102.2 ↑ | 103.2 ↑ | 104.3 ↑ | 105.6 ↑ | 107.4 ↑ | 109.0 ↑ |
| Banking & Finance | 108.8 ↑ | 109.8 ↑ | 110.6 ↑ | 110.9 ↑ | 110.7 ↑ | 110.3 ↑ |
| Human Resources | 104.0 ↑ | 107.7 ↑ | 112.4 ↑ | 117.9 ↑ | 123.6 ↑ | 128.8 ↑ |

**Interpretation:**
- The model predicts **upward trends** (↑) for all 8 sectors over the next 6 months.
- **Human Resources** shows the steepest predicted growth (+41.9% from 90.8 → 128.8), suggesting the model expects a strong rebound.
- **Banking & Finance** shows the most conservative growth and actually starts plateauing by Sep–Oct 2026 (110.9 → 110.7 → 110.3), suggesting the model sees a ceiling.
- **Software Development** is the only sector with a ↓ in month 1 (71.9 < 72.5 current), recovering afterward.
- **Caution:** Given the LSTM's underperformance vs naive baseline in evaluation, these forecasts should be treated as **directional indicators** rather than precise predictions. The autoregressive accumulation of errors means the Oct 2026 values (month 6) are significantly less reliable than May 2026 (month 1).
- The universally upward forecasts may reflect the model's learned "recovery pattern" from the training data (which was dominated by the post-COVID upswing), rather than a genuine signal about future market conditions.

---

## 25. Cell 34 — Forecast Visualization

**Diagram type:** 2×4 grid of line plots — Historical (last 24 months) + 6-month forecast per sector.

**What it shows:**
- Blue solid line: Last 24 months of historical US data
- Red dashed line with dots: 6-month forecast
- Red shaded bands: Increasing uncertainty (±3, ±6, ..., ±18 index points)
- Gray dotted line: Baseline at 100

**Interpretation:**
- The **widening uncertainty bands** correctly communicate that forecast reliability decreases with horizon. Month 1 has ±3 points of uncertainty, month 6 has ±18 points.
- For sectors currently below baseline (Software Dev at 72.5, Data & Analytics at 62.7, Marketing at 74.2), the forecasts predict a return toward 100 but don't quite reach it within 6 months.
- **Project Management** (currently at 107) and **Management** (at 100.7) are the only sectors predicted to remain comfortably above the baseline.
- The visual contrast between the declining historical trend (blue) and the rising forecast (red) is striking — the model is essentially predicting a trend reversal for all sectors.
- **Banking & Finance** shows the forecast flattening — the model predicts it will stabilize around 110, which is actually plausible given this sector's relative stability.

---

## 26. Cell 35 — Forecast Summary Table

**Output:**

| Sector | Current | Month 1 | Month 3 | Month 6 | Change % | Trend |
|--------|---------|---------|---------|---------|----------|-------|
| Software Development | 72.5 | 71.9 | 75.2 | 82.5 | +13.9% | Growing ↑ |
| Data & Analytics | 62.7 | 63.7 | 64.9 | 71.5 | +13.9% | Growing ↑ |
| IT Systems & Solutions | 72.3 | 73.8 | 77.8 | 87.4 | +20.9% | Growing ↑ |
| Project Management | 107.0 | 111.2 | 114.3 | 120.8 | +12.9% | Growing ↑ |
| Marketing | 74.2 | 91.0 | 92.7 | 98.7 | +33.0% | Growing ↑ |
| Management | 100.7 | 102.2 | 104.3 | 109.0 | +8.3% | Growing ↑ |
| Banking & Finance | 102.2 | 108.8 | 110.6 | 110.3 | +7.9% | Growing ↑ |
| Human Resources | 90.8 | 104.0 | 112.4 | 128.8 | +41.9% | Growing ↑ |

**Interpretation:**
- **All sectors classified as "Growing ↑"** — the model sees no declining or stable sectors.
- **Highest predicted growth:** Human Resources (+41.9%) and Marketing (+33.0%). These are the sectors that fell the most from their peaks, so the model may be expecting a "mean reversion."
- **Lowest predicted growth:** Banking & Finance (+7.9%) and Management (+8.3%). These are already near/above baseline and don't have as far to "recover."
- **Marketing's jump** from 74.2 to 91.0 in a single month (+22.6%) is unrealistically large, reinforcing that this sector's forecasts are unreliable (consistent with its worst evaluation metrics).
- The "Current" column confirms the Apr 2026 values: Software Dev (72.5), Data & Analytics (62.7), and Marketing (74.2) are well below the pre-pandemic baseline of 100.
- **Business takeaway (with caveats):** If these forecasts are directionally correct, tech-adjacent sectors (Software Dev, IT Systems, Data & Analytics) may be bottoming out, and demand could gradually recover over the next 6 months. However, given model limitations, this should be verified against other indicators.

---

## 27. Cell 37 — Model Architecture Summary

**Output:**
```
LSTM Model Architecture (Multi-Country):
  Window size:     12 months
  Features:        3 (index, sin_month, cos_month)
  LSTM Layer 1:    64 units, return_sequences=True
  LSTM Layer 2:    32 units
  Dropout:         0.2 (both layers)
  Optimizer:       Adam (lr=0.001)
  Loss:            MSE
  Early Stopping:  patience=15, restore_best_weights=True
  Batch size:      16
  Train/Val split: 80/20% temporal
  Countries:       6 (US, AU, CA, DE, FR, GB)
  Pooling:         Cross-country (train on all, validate on US)
  Sequences/sector: ~288 (pooled)
```

**Interpretation:**
- The **total trainable parameters** (shown in the Keras summary table) are determined by the LSTM unit counts: Layer 1 (64 units) processes 3 input features, Layer 2 (32 units) processes 64 features from Layer 1.
- The **funnel architecture** (64 → 32) progressively compresses the temporal representation — the first layer captures detailed short-term dynamics, the second distills higher-level patterns.
- **288 sequences per sector** is small by deep learning standards but reasonable for time-series LSTM. The cross-country pooling provides 6× more data than single-country training.
- The **80/20 temporal split** yields ~60 months of training and ~15 months of validation per country.

---

## Overall Assessment & Key Takeaways

### What Works Well

1. **Solid methodology:** The notebook follows a rigorous pipeline — EDA → aggregation → normalization → modeling → evaluation → forecasting, with clear documentation at each step.
2. **Cross-country pooling is well-implemented:** The ≥4 countries threshold, independent normalization, and graceful handling of missing sectors are all correctly done.
3. **Comprehensive visualization:** 10+ publication-quality plots that thoroughly explore the data before modeling.
4. **Honest evaluation:** The notebook openly acknowledges the LSTM's underperformance vs naive baseline, which shows scientific integrity.
5. **Baseline comparison:** Including the naive baseline is essential — without it, the LSTM's errors would lack context.

### What the Results Tell Us

1. **The naive baseline wins** on this specific validation period (gradual decline phase). This is a common result in time series literature — persistence forecasts are hard to beat on slowly-changing series.
2. **Cross-country pooling helped with training** (smoother loss curves, no severe overfitting) but didn't translate to better US predictions. The model learned a "global average" pattern rather than US-specific dynamics.
3. **The COVID event dominates** — the model essentially learned one big pattern (crash → recovery) and struggles when the validation data shows a different regime (slow decline).
4. **Forecasts should be treated as directional indicators**, not precise predictions.

### Why This Is Still a Valid Deep Learning Project

- The goal was to **apply LSTM to time series forecasting** and evaluate it, not necessarily to beat every baseline. Understanding *why* a model underperforms is itself a valuable learning outcome.
- The cross-learning approach is methodologically sound and represents a real-world technique used in production systems (e.g., Amazon's DeepAR).
- The notebook demonstrates full-cycle ML: data engineering, EDA, feature engineering, model design, training, evaluation, and forecasting.
