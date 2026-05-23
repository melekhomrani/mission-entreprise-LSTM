# Notebook Comparison Report

## `mission-entreprise-all-dataset.ipynb` vs `job-forecasting-global.ipynb`

---

## Overview

| Aspect | `mission-entreprise-all-dataset` | `job-forecasting-global` |
|--------|----------------------------------|--------------------------|
| **Author** | You | Teammate |
| **Forecast target** | US market (per sector) | Global average (per sector) |
| **Data aggregation** | Keep countries separate, pool sequences | Merge 6 countries → 1 global index |
| **Signal variable** | `total postings` | `new postings` |
| **Denoising** | None (raw monthly mean) | STL decomposition (trend + seasonal) |
| **Prediction target** | Raw normalized index (MinMax [0,1]) | First differences (Δ month-to-month) |
| **Models** | LSTM, GRU, BiLSTM (single seed each) | LSTM, GRU, BiLSTM (7-seed ensemble each) |
| **Feature count** | 3 (index, sin_month, cos_month) | 6+ sectors + 2 seasonal + N volatility + 1 COVID flag |
| **Training split** | 80/20 temporal | 70/15/15 (train/val/test) |
| **Forecast horizon** | 6 months (US) | 6 months (global) |

---

## Key Differences

### 1. What Is Being Predicted

| | Your notebook | Teammate's notebook |
|-|---------------|---------------------|
| **Target** | Next month's US index value per sector (univariate per sector) | Next month's Δ (change) in global index for ALL sectors at once (multivariate) |
| **Interpretation** | "Software Dev in the US will be at 82.5 next month" | "Software Dev globally will increase by +3.2 index points next month" |

**Implication:** Your notebook produces US-specific actionable forecasts. The teammate's produces a worldwide trend estimate.

---

### 2. Cross-Country Strategy

| | Your notebook | Teammate's notebook |
|-|---------------|---------------------|
| **Philosophy** | Cross-learning: train on all countries' sequences independently, validate on US | Averaging: collapse all countries into a single global series |
| **Benefit** | The model sees 6× more training sequences; preserves per-country dynamics | Simpler signal; cancels out country-specific noise |
| **Risk** | Countries with different structures may inject noise | Loses country-level detail; can't forecast a specific market |

---

### 3. Feature Engineering

| Feature | Your notebook | Teammate's notebook |
|---------|---------------|---------------------|
| Index value | MinMax scaled [0,1] | First-differenced, then MinMax [-1,1] |
| Seasonality | sin(month), cos(month) | sin(month), cos(month) |
| Volatility | — | 3-month rolling std per sector |
| COVID flag | — | Binary (Mar–Jun 2020) |
| Multi-sector | Independent model per sector | All sectors predicted jointly |
| STL denoising | No | Yes (removes residual noise) |

**Teammate's is more feature-rich.** STL + differencing + rolling volatility + COVID flag give the model more signal to work with.

---

### 4. Model Architecture & Training

| | Your notebook | Teammate's notebook |
|-|---------------|---------------------|
| LSTM layers | 2-layer (64→32) | Single-layer (48 units) |
| GRU layers | 2-layer (64→32) | Single-layer (48 units) |
| BiLSTM | Bidirectional(64) + LSTM(32) | Bidirectional(24) single-layer |
| Regularization | Dropout 0.2 | L2(5e-3) + dropout 0.2 + recurrent_dropout 0.2 |
| Loss function | MSE | Huber (robust to outliers) |
| LR schedule | Fixed (Adam 1e-3) | ReduceLROnPlateau (halves every 8 stalled epochs) |
| Ensemble | None (1 seed) | 7 seeds per architecture (21 models total) |
| Epochs | 100 max | 250 max |
| Bias correction | No | Yes (validates on val set, corrects systematic offset) |

**Teammate's is significantly more robust.** The 7-seed ensemble reduces variance from random initialization. L2 + recurrent dropout + Huber loss provides stronger regularization. ReduceLROnPlateau allows longer effective training.

---

### 5. Evaluation

| | Your notebook | Teammate's notebook |
|-|---------------|---------------------|
| Metrics | RMSE, MAE | RMSE, MAE, R² |
| Baselines | Compares LSTM vs GRU vs BiLSTM directly | Naive, Seasonal-Naive, + all 3 architectures |
| Test set | Validation only (last 20%) | Separate held-out test (last 15%), distinct from validation |
| Visualization | Actual vs 3 models per sector | Actual vs 3 models per sector + bar charts + heatmaps |

**Teammate's eval is more rigorous** because it has a proper 3-way split (train/val/test), includes baselines, and reports R².

---

### 6. Forecasting

| | Your notebook | Teammate's notebook |
|-|---------------|---------------------|
| Method | Autoregressive (best model per sector) | Autoregressive (best ensemble) |
| Uncertainty | Fixed ±3/step widening bands (heuristic) | Ensemble spread (10th–90th percentile across 7 seeds) |
| Bias correction | No | Yes (learned from validation residuals) |
| Output | Per-sector US forecast table | Per-sector global forecast + ranked bar chart |

**Teammate's uncertainty bands are data-driven** (from ensemble diversity), not manually set.

---

## Strengths & Weaknesses Summary

### Your notebook (`mission-entreprise-all-dataset`)

**Strengths:**
- Forecasts the **US specifically** (more relevant if the project is US-focused)
- Extensive EDA (box plots, heatmaps, seasonality, correlation matrix)
- Cross-learning preserves per-country patterns
- Cleaner, more readable code
- 8 sectors (uses ≥4 countries threshold)

**Weaknesses:**
- No denoising (STL or similar)
- Single seed (results can vary between runs)
- Only 3 features — no volatility, no COVID flag
- No separate test set (validation = test)
- No baseline comparison (naive, seasonal-naive)
- Heuristic uncertainty bands

---

### Teammate's notebook (`job-forecasting-global`)

**Strengths:**
- More sophisticated pipeline (STL → differencing → rolling features → ensemble)
- 7-seed ensemble with uncertainty quantification
- Proper train/val/test 3-way split
- Huber loss (outlier-robust), L2 regularization, LR scheduling
- Bias correction step
- Includes naive & seasonal-naive baselines
- Predicts all sectors jointly (captures cross-sector correlations)

**Weaknesses:**
- Forecasts **global average** (less actionable for a specific market)
- Only 6 sectors (strict intersection — drops Data & Analytics and Banking & Finance)
- Uses `new postings` (not `total postings`) — smaller signal base
- No per-country EDA — can't validate the "countries are similar" assumption
- Harder to explain in a presentation (more moving parts)
- Code is denser and less modular

---

## Recommendation

| If the mentor values... | Present... |
|------------------------|-----------|
| **Statistical rigor** (ensembles, proper eval, baselines) | Teammate's |
| **US-specific forecasts** (the project asks for a target market) | Yours |
| **Feature engineering depth** (STL, differencing, volatility) | Teammate's |
| **Clear EDA and data exploration** | Yours |
| **Presentation clarity** (easy to explain step-by-step) | Yours |
| **Production-readiness** (uncertainty bands, bias correction) | Teammate's |

### Best strategy: Present both as complementary approaches

1. **Lead with yours** for the EDA, cross-country analysis, and US-specific forecasting story
2. **Bring in the teammate's** for the advanced modeling (ensemble, STL, baselines)
3. Frame it as: "We explored two strategies — cross-learning (per-country pooling for US) and global aggregation (STL + ensemble) — and can compare their trade-offs"

This shows the mentor your team explored multiple approaches rather than just running one.

---

## Quick Wins to Strengthen Your Notebook (if time permits)

1. Add a naive baseline to the evaluation (trivial: `y_pred = last_value`)
2. Use 2–3 seeds instead of 1 (just loop the training with different seeds and average)
3. Add `rolling_std` as a 4th feature
4. Split into train/val/test instead of just train/val
