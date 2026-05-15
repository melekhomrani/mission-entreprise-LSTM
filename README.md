# Job Market Demand Forecasting Using LSTM

## Project Overview

This project trains an **LSTM (Long Short-Term Memory)** neural network to forecast monthly job posting volume per occupational sector in the US job market. It uses the [Indeed Job Postings Index](https://www.kaggle.com/datasets/kimminh21/job-postings) dataset and predicts demand for the next 3–6 months.

**Business goal:** Help recruiters plan headcount and candidates time their job search by forecasting where hiring demand is heading — not just where it stands today.

---

## How to Run

### On Kaggle (recommended)

1. Go to [Kaggle](https://www.kaggle.com/) and create a new notebook
2. Click **"Add Data"** (right sidebar) → search **"Indeed Job Postings Index"** by Kim Minh → click **Add**
3. Upload `mission-entreprsie.ipynb` or copy the cells into the Kaggle notebook
4. Select **CPU** accelerator (no GPU needed — training takes ~2 minutes)
5. **Run All** cells top to bottom

### Locally

> Not recommended — requires TensorFlow installed. The notebook is designed for Kaggle's environment.

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
jupyter notebook mission-entreprsie.ipynb
```

You'll need to change `INPUT_DIR` in cell 5 to point to your local copy of the dataset.

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | Indeed Job Postings Index (Kaggle: `kimminh21/job-postings`) |
| Geography | United States |
| Granularity | Daily, seasonally adjusted |
| Baseline | 100 = Feb 1, 2020 (pre-pandemic) |
| Period | Feb 2020 → Apr 2026 (~75 months after aggregation) |
| Variable used | `total postings` (all active listings, not just new ones) |

### Target Sectors (8)

| Sector | Relevance |
|--------|-----------|
| Software Development | Core tech hiring signal |
| Data & Analytics | Data science / analytics demand |
| IT Systems & Solutions | Infrastructure & IT ops |
| Project Management | Cross-functional management roles |
| Marketing | Business/commercial demand |
| Management | General leadership roles |
| Banking & Finance | Financial sector demand |
| Human Resources | HR & people ops hiring |

---

## Notebook Structure

| Section | Cells | Description |
|---------|-------|-------------|
| 1. Setup & Imports | 1–3 | Libraries, random seed, TensorFlow config |
| 2. Dataset Loading | 4–8 | Load CSV, inspect schema, filter to `total postings` |
| 3. EDA | 9–13 | Daily trends, distributions, missing data heatmap, sector selection |
| 4. Monthly Aggregation | 14–19 | Daily → monthly resampling, correlation heatmap, seasonality analysis, summary stats |
| 5. Normalization & Sequences | 20–22 | MinMaxScaler [0,1], sin/cos month encoding, 12-month sliding windows, 80/20 temporal split |
| 6. LSTM Training | 23–25 | 2-layer LSTM (64→32), Dropout 0.2, EarlyStopping, train 8 models (one per sector) |
| 7. Evaluation | 26–29 | RMSE/MAE vs naive baseline, actual-vs-predicted plots, metrics bar chart |
| 8. Forecasting | 30–33 | 6-month autoregressive forecast, forecast visualization with uncertainty bands, summary table |
| 9. Interpretation | 34–35 | Limitations, improvements, model architecture summary |

---

## Model Architecture

```
Input (batch, 12, 3)     ← 12 months lookback, 3 features
       │
LSTM(64, return_sequences=True)
       │
Dropout(0.2)
       │
LSTM(32)
       │
Dropout(0.2)
       │
Dense(1)                 → predicted next month index
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size | 12 months | Captures one full seasonal cycle |
| Features | 3 (index, sin(month), cos(month)) | Index + cyclical time encoding |
| LSTM layers | 2 (64 → 32 units) | Funnel architecture; deeper would overfit on ~48 training samples |
| Dropout | 0.2 | Standard regularization for small datasets |
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, fast convergence |
| Loss | MSE | Standard regression loss |
| Early Stopping | patience=15, restore_best_weights | Prevents overfitting |
| Batch size | 8 | Small dataset needs small batches |
| Train/Val split | 80%/20% temporal | No shuffling — preserves time order |

---

## Evaluation

The LSTM is compared against a **naive baseline** (predict next month = current month):

$$\hat{y}_{t+1}^{naive} = y_t$$

Metrics computed on the 20% held-out validation set (last ~15 months):
- **RMSE** (Root Mean Squared Error) — penalizes large errors
- **MAE** (Mean Absolute Error) — average deviation in index points
- **Improvement %** — how much better LSTM is vs naive

---

## Outputs / Artifacts

The notebook generates these saved figures:

| File | Description |
|------|-------------|
| `daily_trends_all_sectors.png` | Raw daily index for all 8 sectors |
| `sector_distributions.png` | Box plots of index distribution per sector |
| `monthly_trends_per_sector.png` | Monthly aggregated trends (LSTM input) |
| `sector_correlation.png` | Inter-sector correlation heatmap |
| `seasonality_analysis.png` | Month-of-year seasonal patterns |
| `train_val_split.png` | Visual train/validation split per sector |
| `training_loss_curves.png` | Training & validation loss per sector |
| `actual_vs_predicted.png` | LSTM vs Actual vs Naive on validation set |
| `metrics_comparison.png` | RMSE/MAE bar chart: LSTM vs Naive |
| `future_forecast.png` | 6-month ahead forecast with uncertainty bands |

---

## Project Requirements Compliance

Cross-reference with the project description:

| Requirement | Status | Where in Notebook |
|-------------|--------|-------------------|
| **Business objective**: forecast posting volume per role type for next 3–6 months | **Met** | Section 8 — 6-month autoregressive forecast per sector |
| **Data source**: external multi-year jobs dataset (Kaggle) | **Met** | Indeed Job Postings Index (Feb 2020 – Apr 2026) |
| **Maintained as separate pipeline** | **Met** | Standalone notebook, independent from other project CSVs |
| **Train LSTM on monthly job posting volume per role category** | **Met** | Section 6 — one LSTM per sector, trained on monthly data |
| **Model learns temporal patterns: seasonality, growth trends, cyclical variations** | **Met** | Sin/cos month encoding for seasonality; 12-month window captures annual cycles; EDA confirms trend patterns |
| **Justify DL over ML**: LSTM hidden state learns long-range dependencies | **Met** | Section 9 — dedicated comparison table (LSTM vs Classical ML); markdown explanations throughout |
| **Evaluation: RMSE and MAE on held-out forecast window** | **Met** | Section 7 — RMSE/MAE for all 8 sectors + naive baseline comparison |
| **Forecast output optionally surfaced as KPI** | **Met** | Section 8c — forecast summary table with current value, 1/3/6 month predictions, change %, and trend direction |

---

## Known Limitations

1. **Small dataset** — 75 months is limited for deep learning; mitigated with dropout + early stopping
2. **High inter-sector correlation** (0.82–1.00) — sectors move together due to macro trends (COVID)
3. **COVID dominance** — the crash-recovery pattern overwhelms subtler seasonal signals
4. **Autoregressive error accumulation** — 6-month forecasts are less reliable than 1-month
5. **No exogenous variables** — model doesn't see GDP, interest rates, layoff news
6. **US-only** — results don't generalize to other geographies

---

## Tech Stack

- **Python 3.12** (Kaggle default)
- **TensorFlow / Keras** — LSTM model
- **pandas** — data manipulation, time-series resampling
- **NumPy** — numerical operations
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — MinMaxScaler, RMSE, MAE

---

## Team

ESPRIT — 3A INFO 3 — Mission Entreprise

---

## File Structure

```
mission-entreprise-LSTM/
├── mission-entreprsie.ipynb     # Main notebook (run on Kaggle)
├── project description.md       # Original project requirements
├── README.md                    # This file
└── *.png                        # Generated figures (after running)
```
