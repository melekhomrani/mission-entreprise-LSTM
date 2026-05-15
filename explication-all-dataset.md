# Multi-Country LSTM Notebook — Complete Deep Tutorial

> **Purpose of this file:** After reading this, you should be able to answer ANY question about the code — every variable, every function, every parameter, every design decision.

---

## Global Variables Reference

| Variable | Type | Value | Purpose |
|----------|------|-------|---------|
| `SEED` | int | 42 | Random seed for reproducibility across numpy, tensorflow |
| `INPUT_DIR` | str | `/kaggle/input/datasets/kimminh21/job-postings` | Root path where Kaggle mounts the dataset |
| `COUNTRIES` | dict | `{'US': '...csv', 'AU': '...csv', ...}` | Maps 6 country codes to their CSV filenames |
| `dfs_raw` | dict | `{country: DataFrame}` | Raw loaded data per country (before filtering) |
| `dfs_filtered` | dict | `{country: DataFrame}` | Filtered to 'total postings' only |
| `common_sectors` | set | ~26 sector names | Sectors present in ALL 6 countries |
| `DESIRED_SECTORS` | list | 8 sector names | Our wish list of tech/business sectors |
| `MIN_COUNTRIES` | int | 4 | Minimum number of countries a sector must appear in to be included |
| `TARGET_SECTORS` | list | 8 sector names | Final list: sectors from DESIRED that exist in ≥4 countries |
| `monthly_by_country` | dict | `{country: DataFrame}` | Monthly aggregated data per country (index=date, columns=sectors) |
| `df_monthly` | DataFrame | `monthly_by_country['US']` | US monthly data — primary market for EDA/validation/forecasting |
| `WINDOW_SIZE` | int | 12 | How many past months the LSTM looks at to make 1 prediction |
| `TRAIN_SPLIT` | float | 0.80 | 80% of data for training, 20% for validation |
| `results` | dict | `{sector: {...}}` | Stores model, history, scaler, predictions per sector |
| `forecasts` | dict | `{sector: {dates, values}}` | 6-month future predictions per sector |
| `FORECAST_MONTHS` | int | 6 | How many months ahead to forecast |

---

## Dataset Structure

All 6 CSV files have **identical headers**:
```
date,jobcountry,indeed_job_postings_index,variable,display_name
```

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `date` | date | `2020-02-01` | The day this measurement was recorded |
| `jobcountry` | str | `US`, `AU`, etc. | Country code |
| `indeed_job_postings_index` | float | `100.0`, `85.3` | Seasonally-adjusted index. 100 = Feb 1, 2020 baseline |
| `variable` | str | `total postings` or `new postings` | Type of metric. We use only `total postings` |
| `display_name` | str | `Software Development` | The occupational sector name |

**Dataset facts:**
- Date range: Feb 1, 2020 → Apr 24, 2026 (all countries same end date)
- US has 37 sectors, AU has 46, CA has 44, DE has 42, FR has 30, GB has 46
- Each row = 1 day × 1 sector × 1 variable type
- Index value > 100 = more postings than pre-pandemic; < 100 = fewer

**Sector availability for our 8 targets:**
| Sector | US | AU | CA | DE | FR | GB | Count |
|--------|----|----|----|----|----|----|-------|
| Software Development | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |
| Data & Analytics | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 5/6 |
| IT Systems & Solutions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |
| Project Management | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |
| Marketing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |
| Management | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |
| Banking & Finance | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | 5/6 |
| Human Resources | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/6 |

→ Data & Analytics is missing from France. Banking & Finance is missing from Canada.
→ Both still meet the ≥4 threshold, so they're included. The pooling function simply skips the missing country.

---

## Cell-by-Cell Explanation

---

### Cells 1–2 (Markdown) — Title & Library Table

Cell 1 states the objective: **forecast monthly job postings per sector using LSTM trained on 6 countries** (cross-learning approach). Lists the 9 notebook sections.

Cell 2 shows the library table (same libraries as US notebook) and notes that CPU is sufficient.

---

### Cell 3 — Imports & Configuration (lines 43–86)

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```
- `'3'` = ERROR level only. Suppresses INFO (`'0'`), WARNING (`'1'`), and ERROR (`'2'`) messages from TensorFlow's C++ backend about CUDA, GPU registration, etc.

```python
warnings.filterwarnings('ignore')
```
- Tells Python's warning system to silently discard all warnings (FutureWarning, DeprecationWarning, etc.).

```python
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 6),    # default width × height in inches
    'figure.dpi': 100,            # dots per inch (resolution)
    'axes.titlesize': 14,         # title font size
    'axes.labelsize': 12,         # axis label font size
    'lines.linewidth': 2,         # default line thickness
    'font.size': 11               # general text font size
})
```
- Sets global plot defaults so every figure looks professional without repeating settings.

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
- **Why 42?** Convention (Hitchhiker's Guide reference). Any fixed integer works.
- `np.random.seed()` — fixes NumPy's random number generator
- `tf.random.set_seed()` — fixes TensorFlow's (used for weight initialization, dropout masks)
- **Without this:** LSTM weights initialize randomly → different results each run → not reproducible

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
| Import | What it is |
|--------|-----------|
| `Sequential` | A model container — layers stacked linearly (input → layer1 → layer2 → output) |
| `LSTM` | Long Short-Term Memory layer — the recurrent layer that "remembers" past time steps |
| `Dense` | Fully-connected layer — multiplies input by weight matrix, adds bias |
| `Dropout` | Regularization — randomly zeros out a fraction of neurons during training |
| `EarlyStopping` | Callback that monitors a metric and stops training when it stops improving |
| `MinMaxScaler` | Transforms data to [0, 1] range: `x_scaled = (x - x_min) / (x_max - x_min)` |
| `mean_squared_error` | MSE = average of (actual - predicted)² |
| `mean_absolute_error` | MAE = average of |actual - predicted| |

---

### Cell 5 — List Dataset Files (lines 111–127)

```python
INPUT_DIR = '/kaggle/input/datasets/kimminh21/job-postings'
```
- On Kaggle, datasets are mounted at `/kaggle/input/`. The path after that matches the dataset slug.
- Locally (if testing), you'd change this to `'./datasets'`.

```python
for dirname, _, filenames in os.walk(INPUT_DIR):
```
- `os.walk()` recursively traverses directories. Returns `(current_dir, subdirectories, files)`.
- The `_` means we don't use the subdirectory list.
- This is a **debug/verification cell** — it just prints what files exist.

---

### Cell 6 — Load ALL 6 Countries (lines 130–156)

```python
COUNTRIES = {
    'US': 'job_postings_by_sector_US.csv',
    'AU': 'job_postings_by_sector_AU.csv',
    'CA': 'job_postings_by_sector_CA.csv',
    'DE': 'job_postings_by_sector_DE.csv',
    'FR': 'job_postings_by_sector_FR.csv',
    'GB': 'job_postings_by_sector_GB.csv',
}
```
- Dictionary: key = 2-letter country code, value = exact CSV filename.
- **Why these 6?** They're the only countries in the dataset that have **sector-level** data. Countries like EA (Euro Area), ES (Spain), IE (Ireland), IT (Italy), NL (Netherlands) only have aggregate data (no sector breakdown).

```python
dfs_raw = {}
for country, filename in COUNTRIES.items():
    filepath = os.path.join(INPUT_DIR, country, filename)
    if os.path.exists(filepath):
        df_c = pd.read_csv(filepath, parse_dates=['date'])
        df_c['country'] = country
        dfs_raw[country] = df_c
```
- `os.path.join()` — builds path with correct separators (works on Windows and Linux)
- `os.path.exists()` — safety check: if file is missing, print warning instead of crashing
- `parse_dates=['date']` — tells pandas to convert the `date` column from string `"2020-02-01"` to a `Timestamp` object (enables date arithmetic, resampling, etc.)
- `df_c['country'] = country` — adds a new column with the country code (useful if we concatenate later)
- `dfs_raw[country] = df_c` — stores the DataFrame in our dictionary

**Variable `dfs_raw`:** A dictionary where each key is a country code (`'US'`, `'AU'`, etc.) and each value is the full, unfiltered DataFrame for that country.

---

### Cell 7 — Inspect Sectors Per Country (lines 159–176)

```python
for country, df_c in dfs_raw.items():
    print(f"  {country}: {df_c['variable'].unique()}")
```
- `.unique()` returns an array of distinct values in the column.
- **Purpose:** Verify that all countries use `'total postings'` as the variable name. If any country used a different name (e.g., `'total'`), we'd need to handle it.
- **Result:** All 6 countries use exactly `'total postings'` and `'new postings'`.

```python
for country, df_c in dfs_raw.items():
    sectors = sorted(df_c['display_name'].unique())
```
- Lists every sector name per country so we can visually compare which sectors exist where.

---

### Cell 8 — Filter to 'total postings' & Find Common Sectors (lines 179–205)

```python
dfs_filtered = {}
for country, df_c in dfs_raw.items():
    if 'total postings' in df_c['variable'].values:
        df_f = df_c[df_c['variable'] == 'total postings'].copy()
    elif 'total' in df_c['variable'].values:
        df_f = df_c[df_c['variable'] == 'total'].copy()
```
- **Why filter?** Each CSV has 2 rows per day per sector: one for `'total postings'` (all active jobs) and one for `'new postings'` (posted in last 7 days). We only want total.
- `df_c['variable'] == 'total postings'` — creates a boolean mask (True/False for each row)
- `df_c[mask]` — keeps only rows where mask is True
- `.copy()` — creates an independent DataFrame (so editing `df_f` doesn't modify `df_c`)
- The `elif` is a fallback in case a country uses `'total'` instead of `'total postings'`

```python
sector_sets = [set(df_f['display_name'].unique()) for df_f in dfs_filtered.values()]
common_sectors = sector_sets[0]
for s in sector_sets[1:]:
    common_sectors = common_sectors.intersection(s)
```
- Creates a Python `set` of sector names for each country
- `intersection()` returns only elements present in **both** sets
- Iterating through all sets gives us sectors present in **all** countries
- **Result:** ~26 sectors exist in all 6 countries (out of 30-46 per country)

**Variable `dfs_filtered`:** Same structure as `dfs_raw` but only contains `'total postings'` rows.

---

### Cell 9 — Select Target Sectors (lines 208–258)

```python
from collections import Counter

DESIRED_SECTORS = [
    'Software Development', 'Data & Analytics', 'IT Systems & Solutions',
    'Project Management', 'Marketing', 'Management',
    'Banking & Finance', 'Human Resources',
]
```
- `Counter` — a dictionary subclass that counts occurrences of items
- `DESIRED_SECTORS` — our wish list of 8 sectors relevant to tech/business

```python
MIN_COUNTRIES = 4
all_sectors_flat = [s for df_f in dfs_filtered.values() for s in df_f['display_name'].unique()]
sector_counts = Counter(all_sectors_flat)
```
- `all_sectors_flat` — flattens all sector names from all countries into one list. If "Software Development" exists in 6 countries, it appears 6 times.
- `Counter(all_sectors_flat)` — counts: `{'Software Development': 6, 'Data & Analytics': 5, ...}`

```python
TARGET_SECTORS = [s for s in DESIRED_SECTORS if sector_counts.get(s, 0) >= MIN_COUNTRIES]
```
- List comprehension: keep a sector if it appears in at least 4 countries.
- `.get(s, 0)` — returns count for sector `s`, or 0 if not found (safe lookup)
- **Result:** All 8 sectors pass (each exists in ≥5 countries)

**Why ≥4 instead of all 6?**
- `Data & Analytics` is missing from France (5/6)
- `Banking & Finance` is missing from Canada (5/6)
- Requiring all 6 would drop these 2 sectors unnecessarily
- The `prepare_pooled_data()` function already handles missing sectors per country with `if sector not in df_m.columns: continue`

```python
in_all = [s for s in TARGET_SECTORS if s in common_sectors]
partial = [s for s in TARGET_SECTORS if s not in common_sectors]
```
- Separates sectors into "available everywhere" vs "partially available"
- Prints a note for partial sectors showing which countries are missing

---

### Cell 11 — Cross-Country Comparison Plot (lines 264–297)

```python
sample_sector = TARGET_SECTORS[0]  # 'Software Development'
colors = plt.cm.Set2(np.linspace(0, 1, len(dfs_filtered)))
```
- `plt.cm.Set2` — a colormap (palette) with 8 distinct colors
- `np.linspace(0, 1, 6)` — 6 evenly spaced values between 0 and 1
- This gives each country a unique color

```python
for (country, df_f), color in zip(dfs_filtered.items(), colors):
    mask = df_f['display_name'] == sample_sector
    sector_data = df_f[mask].sort_values('date')
    ax.plot(sector_data['date'], sector_data['indeed_job_postings_index'], label=country)
```
- Draws one line per country for the same sector
- **Purpose:** Validate that cross-learning makes sense — all countries should show a similar temporal shape (COVID crash → recovery → normalization)

```python
ax.axhline(y=100, ...)     # horizontal line at baseline
ax.axvspan('2020-03-01', '2020-06-01', ...)  # shaded COVID period
```
- `axhline` = horizontal reference line at index 100 (pre-pandemic level)
- `axvspan` = vertical shaded band highlighting March–June 2020

---

### Cell 12 — US Daily Trends (lines 300–328)

Same as the US-only notebook. Shows all TARGET_SECTORS for US on one chart with colors per sector.

---

### Cell 13 — Box Plots (lines 331–353)

```python
sector_order = (df_us_sectors.groupby('display_name')['indeed_job_postings_index']
                .median().sort_values(ascending=False).index)
sns.boxplot(data=df_us_sectors, x='display_name', y='indeed_job_postings_index',
            order=sector_order, palette='viridis')
```
- `.groupby().median().sort_values()` — orders sectors by their median index (highest first)
- `sns.boxplot()` — draws box-and-whisker plots showing distribution
- Box = 25th–75th percentile, line inside = median, whiskers = 1.5×IQR, dots = outliers

---

### Cell 14 — Data Coverage Heatmap (lines 356–377)

```python
coverage = []
for country, df_f in dfs_filtered.items():
    for sector in TARGET_SECTORS:
        n = (df_f['display_name'] == sector).sum()
        coverage.append({'Country': country, 'Sector': sector, 'Days': n})
df_coverage = pd.DataFrame(coverage)
df_cov_pivot = df_coverage.pivot(index='Sector', columns='Country', values='Days')
```
- Counts data points per country per sector
- `.pivot()` reshapes from long format to a grid (rows=sectors, columns=countries, values=day count)
- Displayed as a colored heatmap with numbers in each cell
- **Expected:** ~2275 days per sector per country (Feb 2020 → Apr 2026)
- **For missing sectors:** The cell shows 0 (e.g., Data & Analytics in FR = 0)

---

### Cell 16 — Monthly Aggregation for ALL Countries (lines 395–429)

```python
monthly_by_country = {}
for country, df_f in dfs_filtered.items():
    df_s = df_f[df_f['display_name'].isin(TARGET_SECTORS)].copy()
```
- `.isin(TARGET_SECTORS)` — keeps rows whose `display_name` is in our target list

```python
    df_pivot = df_s.pivot_table(
        index='date', columns='display_name',
        values='indeed_job_postings_index', aggfunc='first'
    )
```
- Reshapes from long format (one row per day per sector) to wide format (one row per day, one column per sector)
- `aggfunc='first'` — if there are duplicates (shouldn't happen), just take the first value

```python
    df_m = df_pivot.resample('ME').mean()
```
- `resample('ME')` — groups by month-end (`ME` = Month End frequency)
- `.mean()` — averages all daily values within each month into one monthly value
- Result: ~75 rows (months) × 8 columns (sectors)

```python
    df_m = df_m.ffill().bfill()
```
- `ffill()` = forward-fill: if a value is NaN, copy the previous month's value
- `bfill()` = backward-fill: handles NaN at the very beginning (copies from next month)
- Together they guarantee zero NaN values

```python
    available = [s for s in TARGET_SECTORS if s in df_m.columns]
    df_m = df_m[available]
    monthly_by_country[country] = df_m
```
- Only keeps columns that actually exist (handles France missing Data & Analytics)
- Stores the result

```python
df_monthly = monthly_by_country['US']
```
- **`df_monthly`** = the US monthly DataFrame. Used for all EDA, validation, and forecasting. Shape: `(75, 8)` — 75 months × 8 sectors.

**Variable `monthly_by_country`:** Dict of 6 DataFrames. Each has DatetimeIndex (monthly) and sector columns with averaged index values.

---

### Cell 17 — Monthly Trends Subplots, US (lines 432–458)

```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
axes = axes.flatten()
```
- Creates 2×4 = 8 subplots. `sharex=True` links all X-axes (same date range).
- `flatten()` converts 2D array `[[ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]]` → `[ax1,ax2,...,ax8]` for easy looping.

```python
for j in range(len(TARGET_SECTORS), len(axes)):
    axes[j].set_visible(False)
```
- If fewer than 8 sectors, hides unused subplot panels. Safety measure.

---

### Cell 18 — All Countries Monthly Overlay (lines 461–497)

```python
country_colors = {'US': 'steelblue', 'AU': 'coral', 'CA': 'green',
                  'DE': 'purple', 'FR': 'orange', 'GB': 'red'}
```
- Fixed color assignment per country for consistency across plots.

```python
for country, df_m in monthly_by_country.items():
    if sector in df_m.columns:
        ax.plot(df_m.index, df_m[sector], label=country, ...)
```
- For each subplot (=one sector), draws all available countries' monthly trends
- `if sector in df_m.columns` — skips countries that don't have this sector (e.g., France for Data & Analytics)
- **Key visualization:** Shows whether cross-country pooling makes sense. Similar shapes = yes.

---

### Cell 19 — Correlation Heatmap (lines 500–517)

```python
corr = df_monthly.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
```
- `df_monthly.corr()` — Pearson correlation between all pairs of sector columns
- `np.triu()` — creates upper-triangle mask (True above diagonal) to avoid showing redundant symmetric half
- Values range -1 to +1: >0.8 = strongly correlated, <0.5 = weakly correlated

---

### Cell 20 — Seasonality Analysis (lines 520–558)

```python
df_monthly_copy['month'] = df_monthly_copy.index.month
monthly_means = df_monthly_copy.groupby('month')[sector].agg(['mean', 'std'])
```
- `.index.month` extracts month number (1-12) from each date
- Groups all Januaries together, all Februaries together, etc., across years
- Computes mean and standard deviation per calendar month
- **Purpose:** If bars show consistent monthly patterns, the sin/cos encoding helps the LSTM

```python
ax.bar(..., yerr=monthly_means['std'], capsize=3)
```
- `yerr` = error bars showing ±1 standard deviation
- `capsize=3` = small horizontal caps on error bar ends

---

### Cell 21 — Summary Statistics (lines 561–586)

```python
total_months = sum(df_m.shape[0] for df_m in monthly_by_country.values())
```
- Sums rows across all countries: e.g., 6 × 75 = 450 total country-months

```python
summary = df_monthly.describe().T[['mean', 'std', 'min', 'max']]
```
- `.describe()` — computes count, mean, std, min, 25%, 50%, 75%, max for each column
- `.T` — transposes so sectors are rows (more readable)
- Select only the 4 statistics we care about

---

### Cell 23 — Normalization & Cross-Country Sequence Pooling (lines 612–710)

**This is the most critical cell. Three functions define the entire data pipeline.**

#### Constants:

```python
WINDOW_SIZE = 12   # 12 months = 1 full year of lookback
TRAIN_SPLIT = 0.80 # first 80% of data = training, last 20% = validation
```
- **Why 12?** Captures one full seasonal cycle. The model sees Jan→Dec before predicting the next month.
- **Why 80/20?** Standard split. With 75 months: train on months 1–60, validate on months 61–75.

#### Function: `create_sequences(data, window_size)`

```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])       # shape: (window_size, 3)
        y.append(data[i + window_size, 0])        # scalar: next month's index
    return np.array(X), np.array(y)
```

**Parameters:**
- `data` — numpy array, shape `(n_months, 3)`. Columns: [normalized_index, sin_month, cos_month]
- `window_size` — int, how many months per input sequence (12)

**Returns:**
- `X` — shape `(n_samples, window_size, 3)` — input sequences
- `y` — shape `(n_samples,)` — target values (the next month's normalized index)

**How it works — example with 75 months, window=12:**
- i=0: X[0] = months[0:12], y[0] = month[12] column 0
- i=1: X[1] = months[1:13], y[1] = month[13] column 0
- ...
- i=62: X[62] = months[62:74], y[62] = month[74] column 0
- Total: 63 sequences from 75 months (75 - 12 = 63)

**Why `data[i + window_size, 0]`?** The `0` selects the first feature column (the normalized index). We don't predict sin/cos — those are just auxiliary features.

#### Function: `prepare_single_country(df_m, sector, window_size, train_split)`

```python
def prepare_single_country(df_m, sector, window_size, train_split):
    series = df_m[sector].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series)
    
    months = df_m.index.month.values
    month_sin = np.sin(2 * np.pi * months / 12).reshape(-1, 1)
    month_cos = np.cos(2 * np.pi * months / 12).reshape(-1, 1)
    
    features = np.hstack([scaled, month_sin, month_cos])
    
    split_idx = int(len(features) * train_split)
    train_data = features[:split_idx]
    val_data = features[split_idx - window_size:]
    
    X_train, y_train = create_sequences(train_data, window_size)
    X_val, y_val = create_sequences(val_data, window_size)
    
    return X_train, y_train, X_val, y_val, scaler
```

**Parameters:**
- `df_m` — DataFrame with DatetimeIndex (monthly), one row per month
- `sector` — str, column name to extract (e.g., `'Software Development'`)
- `window_size` — int (12)
- `train_split` — float (0.80)

**Step-by-step:**

1. **Extract series:** `df_m[sector].values` → numpy array of raw index values (e.g., [100, 95, 80, ...])
   - `.reshape(-1, 1)` → column vector shape `(75, 1)`. Required by MinMaxScaler.

2. **Normalize:** `scaler.fit_transform(series)` → maps to [0, 1]
   - Formula: `scaled = (x - min) / (max - min)`
   - If min=50, max=200: value 100 → (100-50)/(200-50) = 0.333
   - **The scaler object stores min/max** so we can reverse this later with `scaler.inverse_transform()`

3. **Cyclical month encoding:**
   - `months = df_m.index.month.values` → array like [2, 3, 4, ..., 12, 1, 2, 3, 4]
   - `sin(2π × month / 12)` and `cos(2π × month / 12)` — maps months to a circle
   - **Why sin AND cos?** Sin alone is ambiguous (sin(March) = sin(September)). Together they give a unique 2D coordinate for each month.
   - **Why cyclical?** Month 12 (Dec) and month 1 (Jan) are adjacent in time but numerically far apart (12 vs 1). On the sin/cos circle, they're neighbors.

4. **Stack features:** `np.hstack([scaled, month_sin, month_cos])` → shape `(75, 3)`
   - Column 0: normalized index value
   - Column 1: sin(2π × month/12)
   - Column 2: cos(2π × month/12)

5. **Split:** `split_idx = int(75 * 0.80) = 60`
   - `train_data = features[0:60]` — months 1–60
   - `val_data = features[60 - 12:]` = `features[48:]` — months 49–75

   **Why the overlap (`split_idx - window_size`)?** The first validation sequence needs 12 months of context BEFORE the split point. Without overlap, the first validation input would start at month 60, but it needs months 49–60 as context to predict month 61.

6. **Create sequences:**
   - From `train_data` (60 months): 60 - 12 = 48 sequences
   - From `val_data` (27 months): 27 - 12 = 15 sequences

**Returns:**
- `X_train` shape: `(48, 12, 3)` — 48 training sequences
- `y_train` shape: `(48,)` — 48 target values
- `X_val` shape: `(15, 12, 3)` — 15 validation sequences
- `y_val` shape: `(15,)` — 15 target values
- `scaler` — the fitted MinMaxScaler (needed to convert predictions back)

#### Function: `prepare_pooled_data(monthly_by_country, sector, window_size, train_split)`

```python
def prepare_pooled_data(monthly_by_country, sector, window_size, train_split):
    all_X_train = []
    all_y_train = []
    us_scaler = None
    us_X_val = None
    us_y_val = None
    
    for country, df_m in monthly_by_country.items():
        if sector not in df_m.columns:
            continue  # skip countries that don't have this sector
        
        X_tr, y_tr, X_v, y_v, scaler = prepare_single_country(
            df_m, sector, window_size, train_split
        )
        
        if len(X_tr) > 0:
            all_X_train.append(X_tr)
            all_y_train.append(y_tr)
        
        if country == 'US':
            us_scaler = scaler
            us_X_val = X_v
            us_y_val = y_v
    
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    
    return X_train, y_train, us_X_val, us_y_val, us_scaler
```

**Parameters:**
- `monthly_by_country` — dict of 6 monthly DataFrames
- `sector` — which sector to process
- `window_size` — 12
- `train_split` — 0.80

**Logic:**
1. Loop through all countries
2. `if sector not in df_m.columns: continue` — gracefully skips countries that don't have this sector (e.g., France for Data & Analytics)
3. For each available country: prepare its sequences independently (each with its own scaler)
4. Append all training sequences to `all_X_train` list
5. Only save US validation data and US scaler (we forecast/evaluate on US only)
6. `np.concatenate(all_X_train, axis=0)` — stacks all countries' sequences vertically

**Returns:**
- `X_train` shape: `(~288, 12, 3)` — pooled from 6 countries (6 × 48 = 288 for sectors in all 6; 5 × 48 = 240 for sectors in 5)
- `y_train` shape: `(~288,)`
- `us_X_val` shape: `(15, 12, 3)` — US validation only
- `us_y_val` shape: `(15,)`
- `us_scaler` — US-specific scaler for inverse transform

**Key design decisions:**
- **Independent scalers per country:** Germany's index ranging 60–120 is different from US's 50–200. Each is normalized to [0,1] independently, making them comparable.
- **US-only validation:** We forecast the US market, so we validate against US ground truth.
- **Pooled training:** The LSTM sees the "crash→recovery" pattern from 6 different economies, learning the general shape rather than overfitting to US-specific noise.

---

### Cell 24 — Train/Val Split Visualization (lines 713–751)

```python
split_month = int(len(df_monthly) * TRAIN_SPLIT)  # = 60
```
- Shows blue (train, months 1–60) and coral (validation, months 61–75) for each sector
- The vertical dashed line marks the split point

---

### Cell 26 — Build & Train LSTM (lines 785–853)

#### `build_lstm_model(input_shape)`

```python
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

**Parameter:** `input_shape` = `(12, 3)` — 12 time steps, 3 features each.

**Layer-by-layer breakdown:**

| Layer | Parameters | Output Shape | What it does |
|-------|-----------|--------------|--------------|
| `LSTM(64, return_sequences=True)` | 17,408 | `(batch, 12, 64)` | Reads 12 months sequentially. At each step, updates a 64-dimensional hidden state. `return_sequences=True` outputs the state at EVERY time step (12 outputs). |
| `Dropout(0.2)` | 0 | `(batch, 12, 64)` | Randomly zeros 20% of values during training. Prevents co-adaptation of neurons. |
| `LSTM(32)` | 12,416 | `(batch, 32)` | Reads the 12 × 64 outputs from layer 1. Compresses into a single 32-dim vector (`return_sequences=False` = only output the LAST step). |
| `Dropout(0.2)` | 0 | `(batch, 32)` | Another 20% dropout. |
| `Dense(1)` | 33 | `(batch, 1)` | Linear transformation: 32 inputs → 1 output (the predicted next month's normalized index). |

**Total parameters:** ~29,857

**Why `return_sequences=True` for layer 1 but not layer 2?**
- Layer 1 → Layer 2: The second LSTM needs to see the full sequence of hidden states (12 steps of 64-dim vectors) to learn higher-level patterns.
- Layer 2 → Dense: The Dense layer only needs one vector (the final summary) to produce one prediction.

**Compile settings:**
- `optimizer='adam'` — Adam (Adaptive Moment Estimation): maintains per-parameter learning rates. Default lr=0.001. Combines benefits of AdaGrad (adapts to sparse gradients) and RMSprop (adapts to recent gradient magnitude).
- `loss='mse'` — Mean Squared Error: `(1/n) × Σ(ŷ - y)²`. Penalizes large errors quadratically.

#### Training loop:

```python
for sector in TARGET_SECTORS:
    X_train, y_train, X_val, y_val, scaler = prepare_pooled_data(
        monthly_by_country, sector, WINDOW_SIZE, TRAIN_SPLIT
    )
```
- One model per sector (8 total)
- Uses pooled data (all countries' training sequences)

```python
    early_stop = EarlyStopping(
        monitor='val_loss',    # watch validation MSE
        patience=15,           # stop if no improvement for 15 epochs
        restore_best_weights=True,  # roll back to the best epoch's weights
        verbose=0              # don't print each time it triggers
    )
```
- **`monitor='val_loss'`:** Tracks validation loss after each epoch
- **`patience=15`:** If val_loss doesn't decrease for 15 consecutive epochs, stop
- **`restore_best_weights=True`:** After stopping, the model weights are set back to the epoch that had the lowest val_loss (not the last epoch)
- **Why 15?** Enough slack for random fluctuations, but prevents training for 50+ useless epochs

```python
    history = model.fit(
        X_train, y_train,              # training inputs and targets
        validation_data=(X_val, y_val), # validation set (US only)
        epochs=100,                     # maximum epochs (usually stopped by EarlyStopping)
        batch_size=16,                  # process 16 sequences before updating weights
        callbacks=[early_stop],         # attach our early stopping
        verbose=0                       # no progress bars
    )
```
- **`epochs=100`:** Upper limit. EarlyStopping will usually stop at 30–60 epochs.
- **`batch_size=16`:** Number of sequences processed before one weight update. With ~288 sequences, that's ~18 gradient updates per epoch.
  - Why 16 (not 8 like US notebook)? More training data → can use larger batches → more stable gradients, faster training.
  - Too large (e.g., 128) → too few updates per epoch → slow convergence.
- **`callbacks=[early_stop]`:** List of functions called after each epoch.

```python
    results[sector] = {
        'model': model,         # the trained Keras model
        'history': history,     # training history (loss values per epoch)
        'scaler': scaler,       # US MinMaxScaler (for inverse_transform)
        'X_train': X_train,     # pooled training sequences
        'y_train': y_train,     # pooled training targets
        'X_val': X_val,         # US validation sequences
        'y_val': y_val,         # US validation targets
    }
```

---

### Cell 27 — Loss Curves (lines 856–887)

```python
h = results[sector]['history'].history  # dict: {'loss': [...], 'val_loss': [...]}
ax.plot(h['loss'], label='Train Loss')
ax.plot(h['val_loss'], label='Val Loss')
```
- `history.history` is a dict with one key per metric, each containing a list of values (one per epoch)
- **Good pattern:** Both curves decrease and converge → model is learning, not overfitting
- **Bad pattern:** Train goes down, val goes up → overfitting (EarlyStopping should have caught this)
- With pooled multi-country data: expect smoother curves (more data = less noise per batch)

---

### Cell 29 — Evaluation (lines 905–958)

```python
y_pred_scaled = r['model'].predict(r['X_val'], verbose=0).flatten()
```
- `.predict()` — feeds all validation sequences through the trained model
- Returns shape `(15, 1)` → `.flatten()` → `(15,)` — one prediction per validation sample
- `verbose=0` — no progress bar

```python
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = scaler.inverse_transform(r['y_val'].reshape(-1, 1)).flatten()
```
- Converts from normalized [0,1] back to real index values (e.g., 85.3)
- **Formula:** `original = scaled × (max - min) + min`
- Uses the **US scaler** (stored during `prepare_pooled_data`)

```python
y_naive = scaler.inverse_transform(
    r['X_val'][:, -1, 0].reshape(-1, 1)
).flatten()
```
- **Naive baseline:** "predict that next month = this month"
- `X_val[:, -1, 0]` — for each sequence, take the last time step (`-1`), first feature (`0` = normalized index)
- This is the most recent known value in each input window

```python
lstm_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
lstm_mae = mean_absolute_error(y_actual, y_pred)
```
- **RMSE:** $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ — in index points. Penalizes big errors more.
- **MAE:** $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ — in index points. More interpretable: "on average, we're off by X points."

```python
'RMSE_Improvement_%': round((1 - lstm_rmse/naive_rmse) * 100, 1)
```
- Positive = LSTM is better than naive by this percentage
- Negative = naive wins (LSTM is worse)
- Formula: `improvement = (1 - our_error/baseline_error) × 100`

---

### Cell 30 — Actual vs Predicted Plots (lines 961–998)

```python
split_month = int(len(df_monthly) * TRAIN_SPLIT)  # 60
val_dates = df_monthly.index[split_month:]          # dates from month 61 onward
dates = val_dates[:n]  # align with number of predictions (15)
```
- Gets the actual calendar dates for the validation period
- `val_dates[:n]` ensures we don't exceed the number of predictions

Three lines per subplot:
- **Black solid** = actual values (ground truth)
- **Blue dashed** = LSTM predictions
- **Coral dotted** = naive baseline

---

### Cell 31 — RMSE/MAE Bar Chart (lines 1001–1032)

```python
x = np.arange(len(sectors_list))  # [0, 1, 2, ..., 7]
width = 0.35                       # bar width
axes[0].bar(x - width/2, df_eval['LSTM_RMSE'], width, label='LSTM')
axes[0].bar(x + width/2, df_eval['Naive_RMSE'], width, label='Naive')
```
- Places LSTM bars slightly left and Naive bars slightly right of each tick mark
- Shorter bars = better (lower error)

---

### Cell 33 — Future Forecasts (lines 1050–1106)

```python
series = df_monthly[sector].values.reshape(-1, 1)
scaled_series = scaler.transform(series)  # note: transform, NOT fit_transform
```
- Uses US data and the **already-fitted** US scaler
- `transform()` applies the saved min/max without recalculating them

```python
last_window = features[-WINDOW_SIZE:].copy()  # last 12 months
```
- Starting point for autoregressive forecasting

```python
for step in range(FORECAST_MONTHS):    # 6 iterations
    input_seq = last_window.reshape(1, WINDOW_SIZE, 3)  # add batch dimension
    pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
```
- `reshape(1, 12, 3)` — model expects batch dimension: `(batch=1, steps=12, features=3)`
- `[0, 0]` — extract the single scalar prediction from shape `(1, 1)`

```python
    next_month = (last_month % 12) + 1  # wraps: 12→1, 1→2, ..., 11→12
    next_sin = np.sin(2 * np.pi * next_month / 12)
    next_cos = np.cos(2 * np.pi * next_month / 12)
```
- Calculate sin/cos for the **predicted** month (the model needs these as input for the next step)
- `% 12 + 1`: ensures month cycles correctly (December % 12 = 0, + 1 = January)

```python
    pred_original = scaler.inverse_transform([[pred_scaled]])[0, 0]
    future_preds.append(pred_original)
```
- Convert prediction back to real index scale for display

```python
    new_row = np.array([[pred_scaled, next_sin, next_cos]])
    last_window = np.vstack([last_window[1:], new_row])
    last_month = next_month
```
- **Slide the window:** drop oldest month `[1:]`, append prediction `new_row`
- Now window = 11 real months + 1 predicted month
- Next iteration uses this as input → **autoregressive** (predictions feed back as input)
- **Error accumulation:** Each prediction contains some error. When fed back as input, that error compounds. Month 6 is less reliable than month 1.

```python
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                              periods=FORECAST_MONTHS, freq='ME')
```
- Creates 6 future dates (e.g., May 2026, Jun 2026, ..., Oct 2026)
- `pd.DateOffset(months=1)` — adds exactly 1 month
- `freq='ME'` — month-end dates

---

### Cell 34 — Forecast Visualization (lines 1109–1155)

```python
recent = df_monthly[sector].iloc[-24:]  # last 24 months of actual data
```
- Shows 2 years of history for context (not all 75 months — too compressed)

```python
for j in range(len(f['values'])):
    uncertainty = (j + 1) * 3  # month 1: ±3, month 6: ±18
    ax.fill_between([f['dates'][j]], [f['values'][j] - uncertainty],
                    [f['values'][j] + uncertainty], color='red', alpha=0.1)
```
- **Uncertainty bands:** Linearly growing shading around forecasts
- `(j + 1) * 3` — rough heuristic: uncertainty grows by 3 index points per month
- Not a statistical confidence interval — just a visual indicator of increasing unreliability

---

### Cell 35 — Forecast Summary Table (lines 1158–1188)

```python
change_pct = ((end_val - current) / current) * 100
trend = "Growing ↑" if change_pct > 2 else ("Declining ↓" if change_pct < -2 else "Stable →")
```
- `current` = last known month's actual value
- `end_val` = month 6 forecast value
- Labels: >2% growth = "Growing", <-2% = "Declining", else "Stable"

---

### Cell 37 — Model Summary (lines 1238–1262)

```python
sample_model.summary()
```
- Keras built-in: prints table of layer names, output shapes, parameter counts
- Total params, trainable params, non-trainable params

Additional output shows multi-country specifics:
- Number of countries used
- Pooling strategy ("train on all, validate on US")
- Total training sequences per sector

---

## Key Differences from the US-Only Notebook

| Aspect | US Notebook | Multi-Country Notebook |
|--------|-------------|----------------------|
| Data loaded | 1 CSV (US only) | 6 CSVs (US, AU, CA, DE, FR, GB) |
| Sector selection | From US dataset only | Sectors in ≥4/6 countries (keeps all 8) |
| Monthly aggregation | US only → `df_monthly` | All 6 countries → `monthly_by_country` dict |
| Normalization | 1 scaler per sector | 1 scaler per sector **per country** (independent) |
| Training sequences | ~48 per sector (US only) | ~240–288 per sector (5–6 countries pooled) |
| Validation data | US last 20% | US last 20% (same) |
| Batch size | 8 | 16 (more data → bigger batch) |
| Architecture | 64→32 LSTM | 64→32 LSTM (same) |
| Forecasting | US model → US forecast | Multi-country model → US forecast |
| New EDA | — | Cross-country comparison, data coverage heatmap, multi-country monthly overlay |
| Missing sector handling | N/A (all in US) | `if sector not in df_m.columns: continue` |

---

## Frequently Asked Questions

**Q: Why not train a separate model per country?**
A: We only have ~48 sequences per country. That's very little for an LSTM to learn from. By pooling 6 countries, we get ~288 sequences — the model sees more examples of the crash→recovery pattern.

**Q: Why validate on US only?**
A: Our business goal is forecasting the US job market. Other countries' data helps train, but our evaluation metric must be on the target market.

**Q: Why use MinMaxScaler instead of StandardScaler?**
A: MinMaxScaler maps to [0,1] which is bounded — LSTM outputs pass through sigmoid/tanh activations that work best with bounded inputs. StandardScaler can produce large negative values.

**Q: What happens if a sector is missing in one country?**
A: The `prepare_pooled_data()` function has `if sector not in df_m.columns: continue` — it simply skips that country. The sector gets pooled from fewer countries (e.g., 5 instead of 6).

**Q: Why sin/cos encoding instead of one-hot for months?**
A: One-hot would add 12 features (one per month), and months would be treated as unrelated categories. Sin/cos uses only 2 features and preserves the circular nature (December is close to January).

**Q: What does `return_sequences=True` do?**
A: It makes the LSTM output its hidden state at every time step (12 outputs), not just the last one. The second LSTM layer needs the full sequence to read.

**Q: Why EarlyStopping patience=15 (not 5 or 50)?**
A: With noisy validation loss on small data, val_loss can fluctuate. Patience=5 might stop too early during a temporary fluctuation. Patience=50 wastes time on clearly overfitting models. 15 is a safe middle ground.

**Q: Why batch_size=16 instead of 8?**
A: We have 6x more training data (~288 vs ~48 sequences). Larger batches give more stable gradient estimates. With only 48 sequences, batch=16 would mean only 3 updates per epoch — too few. With 288, batch=16 means 18 updates per epoch — good.

**Q: What is "cross-learning" / "global modeling"?**
A: Training one model on data from multiple related time series (in our case, multiple countries). The model learns shared patterns (COVID shock shape) rather than one series' noise. Common technique in demand forecasting (e.g., Amazon trains one model across millions of products).

**Q: Why does the naive baseline often win?**
A: With only 75 months of data dominated by one massive event (COVID), the validation period (months 61–75) shows a smooth downward trend. Naive forecasting ("next month ≈ this month") works well on smooth trends. The LSTM would shine on data with more seasonality, regime changes, or longer history.

---

## TL;DR — The Full Pipeline in One Sentence Each

1. **Load** 6 countries' CSVs of daily job postings per sector
2. **Filter** to 'total postings' and find which sectors exist in ≥4 countries
3. **EDA**: cross-country comparison, US trends, distributions, coverage heatmap, correlation, seasonality
4. **Aggregate** daily → monthly for each country independently, store in `monthly_by_country`
5. **Normalize** each country independently to [0,1], add sin/cos month features, create 12-month sliding windows
6. **Pool** all countries' training sequences together (~288 sequences vs ~48 for US only)
7. **Train** a 2-layer LSTM (64→32, dropout 0.2, batch=16) on pooled data, validate on US only
8. **Evaluate**: compare LSTM predictions vs naive baseline on US data using RMSE/MAE
9. **Forecast** 6 months ahead for US market using the multi-country trained model
10. **Interpret** results: did cross-learning improve generalization over single-country?
