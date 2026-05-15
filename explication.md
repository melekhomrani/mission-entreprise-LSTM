Here's your complete baby-step tutorial of the entire notebook:

---

## Complete Line-by-Line Tutorial

---

### Cell 1 & 2 (Markdown) — Title & Library Table

Just text. Cell 1 explains the project goal: **predict how many job postings each sector will have in the future, using an LSTM neural network.** Cell 2 lists the libraries we'll use and why.

---

### Cell 3 — Imports & Configuration (lines 41–84)

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```
**What:** Tells TensorFlow to shut up — suppresses noisy warning messages about CUDA/GPU that clutter the output on Kaggle (which runs on CPU).

```python
import warnings
warnings.filterwarnings('ignore')
```
**What:** Hides all Python warning messages (like deprecation notices). Keeps output clean.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
**What:** Loads the 4 core libraries:
- `numpy` = math on arrays (numbers crunching)
- `pandas` = tables/spreadsheets in Python (load CSV, filter, group)
- `matplotlib` = draw charts
- `seaborn` = draw prettier charts (built on top of matplotlib)

```python
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({...})
```
**What:** Sets visual defaults for all plots: figure size, font size, line thickness. So every plot looks nice without repeating these settings.

```python
SEED = 42
np.random.seed(SEED)
```
**What:** Fixes the random number generator so results are **reproducible**. Without this, LSTM training would give different results every run because weights start randomly.

```python
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
**What:**
- `tensorflow` / `keras` = the deep learning library. We import:
  - `Sequential` = a model that stacks layers one after another (like a sandwich)
  - `LSTM` = the special recurrent layer that remembers past time steps
  - `Dense` = a regular fully-connected layer (outputs the final prediction)
  - `Dropout` = randomly turns off some neurons during training to prevent overfitting
  - `EarlyStopping` = stops training automatically when the model stops improving
- `MinMaxScaler` = squishes values to the range [0, 1] (normalization)
- `mean_squared_error`, `mean_absolute_error` = formulas to measure prediction accuracy

---

### Cell 5 — List Dataset Files (lines 108–124)

```python
INPUT_DIR = '/kaggle/input/datasets/kimminh21/job-postings'
```
**What:** The folder where Kaggle stores the dataset after you click "Add Data."

```python
for dirname, _, filenames in os.walk(INPUT_DIR):
    ...
```
**What:** Walks through every folder and subfolder, printing file names and sizes. This is a **debug step** — if the path is wrong, you'll see nothing here and know to fix it.

---

### Cell 6 — Load the CSV (lines 127–141)

```python
SECTOR_FILE = os.path.join(INPUT_DIR, 'US', 'job_postings_by_sector_US.csv')
df_raw = pd.read_csv(SECTOR_FILE, parse_dates=['date'])
```
**What:**
- Builds the file path: `…/US/job_postings_by_sector_US.csv`
- `pd.read_csv()` reads the CSV file into a DataFrame (a table). `parse_dates=['date']` automatically converts the `date` column from text to a proper date object so we can do date math later.
- `df_raw` = the raw, unfiltered dataset. Each row = one day, one sector, one index value.

---

### Cell 7 — Dataset Statistics (lines 144–158)

```python
df_raw.dtypes              # What type is each column? (int, float, string, date)
df_raw.isnull().sum()       # How many missing values per column?
df_raw['display_name'].nunique()  # How many unique sector names?
df_raw['variable'].unique()       # What types of postings? ('total postings', 'new postings')
df_raw['indeed_job_postings_index'].describe()  # min, max, mean, std of the index
```
**What:** Basic health check of the data. We need to know:
- Are there missing values? (hopefully no)
- What sectors exist? (so we can pick the right names)
- What does `variable` column contain? (we'll filter on it next)

---

### Cell 8 — Filter to 'total postings' (lines 161–175)

```python
df = df_raw[df_raw['variable'] == 'total postings'].copy()
df = df.drop(columns=['variable', 'jobcountry'])
```
**What:**
- The dataset has 2 types of data: `'total postings'` (all active jobs) and `'new postings'` (jobs posted in the last 7 days). We only want **total** because it shows the full demand picture.
- `df_raw[df_raw['variable'] == 'total postings']` = keep only rows where the `variable` column equals `'total postings'`. Like a SQL `WHERE` clause.
- `.copy()` = make an independent copy (so editing `df` won't accidentally change `df_raw`)
- `.drop(columns=...)` = remove columns we don't need anymore

---

### Cell 10 — Select 8 Target Sectors (lines 193–228)

```python
TARGET_SECTORS = [
    'Software Development', 'Data & Analytics', 'IT Systems & Solutions',
    'Project Management', 'Marketing', 'Management',
    'Banking & Finance', 'Human Resources',
]
```
**What:** We pick 8 sectors relevant to tech/business. The dataset has 20+ sectors but we don't need all of them.

```python
df_sectors = df[df['display_name'].isin(TARGET_SECTORS)].copy()
```
**What:** Keep only rows whose `display_name` is in our list. Like `WHERE display_name IN ('Software Development', 'Data & Analytics', ...)`.

```python
found = df_sectors['display_name'].unique()
missing = set(TARGET_SECTORS) - set(found)
```
**What:** Safety check — if we typed a sector name wrong, it won't be found and this tells us.

---

### Cell 11 — Daily Trend Plot (lines 231–260)

```python
for sector in TARGET_SECTORS:
    mask = df_sectors['display_name'] == sector
    sector_data = df_sectors[mask].sort_values('date')
    ax.plot(sector_data['date'], sector_data['indeed_job_postings_index'], label=sector)
```
**What:** For each sector, filter rows belonging to that sector, sort by date, and draw a line on the chart (date on X-axis, index value on Y-axis).

```python
ax.axhline(y=100, ...)    # horizontal line at 100 (pre-pandemic baseline)
ax.axvspan(...)            # shaded rectangle highlighting March–June 2020 (COVID crash)
```
**What:** Visual markers to help interpret the chart. The 100 line = "same as before COVID". Below 100 = fewer job postings than pre-pandemic.

---

### Cell 12 — Box Plots (lines 263–288)

```python
sns.boxplot(data=df_sectors, x='display_name', y='indeed_job_postings_index', ...)
```
**What:** A box plot shows the distribution (min, 25th percentile, median, 75th percentile, max) of index values for each sector. Sectors with tall boxes = more volatile (big swings in job postings).

---

### Cell 13 — Missing Data Heatmap (lines 291–324)

```python
df_pivot_check = df_sectors.pivot_table(index='date', columns='display_name',
                                         values='indeed_job_postings_index')
```
**What:** Reshapes the data from "long format" (one row per day per sector) to "wide format" (one row per day, one column per sector). Think of it as a spreadsheet where rows = dates and columns = sector names.

```python
sns.heatmap(df_pivot_check.isnull().T, ...)
```
**What:** Shows a grid where red = missing data. If it's all white, we have no missing values.

---

### Cell 15 — Monthly Aggregation (lines 346–373)

This is a **critical step** — converting daily data to monthly.

```python
df_pivot = df_sectors.pivot_table(index='date', columns='display_name',
                                   values='indeed_job_postings_index', aggfunc='first')
```
**What:** Same pivot as before — reshape to wide format (rows=dates, columns=sectors).

```python
df_monthly = df_pivot.resample('ME').mean()
```
**What:** `resample('ME')` groups dates by month-end. `.mean()` takes the average of all daily values within each month. So if January has 31 daily values of [101, 102, 99, ...], we get one value: the average.

**Why monthly?** Daily data is too noisy (weekends, holidays cause random dips). Monthly smooths this out. Also, 75 months is a manageable size for LSTM training.

```python
df_monthly = df_monthly.ffill().bfill()
```
**What:** Forward-fill then backward-fill any remaining gaps. `ffill()` = if a value is missing, copy the previous month's value. `bfill()` = same but going backward (for the very first month if it's missing).

---

### Cell 16 — Monthly Trend Subplots (lines 376–400)

```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
axes = axes.flatten()
```
**What:** Creates a 2×4 grid of subplots (8 total — one per sector). `flatten()` turns the 2D array of axes into a 1D list so we can loop with `axes[0], axes[1], ...`.

```python
for i, sector in enumerate(TARGET_SECTORS):
    ax = axes[i]
    ax.plot(df_monthly.index, df_monthly[sector], ...)
```
**What:** For each sector, plot its monthly time series in the corresponding subplot.

---

### Cell 17 — Correlation Heatmap (lines 403–426)

```python
corr = df_monthly.corr()
```
**What:** Calculates the **Pearson correlation** between every pair of sectors. Correlation ranges from -1 to +1:
- +1 = sectors move in perfect lockstep
- 0 = no relationship
- -1 = they move in opposite directions

```python
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, ...)
```
**What:** `mask` hides the upper triangle (because correlation is symmetric — A↔B = B↔A). `annot=True` prints the number in each cell.

**Key finding:** Most sectors have correlation >0.8, meaning they all crashed and recovered together (driven by COVID).

---

### Cell 18 — Seasonality Analysis (lines 429–467)

```python
df_monthly_copy['month'] = df_monthly_copy.index.month
monthly_means = df_monthly_copy.groupby('month')[sector].agg(['mean', 'std'])
```
**What:** Groups all data by month-of-year (all Januaries together, all Februaries together, etc.) and computes the average and standard deviation. This shows if there's a recurring seasonal pattern (e.g., hiring dips every December).

```python
ax.bar(monthly_means.index, monthly_means['mean'], yerr=monthly_means['std'], ...)
```
**What:** Bar chart with error bars. If all bars are similar height, there's no strong seasonality.

---

### Cell 19 — Summary Statistics (lines 470–490)

```python
summary = df_monthly.describe().T[['mean', 'std', 'min', 'max']]
```
**What:** `describe()` computes statistics for every column. `.T` transposes (rows↔columns) so sectors are rows. We pick only the columns we care about. This table can go directly into your report.

---

### Cell 21 — Normalization & Sequence Creation (lines 516–590)

**This is the most important cell.** It prepares data for the LSTM.

#### `create_sequences()` function:

```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])     # input: 12 months of data
        y.append(data[i + window_size, 0])       # output: next month's index
    return np.array(X), np.array(y)
```
**What this does — imagine you have months [Jan, Feb, Mar, Apr, ..., Dec, Jan, Feb]:**
- Window 1: Input = [Jan→Dec], Output = Jan (next year)
- Window 2: Input = [Feb→Jan], Output = Feb
- Window 3: Input = [Mar→Feb], Output = Mar
- ...and so on. Each "window" slides forward by 1 month.

The LSTM learns: "given these 12 months, what comes next?"

`y.append(data[i + window_size, 0])` — the `0` means we only predict the first feature (the index value), not the sin/cos features.

#### `prepare_sector_data()` function:

```python
series = df_monthly[sector].values.reshape(-1, 1)
```
**What:** Extracts one sector's column as a numpy array. `reshape(-1, 1)` makes it a column vector (required by the scaler).

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(series)
```
**What:** Normalizes values to [0, 1]. If the sector's index ranges from 50 to 200:
- 50 → 0.0
- 125 → 0.5
- 200 → 1.0

**Why?** LSTMs work better when inputs are small numbers. Without normalization, a sector with values around 200 would dominate training over a sector with values around 80.

The `scaler` object remembers the transformation so we can **reverse it later** (convert predictions back to real index values).

```python
months = df_monthly.index.month.values           # [2, 3, 4, ..., 12, 1, 2, ...]
month_sin = np.sin(2 * np.pi * months / 12)      # cyclical encoding
month_cos = np.cos(2 * np.pi * months / 12)
```
**What:** Encodes the month as a point on a circle using sin/cos. Why not just use 1-12? Because month 12 (December) and month 1 (January) are adjacent in time but far apart numerically (12 vs 1). Sin/cos encoding places them close together on a circle:
- Jan: sin=0.5, cos=0.87
- Dec: sin=-0.5, cos=0.87
- They're neighbors on the circle!

```python
features = np.hstack([scaled, month_sin, month_cos])
```
**What:** Combines 3 columns side by side: `[normalized_index, sin(month), cos(month)]`. Each month is now described by 3 numbers.

```python
split_idx = int(len(features) * train_split)     # 80% mark = month 60
train_data = features[:split_idx]                 # months 0-59
val_data = features[split_idx - window_size:]     # months 48-74 (overlap!)
```
**What:** Splits data into training (first 80%) and validation (last 20%). The `- window_size` overlap is critical: the first validation sequence needs 12 months of **preceding** context as input, so we include those 12 months from the training set.

```python
X_train, y_train = create_sequences(train_data, window_size)
X_val, y_val = create_sequences(val_data, window_size)
```
**What:** Creates the sliding window sequences for both training and validation sets.

**Resulting shapes:**
- `X_train`: (48, 12, 3) = 48 samples, each is 12 months, each month has 3 features
- `y_train`: (48,) = 48 target values (the "answer" for each sample)
- `X_val`: (15, 12, 3) = 15 validation samples
- `y_val`: (15,) = 15 validation targets

---

### Cell 22 — Train/Val Split Visualization (lines 593–622)

```python
ax.plot(df_monthly.index[:split_month], df_monthly[sector].iloc[:split_month],
        color='steelblue', label='Train')
ax.plot(df_monthly.index[split_month:], df_monthly[sector].iloc[split_month:],
        color='coral', label='Validation')
```
**What:** Draws each sector's time series with the training portion in blue and validation portion in red. The vertical dashed line shows where the split happens.

---

### Cell 24 — Build & Train the LSTM (lines 655–724)

#### `build_lstm_model()`:

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
```
**What — layer by layer:**

1. **LSTM(64, return_sequences=True)**: First LSTM layer with 64 "memory units." It reads the 12-month sequence step by step, updating its hidden state at each step. `return_sequences=True` = output the hidden state at **every** time step (not just the last one), because the next LSTM layer needs the full sequence.

2. **Dropout(0.2)**: Randomly sets 20% of the neurons to zero during each training step. This forces the network to not rely on any single neuron too much — prevents overfitting (memorizing the training data instead of learning patterns).

3. **LSTM(32)**: Second LSTM layer with 32 units. Reads the output of the first layer. `return_sequences=False` (default) = only output the hidden state at the **last** time step. This compresses the 12-step sequence into one vector.

4. **Dropout(0.2)**: Another dropout for regularization.

5. **Dense(1)**: A single output neuron. Takes the 32-dimensional vector from the second LSTM and outputs one number: the predicted next month's index value (normalized).

```python
model.compile(optimizer='adam', loss='mse')
```
**What:**
- `optimizer='adam'` = the algorithm that adjusts weights during training. Adam adapts the learning rate for each weight — works well out of the box.
- `loss='mse'` = Mean Squared Error. The model tries to minimize the average of (prediction - actual)². Large errors are penalized more than small ones.

#### Training loop:

```python
for sector in TARGET_SECTORS:
```
**What:** We train a **separate model for each sector**. 8 sectors = 8 independent models.

```python
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
```
**What:** Watches the validation loss after each epoch. If it doesn't improve for 15 epochs in a row, stop training and go back to the weights that gave the best validation loss. This prevents overfitting — the model stops before it starts memorizing noise.

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=8,
    callbacks=[early_stop],
    verbose=0
)
```
**What:**
- `epochs=100` = maximum 100 passes through the training data (but EarlyStopping will likely stop before)
- `batch_size=8` = update weights after every 8 samples (not all at once, not one at a time — a balance between speed and stability)
- `validation_data` = after each epoch, evaluate on the validation set to track overfitting
- `verbose=0` = don't print progress bars (we print our own summary)

```python
results[sector] = { 'model': model, 'history': history, 'scaler': scaler, ... }
```
**What:** Stores everything in a dictionary so we can access any sector's model, training history, scaler, and data later.

---

### Cell 25 — Loss Curves (lines 727–750)

```python
h = results[sector]['history'].history
ax.plot(h['loss'], label='Train Loss')
ax.plot(h['val_loss'], label='Val Loss')
```
**What:** Plots how the loss (error) changed over training epochs. Ideally:
- Both curves go down = model is learning
- They converge together = no overfitting
- If val_loss goes up while train_loss goes down = overfitting (model memorizes training data)

---

### Cells 26–27 — Hyperparameter Sensitivity Analysis (lines 753–843)

**Cell 26 (Markdown):** Explains why we test multiple configurations — to demonstrate the chosen architecture wasn't arbitrary.

**Cell 27 (Code):**

```python
HP_CONFIGS = {
    'A: 32→16, W=12': {'units': (32, 16), 'window': 12},
    'B: 48→24, W=12': {'units': (48, 24), 'window': 12},
    'C: 64→32, W=12 (ours)': {'units': (64, 32), 'window': 12},
    'D: 64→32, W=6':  {'units': (64, 32), 'window': 6},
    'E: 64→32, W=9':  {'units': (64, 32), 'window': 9},
}
```
**What:** Defines 5 model configurations varying LSTM layer sizes (32→16, 48→24, 64→32) and window sizes (6, 9, 12 months). Each is trained on a single sector (Software Development) and evaluated by validation loss.

**Why?** Shows professors we tested alternatives. If all configs give similar validation loss, it confirms the data is the bottleneck, not the architecture.

The cell produces a bar chart comparing validation MSE across configs. Our chosen config (C: 64→32, W=12) is highlighted in red.

---

### Cell 29 — Evaluation: LSTM vs Naive Baseline (lines 861–919)

```python
y_pred_scaled = r['model'].predict(r['X_val'], verbose=0).flatten()
```
**What:** Feeds all validation sequences into the trained model, gets predictions. `.flatten()` converts from shape (15, 1) to (15,).

```python
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = scaler.inverse_transform(r['y_val'].reshape(-1, 1)).flatten()
```
**What:** Converts predictions and actual values back from [0,1] range to real index values (e.g., 85.3, 102.7). This is why we saved the `scaler` — it remembers the original min/max.

```python
y_naive = scaler.inverse_transform(r['X_val'][:, -1, 0].reshape(-1, 1)).flatten()
```
**What:** The **naive baseline** prediction. `X_val[:, -1, 0]` = for each validation sequence, take the last time step (`-1`), first feature (`0` = the index). In other words: "predict that next month equals this month." This is the simplest possible forecast.

```python
lstm_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
lstm_mae = mean_absolute_error(y_actual, y_pred)
```
**What:**
- **RMSE**: $\sqrt{\frac{1}{n}\sum(y_{actual} - y_{pred})^2}$ — penalizes big mistakes heavily
- **MAE**: $\frac{1}{n}\sum|y_{actual} - y_{pred}|$ — average absolute error in index points

```python
'RMSE_Improvement_%': round((1 - lstm_rmse/naive_rmse) * 100, 1)
```
**What:** If LSTM_RMSE < Naive_RMSE, improvement is positive (LSTM is better). If negative, naive wins.

---

### Cell 30 — Actual vs Predicted Plots (lines 922–954)

```python
split_month = int(len(df_monthly) * TRAIN_SPLIT)
val_dates = df_monthly.index[split_month:]
```
**What:** Gets the date labels for the validation period. `split_month` = month 60, so `val_dates` = months 60 through 74 (15 dates matching our 15 predictions).

```python
ax.plot(dates, r['y_actual'], label='Actual', color='black')
ax.plot(dates, r['y_pred'], label='LSTM', color='steelblue', linestyle='--')
ax.plot(dates, r['y_naive'], label='Naive', color='coral', linestyle=':')
```
**What:** For each sector, overlays 3 lines:
- **Black solid** = actual values (ground truth)
- **Blue dashed** = LSTM predictions
- **Coral dotted** = naive baseline

If the blue line tracks the black line better than the coral line, the LSTM learned something useful.

---

### Cell 31 — RMSE/MAE Bar Chart (lines 957–989)

```python
axes[0].bar(x - width/2, df_eval['LSTM_RMSE'], width, label='LSTM')
axes[0].bar(x + width/2, df_eval['Naive_RMSE'], width, label='Naive')
```
**What:** Side-by-side bar chart comparing LSTM error vs naive error for each sector. Shorter bars = more accurate.

---

### Cell 33 — Future Forecasts (lines 1007–1070)

This is **autoregressive forecasting** — predicting beyond the known data.

```python
last_window = features[-WINDOW_SIZE:].copy()
```
**What:** Takes the last 12 months of actual data as the starting input.

```python
for step in range(FORECAST_MONTHS):    # 6 steps
    input_seq = last_window.reshape(1, WINDOW_SIZE, 3)
    pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
```
**What:** Feed the 12-month window into the model → get 1 prediction.

```python
    next_month = (last_month % 12) + 1
    next_sin = np.sin(2 * np.pi * next_month / 12)
    next_cos = np.cos(2 * np.pi * next_month / 12)
```
**What:** Calculate sin/cos features for the predicted month (the model needs these as input for the next step).

```python
    new_row = np.array([[pred_scaled, next_sin, next_cos]])
    last_window = np.vstack([last_window[1:], new_row])
```
**What:** **Slide the window forward**: drop the oldest month (`[1:]`), append the new prediction at the end. Now the window contains 11 real months + 1 predicted month. Feed this back in for the next prediction.

**Key insight:** Each prediction uses the previous prediction as input. Errors compound — month 6's forecast is less reliable than month 1's.

```python
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                              periods=FORECAST_MONTHS, freq='ME')
```
**What:** Creates 6 future date labels (e.g., May 2026, Jun 2026, ..., Oct 2026).

---

### Cell 34 — Forecast Visualization (lines 1073–1116)

```python
recent = df_monthly[sector].iloc[-24:]
ax.plot(recent.index, recent.values, label='Historical')
ax.plot(f['dates'], f['values'], linestyle='--', marker='o', label='Forecast')
```
**What:** Shows the last 24 months of real data (blue solid) + 6 months of forecast (red dashed with dots).

```python
for j in range(len(f['values'])):
    uncertainty = (j + 1) * 3
    ax.fill_between([f['dates'][j]], [f['values'][j] - uncertainty],
                    [f['values'][j] + uncertainty], color='red', alpha=0.1)
```
**What:** Draws growing red shading around each forecast point. Month 1 has ±3 uncertainty, month 6 has ±18. Visually shows that further forecasts are less certain.

---

### Cell 35 — Forecast Summary Table (lines 1119–1149)

```python
change_pct = ((end_val - current) / current) * 100
trend = "Growing ↑" if change_pct > 2 else ("Declining ↓" if change_pct < -2 else "Stable →")
```
**What:** Compares the 6-month forecast to current value. Labels each sector as growing, declining, or stable.

---

### Cell 37 — Model Summary (lines 1197–1216)

```python
sample_model.summary()
```
**What:** Prints the LSTM architecture as a table — layer names, output shapes, and parameter counts. Useful for your report to show exactly what the model looks like.

---

### TL;DR — The Full Pipeline in One Sentence Each

1. **Load** the CSV of daily job postings per sector
2. **Filter** to 8 tech-relevant sectors, keep only "total postings"
3. **EDA**: plot trends, distributions, missing data, correlations, seasonality
4. **Aggregate** daily → monthly (average), fill gaps
5. **Normalize** each sector to [0,1], add sin/cos month features, create 12-month sliding windows, split 80/20 in time order
6. **Train** a 2-layer LSTM (64→32 units, dropout 0.2) for each sector with early stopping
7. **Hyperparameter sensitivity**: test 5 configs (varying layer sizes & window lengths) to validate architecture choice
8. **Evaluate**: compare LSTM predictions vs naive baseline using RMSE/MAE
9. **Forecast** 6 months ahead by feeding predictions back as input
10. **Interpret** results and discuss limitations