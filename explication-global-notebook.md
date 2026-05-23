# Complete Explanation of `job-forecasting-global.ipynb`

## For Someone Who Knows Nothing About ML/DL

---

## Part 1 — The Big Picture (What This Notebook Does)

**Goal:** Predict what the global job market will look like 6 months from now, for each sector (Software Development, Marketing, etc.).

**How:** Feed historical job posting data (Feb 2020 → Apr 2026) into a neural network that learns patterns and uses them to predict the future.

**Analogy:** Imagine you record the temperature every day for 6 years. You notice patterns: summer is hot, winter is cold, and there's a global warming trend. A neural network does the same thing — it finds patterns in numbers and uses them to guess what comes next.

---

## Part 2 — Key Vocabulary (Must Know)

| Term | Simple Explanation |
|------|-------------------|
| **Neural Network** | A computer program that learns patterns from data, inspired by how brain neurons connect |
| **Deep Learning (DL)** | Using neural networks with multiple "layers" (hence "deep") |
| **Machine Learning (ML)** | Broader field: algorithms that learn from data without being explicitly programmed |
| **AI** | Umbrella term for machines doing tasks that normally require human intelligence |
| **Training** | Showing the model thousands of examples so it learns patterns |
| **Epoch** | One complete pass through all training data |
| **Loss function** | A number that measures how wrong the model's predictions are (lower = better) |
| **Overfitting** | When the model memorizes training data but fails on new data (like memorizing exam answers instead of understanding the subject) |
| **Time series** | A sequence of values measured over time (e.g., monthly job index) |
| **Forecasting** | Predicting future values of a time series |

---

## Part 3 — The Data

### Source
- **Dataset:** Indeed Job Postings Index from Kaggle (`kimminh21/job-postings`)
- **Countries:** US, Australia, Canada, Germany, France, UK (6 total)
- **Time span:** February 2020 → April 2026 (~75 months)
- **What it measures:** An index where 100 = the level of job postings in Feb 2020 (pre-COVID baseline)

### What Variable Is Used
- **`new postings`** — the number of NEW job ads posted that day (not total active postings)
- Why: new postings reflect current hiring intent, while total postings can stay high just because old listings haven't been removed

### How It Becomes "Global"
- For each month, compute the average index for each country
- Then average all 6 countries together → one global number per sector per month
- This smooths out country-specific quirks and gives a worldwide trend

### Sectors (6 kept)
Only sectors present in ALL 6 countries are kept:
- Software Development, IT Systems & Solutions, Project Management, Marketing, Management, Human Resources
- Data & Analytics and Banking & Finance are **dropped** because they don't exist in every country's dataset

---

## Part 4 — STL Decomposition (Step 3)

### What Is STL?
**STL = Seasonal and Trend decomposition using Loess**

Any time series can be seen as 3 components added together:

```
Observed = Trend + Seasonal + Residual
```

| Component | What it captures | Example |
|-----------|-----------------|---------|
| **Trend** | Long-term direction (up, down, flat) | "Software Dev hiring has been slowly declining since 2022" |
| **Seasonal** | Repeating yearly pattern | "January always has more postings than August" |
| **Residual** | Random noise / unpredictable spikes | "One weird month where data jumped for no clear reason" |

### Why Use It?
- **We keep:** Trend + Seasonal (the meaningful signal)
- **We discard:** Residual (the random noise)
- This gives the neural network cleaner data to learn from

### The `robust=True` Parameter
- Makes STL resistant to **outliers** (extreme values)
- Specifically: the COVID crash (March–June 2020) was an extreme anomaly. Without `robust=True`, STL would distort the trend line trying to fit that crash. With it, the crash is downweighted.

### Academic Question: "Why not use the raw data directly?"
> Answer: Raw data contains random noise that makes patterns harder to learn. STL removes noise while preserving the meaningful trend and seasonal components. The model generalizes better on clean data.

---

## Part 5 — Feature Engineering (Step 5)

"Features" = the input variables you feed into the model. The more informative they are, the better the model can learn.

### Feature 1: First Differencing

Instead of feeding the model the absolute index value (e.g., "Software Dev is at 120"), we feed the **change from month to month** (e.g., "Software Dev went up by +3 this month").

**Why?**
- The model needs **stationary** data (data whose statistical properties don't change over time)
- Raw values have a trend → non-stationary
- Differences remove the trend → stationary
- Academic term: **first-order differencing** or Δ (delta)

```
diff[t] = value[t] - value[t-1]
```

### Feature 2: Cyclical Month Encoding (sin/cos)

Months are encoded as:
```
sin(2π × month / 12)
cos(2π × month / 12)
```

**Why not just use month number (1–12)?**
- If you use raw numbers, the model thinks December (12) is "far" from January (1)
- But they're adjacent months! Using sin/cos places them on a circle where December and January are neighbors
- This is called **cyclical encoding**

### Feature 3: Rolling Standard Deviation (Volatility)

For each month, compute the standard deviation of the previous 3 months.

**What it captures:** How unstable/volatile the market has been recently.
- High volatility → market is turbulent, predictions are less certain
- Low volatility → market is stable

### Feature 4: COVID Binary Flag

A simple 0/1 indicator:
- 1 if the month is between March–June 2020
- 0 otherwise

**Why?** COVID caused an unprecedented crash. This flag tells the model "this period is abnormal, don't try to learn normal patterns from it."

### Feature 5: Multi-Sector Input

All 6 sectors are predicted **simultaneously** (multivariate output). The model sees all sectors' data as input and outputs predictions for all sectors at once.

**Why?** Sectors are correlated. When Software Dev goes up, Data & Analytics often follows. The model can exploit these cross-sector relationships.

### Summary of Features

| Feature | Count | Purpose |
|---------|-------|---------|
| Differenced sector values | 6 | Main signal (month-over-month change for each sector) |
| sin(month), cos(month) | 2 | Seasonal position in the year |
| Rolling std per sector | 6 | Volatility indicator |
| COVID flag | 1 | Mark anomalous period |
| **Total input features** | **15** | |
| **Total output targets** | **6** | (predict change for all sectors) |

---

## Part 6 — Data Splitting (Train / Val / Test)

The data is split **chronologically** (never shuffle time series!):

```
|---- Train (70%) ----|--- Val (15%) ---|--- Test (15%) ---|
Feb 2020                                                    Apr 2026
```

| Split | Purpose |
|-------|---------|
| **Train** | Model learns patterns from this data |
| **Validation** | Used during training to detect overfitting. Model never trains on this. |
| **Test** | Final evaluation AFTER training is complete. Never seen by the model at all. |

**Why 3 splits instead of 2?**
- If you only have train/test, you might tune your model to do well on test → you're cheating
- Validation is for tuning decisions (when to stop training, which architecture is best)
- Test is the final honest grade

**Academic Question: "Why not random split?"**
> Answer: Time series has temporal dependency. A random split would leak future information into training (data leakage). We always split chronologically.

---

## Part 7 — Scaling / Normalization

All features are scaled to [-1, 1] using **MinMaxScaler**.

**Why?**
- Neural networks work best when input values are small and centered around 0
- If one feature ranges [0, 200] and another [0, 1], the first one dominates learning
- Scaling puts everything on equal footing

**Critical rule:** The scaler is fit ONLY on training data, then applied to val/test. Otherwise you'd leak information about the future into training.

---

## Part 8 — Sliding Windows (Sequences)

The model doesn't see one month at a time. It sees a **window of 12 consecutive months** and predicts the 13th.

```
Window: [month1, month2, ..., month12] → Predict: month13
```

This is called `LOOK_BACK = 12`.

**Why 12?** Because there's a yearly cycle. 12 months lets the model see a full seasonal pattern before making a prediction.

The code creates these windows by sliding through the data:
```
Window 1: months [1–12]  → predict month 13
Window 2: months [2–13]  → predict month 14
Window 3: months [3–14]  → predict month 15
...
```

---

## Part 9 — The Neural Network Architectures

### What Is a Recurrent Neural Network (RNN)?

A regular neural network looks at one input and gives one output. An **RNN** processes sequences — it reads data **one step at a time** and maintains a "memory" of what it has seen so far.

Analogy: Reading a sentence word by word. Your understanding of word 5 depends on words 1–4. An RNN works the same way with time series data.

### Problem with Basic RNNs: Vanishing Gradient

Basic RNNs forget old information quickly. If you show it 12 months, it might only remember months 10–12 and forget months 1–4. This is called the **vanishing gradient problem**.

### LSTM (Long Short-Term Memory)

**LSTM** solves the vanishing gradient problem by adding **gates**:

| Gate | Purpose |
|------|---------|
| **Forget gate** | Decides what to throw away from memory |
| **Input gate** | Decides what new information to store |
| **Output gate** | Decides what to output from memory |

Think of it as a notepad with a smart eraser: it can selectively remember important things from far back and forget irrelevant things.

**In this notebook:** 48 units (48 "neurons" in the LSTM layer)

### GRU (Gated Recurrent Unit)

A **simplified LSTM** with only 2 gates instead of 3:
- **Reset gate:** How much past info to forget
- **Update gate:** How much new info to keep

**Compared to LSTM:**
- Fewer parameters → trains faster
- Often performs similarly
- Better for smaller datasets (less prone to overfitting)

**In this notebook:** 48 units

### BiLSTM (Bidirectional LSTM)

A normal LSTM reads left-to-right (past → future). A **BiLSTM** reads BOTH directions:
- Forward LSTM: reads month 1 → 12
- Backward LSTM: reads month 12 → 1
- Combines both → richer understanding of context

**Analogy:** When reading a sentence, sometimes the meaning of a word becomes clear from words that come AFTER it.

**In this notebook:** 24 units per direction × 2 directions = 48 total

### Academic Question: "Why compare 3 architectures?"
> Answer: No single architecture is universally best. Different data patterns favor different models. GRU has fewer parameters (better for small data), LSTM has more expressive memory, and BiLSTM captures bidirectional context. We train all three and select the best empirically.

---

## Part 10 — Regularization (Preventing Overfitting)

The notebook uses **5 different regularization techniques**:

| Technique | What it does | Analogy |
|-----------|-------------|---------|
| **Dropout (0.20)** | Randomly turns off 20% of neurons during each training step | Like studying with random notes removed — forces you to not rely on any single note |
| **Recurrent Dropout (0.20)** | Same but applied to the recurrent (memory) connections | Prevents the memory from being too dependent on any single pathway |
| **L2 Regularization (5e-3)** | Adds a penalty for large weights — pushes all parameters to stay small | Like a tax on complexity — forces the model to use simple patterns |
| **EarlyStopping (patience=20)** | Stops training when validation loss hasn't improved for 20 epochs | Stop studying once you've peaked — more studying only makes you overthink |
| **ReduceLROnPlateau** | Halves the learning rate when validation stalls for 8 epochs | When you hit a wall, take smaller steps to find the way |

### Academic Question: "What is the learning rate?"
> Answer: The learning rate controls how much the model adjusts its parameters after each batch. Too high → unstable, jumps over good solutions. Too low → stuck, trains forever. Starting at 0.001 and halving when progress stalls is a common strategy.

### Academic Question: "What is L2 regularization mathematically?"
> Answer: The loss function becomes: `Loss = Original_Loss + λ × Σ(w²)` where λ=0.005 and w are the model weights. This penalizes large weights, forcing the model to spread information across many small connections rather than relying on a few large ones.

---

## Part 11 — Loss Function: Huber Loss

Instead of the common MSE (Mean Squared Error), this notebook uses **Huber loss**.

### MSE vs Huber

| | MSE | Huber |
|-|-----|-------|
| Formula (for small errors) | error² | error² |
| Formula (for large errors) | error² (huge!) | \|error\| (linear, not squared) |
| Sensitivity to outliers | Very sensitive | Robust |

**Why it matters:** COVID months have extreme errors. With MSE, one bad month dominates the entire loss. Huber says "okay that month was bad, but let's not let it hijack the whole training."

The `delta=1.0` parameter is the threshold: errors below 1.0 are treated quadratically, above 1.0 are treated linearly.

### Academic Question: "Why not MAE (Mean Absolute Error) then?"
> Answer: MAE is not differentiable at zero, which causes numerical issues. Huber is smooth everywhere — it combines the best of MSE (smooth near zero) and MAE (robust to outliers).

---

## Part 12 — Ensemble Learning (7 Seeds)

### What Is an Ensemble?

Instead of training ONE model, we train **7 models** with different random initializations (seeds) and average their predictions.

### Why?

Neural networks start with random weights. Different starting points → different final models. Some will be better, some worse. By averaging 7, we:
1. **Reduce variance** (less sensitive to random luck)
2. **Get uncertainty estimates** (if all 7 agree → confident; if they disagree → uncertain)
3. **Improve accuracy** (averaging smooths out individual errors)

### The Seeds
```python
SEEDS = [7, 23, 42, 101, 314, 1729, 2718]
```
Each seed fixes the random number generator so results are **reproducible** (same seed → same model every time).

### Total Models Trained
- 3 architectures × 7 seeds = **21 models** total

### Uncertainty Bands
The 10th and 90th percentile across the 7 forecasts give the **confidence interval**:
- If all 7 models predict similar values → narrow band → high confidence
- If they disagree → wide band → low confidence

### Academic Question: "Why 7 seeds specifically?"
> Answer: Odd number (so there's always a clear majority), and enough to get statistical diversity without excessive training time. 7 is a common choice in the literature — enough to reduce variance while being computationally feasible.

---

## Part 13 — Evaluation Metrics

### RMSE (Root Mean Squared Error)
```
RMSE = √(mean of (predicted - actual)²)
```
- Same units as the data (index points)
- Penalizes large errors more than small ones
- **Lower is better**

### MAE (Mean Absolute Error)
```
MAE = mean of |predicted - actual|
```
- Average absolute mistake
- Easier to interpret: "on average, the model is off by X points"
- **Lower is better**

### R² (Coefficient of Determination)
```
R² = 1 - (sum of squared errors / sum of squared differences from mean)
```
- Ranges from -∞ to 1
- R² = 1 → perfect predictions
- R² = 0 → model is as good as always predicting the mean
- R² < 0 → model is WORSE than just guessing the mean
- **Higher is better**

### Academic Question: "What's the difference between RMSE and MAE?"
> Answer: RMSE penalizes big errors more (because of squaring). If you have a few very wrong predictions, RMSE will be much higher than MAE. If errors are uniform, they'll be similar. Using both gives a fuller picture.

---

## Part 14 — Baselines (Naive and Seasonal-Naive)

### Why Baselines?

A model is only useful if it beats simple heuristics. If a fancy neural network can't beat "just predict yesterday's value," it's worthless.

### Naive Baseline
```
prediction[t] = actual[t-1]
```
"Tomorrow will be the same as today." Dead simple but surprisingly hard to beat in many time series.

### Seasonal-Naive Baseline
```
prediction[t] = actual[t-12]
```
"This month will be the same as the same month last year." Captures yearly cycles.

### Academic Question: "What if the model doesn't beat the baselines?"
> Answer: Then the model is not adding value. It means the data doesn't have learnable patterns beyond simple persistence or seasonality, or the model is poorly configured. In this notebook, the DL models DO beat both baselines.

---

## Part 15 — Bias Correction

After training, the model might have a systematic offset (always predicting a bit too high or too low). 

**How it works:**
1. Run the model on the validation set
2. Compute the average error per sector: `bias = mean(actual - predicted)` on validation
3. Add this bias back to test/forecast predictions

**Example:** If the model always under-predicts Software Dev by 2 points, add +2 to all its Software Dev predictions.

**Important:** This is computed on VALIDATION data only — never on test.

### Academic Question: "Isn't this cheating?"
> Answer: No, because the bias is computed on validation data (which the model has already seen indirectly through early stopping). It's a standard post-processing step used in production forecasting systems. It corrects systematic errors without overfitting.

---

## Part 16 — Recursive Forecasting (Step 9)

To predict 6 months into the future, the model uses **autoregressive forecasting**:

```
Step 1: Use last 12 months of real data → predict month +1
Step 2: Use last 11 real months + predicted month +1 → predict month +2
Step 3: Use last 10 real months + predicted months +1,+2 → predict month +3
...
Step 6: Use last 6 real months + 5 predicted months → predict month +6
```

Each prediction feeds into the next as input. This is why it's called "recursive" or "autoregressive."

**Risk:** Errors compound. If month +1 is wrong, month +2 starts from a wrong position, and it gets progressively less accurate.

**Mitigation:** The ensemble (7 seeds) averages out individual errors, and bias correction fixes systematic drift.

---

## Part 17 — The Full Pipeline Summary

```
Raw CSVs (6 countries, daily)
    ↓ aggregate to monthly
Monthly per-country series
    ↓ average across countries  
Global monthly index (1 series per sector)
    ↓ STL decomposition
Denoised series (trend + seasonal, no noise)
    ↓ first differencing  
Stationary Δ values
    ↓ add features (sin/cos, rolling_std, COVID flag)
Feature matrix (15 features per month)
    ↓ scale to [-1, 1]
    ↓ create 12-month sliding windows
X_train, X_val, X_test
    ↓ train 21 models (3 archs × 7 seeds)
    ↓ evaluate on test set
Best architecture selected
    ↓ recursive 6-month forecast
    ↓ bias correction
    ↓ uncertainty bands (10th–90th percentile)
Final forecast with confidence intervals
```

---

## Part 18 — Potential Mentor Questions & Answers

### Q: "Why use deep learning and not classical methods (ARIMA, Prophet)?"
> A: Classical methods struggle with non-linear patterns and multi-sector correlations. RNNs can capture complex temporal dependencies that ARIMA's linear framework cannot. Also, this is an academic project meant to explore deep learning approaches.

### Q: "What is the vanishing gradient problem?"
> A: In basic RNNs, when training through backpropagation through time (BPTT), gradients get multiplied many times. If they're small (<1), they shrink exponentially (vanish). The network can't learn long-term dependencies. LSTM/GRU solve this with gating mechanisms that allow gradients to flow unchanged.

### Q: "What is backpropagation?"
> A: The algorithm for training neural networks. It computes how much each weight contributed to the error, then adjusts weights in the direction that reduces error. For time series, it's called "Backpropagation Through Time" (BPTT) because we unroll the RNN over each timestep.

### Q: "Why MinMaxScaler and not StandardScaler?"
> A: MinMaxScaler bounds values to a known range [-1,1], which works well with tanh activations in LSTM/GRU gates. StandardScaler (zero mean, unit variance) doesn't guarantee bounded inputs, which can be problematic for recurrent networks.

### Q: "What is the Adam optimizer?"
> A: Adam = Adaptive Moment Estimation. It maintains per-parameter learning rates that adapt during training. It combines two ideas: momentum (moving average of gradients) and RMSProp (moving average of squared gradients). It's the default choice for deep learning because it converges faster than basic SGD.

### Q: "How does EarlyStopping work?"
> A: After each epoch, check validation loss. If it hasn't improved for 20 consecutive epochs (patience=20), stop training and restore the weights from the best epoch. This prevents the model from training too long and overfitting.

### Q: "Why predict differences (Δ) instead of absolute values?"
> A: Stationarity. A stationary series has constant mean and variance over time. Neural networks learn better from stationary data because the statistical patterns don't shift. Differencing removes trends, making the data stationary.

### Q: "What does `batch_size=8` mean?"
> A: The model updates its weights after seeing 8 training examples (not after each one, and not after all of them). Small batches add noise → acts as regularization and helps escape bad local minima. Large batches are faster but can overfit.

### Q: "What's the difference between `dropout` and `recurrent_dropout`?"
> A: Regular dropout applies to the input-to-hidden connections (applied fresh each timestep). Recurrent dropout applies to the hidden-to-hidden connections (the memory path). Both prevent co-adaptation but in different parts of the architecture.

### Q: "Why use sin/cos for months instead of one-hot encoding?"
> A: One-hot creates 12 binary features and doesn't encode proximity (model doesn't know Nov is close to Dec). Sin/cos creates only 2 features and naturally encodes cyclical proximity — December and January are adjacent on the unit circle.

### Q: "What is the purpose of the COVID flag?"
> A: It's a binary indicator that tells the model "this period is an unprecedented external shock." Without it, the model might try to learn from COVID months as if they're normal patterns, which would hurt forecasting accuracy.

### Q: "Can you explain what `input_shape=(12, 15)` means?"
> A: The model receives sequences of 12 timesteps (months), each with 15 features. So the input tensor has shape (batch_size, 12, 15) — a 3D array where each sample is a 12×15 matrix.

### Q: "What would you improve?"
> A: (1) Add attention mechanism to focus on most relevant months, (2) Try Transformer architecture, (3) Add external features (GDP, unemployment rate), (4) Use cross-validation instead of single split, (5) Increase ensemble size, (6) Try different lookback periods.

---

## Part 19 — Quick Reference Card

| Concept | Value in This Notebook |
|---------|----------------------|
| Dataset | Indeed Job Postings, 6 countries |
| Target variable | `new postings` (first-differenced) |
| Sectors predicted | 6 (jointly) |
| Denoising | STL (trend + seasonal) |
| Input features | 15 (6 diff + 2 seasonal + 6 volatility + 1 COVID) |
| Look-back window | 12 months |
| Train/Val/Test split | 70% / 15% / 15% (chronological) |
| Architectures | LSTM(48), GRU(48), BiLSTM(24×2) |
| Regularization | L2(0.005) + Dropout(0.20) + RecurrentDropout(0.20) |
| Loss function | Huber (δ=1.0) |
| Optimizer | Adam (lr=0.001, with ReduceLROnPlateau) |
| Ensemble | 7 seeds per architecture (21 total) |
| Evaluation metrics | RMSE, MAE, R² |
| Baselines | Naive, Seasonal-Naive |
| Forecast horizon | 6 months (recursive) |
| Uncertainty | 10th–90th percentile across ensemble members |
| Post-processing | Bias correction from validation residuals |
