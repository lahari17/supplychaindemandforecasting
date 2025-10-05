## Supply Chain Demand Forecasting (Local) - Project Scope

### 1) Overview
- **Goal**: Build a local, reproducible pipeline to forecast 28-day unit sales at the bottom level (item_id x store_id) and provide coherent rollups across the hierarchy (item, store, dept, cat, state, total), with results saved as artifacts and an optional Kaggle-style `submission.csv`.
- **Context**: Uses Walmart M5-style dataset provided in this repository; no cloud infra (S3/Redshift) in scope for this phase.
- **Primary stakeholders**: Portfolio reviewers, hiring managers, and data science peers.

### 2) Objectives & Success Criteria
- **O1 - Accuracy**: Achieve competitive forecast accuracy measured by WRMSSE on the standard validation window (d_1914-d_1941). Target: beat seasonal-naive baseline by a meaningful margin (>= 10% WRMSSE improvement).
- **O2 - Reproducibility**: One-command local run to prepare data, train, evaluate, and generate forecasts.
- **O3 - Explainability**: Provide feature importance and brief error analysis by hierarchy level.
- **O4 - Coherence**: Provide bottom-up reconciliation to ensure aggregate consistency; optional MINt.
- **O5 - Artifacts**: Produce saved models, metrics report, plots, and `submission.csv` in `outputs/`.

### 3) Datasets (Local Files)
- `calendar.csv`: Calendar metadata for each day (wday, month, year, events, SNAP flags).
- `sell_prices.csv`: Weekly item/store prices (`wm_yr_wk`, `sell_price`).
- `sales_train_validation.csv`: Historical daily unit sales in wide format (d_1..d_1913).
- `sales_train_evaluation.csv`: Extended training (for final/evaluation phase, optional in local dev).
- `sample_submission.csv`: Sample 28-day forecast format (F1..F28) for validation/evaluation.

### 4) Forecasting Target & Horizon
- **Target**: Daily unit sales per `id` (bottom level: item_id x store_id).
- **Horizon**: 28 days (F1-F28) following the training cut-off.
- **Validation window**: d_1914-d_1941 (aligns to M5 public validation).

### 5) Evaluation Metrics
- **Primary**: WRMSSE (Weighted Root Mean Squared Scaled Error) across 12 aggregation levels.
- **Secondary**: RMSE/MAE at bottom level; level-wise error breakdown; seasonal-naive baseline comparison.
- **Service metrics**: Training runtime, peak memory, size of model/artifacts.

### 6) Methodology
- **Data preparation**:
  - Convert `sales_train_validation.csv` from wide to long format.
  - Join with `calendar.csv` and `sell_prices.csv`.
  - Persist intermediate datasets to Parquet in `data/` to speed iteration.
- **Feature engineering**:
  - Temporal: `wday`, `week`, `month`, `year`, month/quarter boundaries, holiday/event flags, SNAP.
  - Lags: 7/14/28 (optionally 35/42).
  - Rolling stats (leakage-safe): rolling mean/median/std over 7/14/28 with a 28-day shift.
  - Price: `sell_price`, price deltas, price change indicators, rolling price stats.
  - Encodings: `item_id` frequency (present in notebook); optional time-aware target encoding.
- **Models**:
  - Gradient boosting regressors (LightGBM/XGBoost) as primary.
  - Start with a single global model; optionally experiment with per-item or per-dept models.
  - Optional light ensembling with seasonal-naive/ETS.
- **Validation protocol**:
  - Rolling-origin holdout aligned to M5: train up to d_1913, validate on d_1914-d_1941.
  - Optional additional folds for robustness.
  - Use WRMSSE evaluator implemented locally to pick models/params.
- **Hierarchical reconciliation**:
  - Phase 1: Bottom-Up (BU) aggregation.
  - Phase 2 (optional): OLS/MINt with diagonal covariance approximation.
- **Outputs**:
  - `outputs/forecasts/preds_<run_id>.parquet` - bottom and aggregate forecasts.
  - `outputs/reports/metrics_<run_id>.json` - WRMSSE and breakdowns.
  - `outputs/plots/` - forecast vs actual; error by level/cohort.
  - `outputs/submission.csv` - Kaggle-style output for validation grid.

### 7) Local Project Structure
- `src/io.py`: Read/write helpers (CSV/Parquet), melt long, joins.
- `src/features.py`: Feature builders (lags, rolling, price/event, encodings) with leak-safe windows.
- `src/metrics_wrmsse.py`: WRMSSE computation, level weights, per-level metrics.
- `src/models.py`: Train/predict APIs for LightGBM/XGBoost, CV loops, early stopping.
- `src/reconcile.py`: Bottom-up and optional MINt reconciliation utilities.
- `src/cli.py`: CLI entry points: prepare, train, evaluate, predict, reconcile, submit.
- `notebooks/m5_local_dev.ipynb`: Exploration; mirrors features used in `src/`.
- `data/`: Cached Parquet datasets (local only, git-ignored).
- `outputs/`: Models, metrics, plots, submissions (git-ignored).

### 8) Environment & Tooling (Local)
- **Python**: 3.10+
- **Dependencies**: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`, `seaborn`, `tqdm`.
- **Setup** (example):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy pandas pyarrow scikit-learn lightgbm xgboost matplotlib seaborn tqdm
```

### 9) Deliverables
- Reproducible local pipeline (CLI) that outputs forecasts and evaluation.
- WRMSSE report with per-level breakdown, plots, and comparison vs seasonal-naive baseline.
- Clean `submission.csv` for validation window.
- Concise README with run instructions and brief discussion of results.

### 10) Milestones & Timeline (Local)
- **M1 - Data & caching (Day 1)**: Localize paths, melt/join, Parquet cache.
- **M2 - Features (Day 2)**: Implement lag/rolling/price/event features; leak-safe checks.
- **M3 - Metric & baseline (Day 2)**: Implement WRMSSE; seasonal-naive baseline.
- **M4 - Modeling (Day 3)**: Train LightGBM/XGBoost, early stopping, single-fold validation.
- **M5 - Reconciliation (Day 3)**: Bottom-up rollups; per-level metrics.
- **M6 - Reporting (Day 4)**: Plots, metrics JSON; generate `submission.csv`.
- **M7 - Polish (Day 4)**: Small HPO sweep; README; code tidy.

### 11) Risks & Mitigations
- **Data leakage via windows**: Enforce 28-day shift before rolling stats.
- **Memory limits**: Use Parquet caching; downcast dtypes; selective column reads.
- **Class imbalance/intermittent demand**: Add robust baselines; consider quantile loss later.
- **Metric mismatch**: Validate WRMSSE on a small, known subset to ensure parity with references.

### 12) Acceptance Criteria
- Single CLI command sequence runs end-to-end locally without errors.
- WRMSSE report shows improvement over seasonal-naive.
- Coherent aggregates present (BU) and verified by summation checks.
- Artifacts present in `outputs/` with timestamps/run IDs.

### 13) Out-of-Scope (for this phase)
- Cloud data lake/warehouse, streaming inference, advanced MINt covariance estimation beyond diagonal approximation.

### 14) Future Extensions (Optional)
- Add MINt reconciliation with better covariance estimates.
- Add quantile forecasts and probabilistic evaluation (e.g., pinball loss).
- Port to cloud stack (S3/EMR/SageMaker/Redshift) with feature store and MLOps.

### Appendix: WRMSSE (Brief)
- For each series, compute scale as the mean of squared first differences on the training window.
- Compute per-level weights following M5 Participants Guide (value-weighted aggregation).
- At each level, compute RMSSE (scaled RMSE), then apply level weights and sum to get overall WRMSSE.

### 15) PySpark (Local) Addendum
- **Purpose**: Enable PySpark for local ETL and feature engineering to handle wide-to-long transforms, joins, and rolling features efficiently.
- **Install**: `pip install pyspark`
- **Optional JVM**: Ensure Java is installed; set `JAVA_HOME` if needed.
- **Spark session (local)**:

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("m5-local")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
```

- **Reading CSVs** (headers, types inferred or specified):

```python
calendar_sdf = spark.read.option("header", True).csv("calendar.csv")
prices_sdf   = spark.read.option("header", True).csv("sell_prices.csv")
sales_wide   = spark.read.option("header", True).csv("sales_train_validation.csv")
```

- **Wide-to-long melt** (Sketch): unpivot day columns `d_1..d_1913` into rows. For local scope, use Spark SQL stack/unpivot or reshape via `selectExpr`:

```python
day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
id_cols = [c for c in sales_wide.columns if c not in day_cols]

long_exprs = []
for c in day_cols:
    long_exprs.append(f"SELECT {', '.join(id_cols)}, '{c}' AS d, CAST(`{c}` AS INT) AS sales FROM sales_wide")

sales_long = sales_wide.createOrReplaceTempView("sales_wide") or None
sales_long = spark.sql(" UNION ALL ".join(long_exprs))
```

- **Joins and caching to Parquet**:

```python
df = (sales_long
      .join(calendar_sdf, on="d", how="left")
      .join(prices_sdf, on=["store_id", "item_id", "wm_yr_wk"], how="left"))

df.write.mode("overwrite").parquet("data/curated.parquet")
```

- **Lag and rolling features with leak-safe 28-day shift**:

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

w_id = Window.partitionBy("id").orderBy(F.col("d").cast("int"))

# Lags
df = df.withColumn("sales_lag_7",  F.lag("sales", 7).over(w_id))
df = df.withColumn("sales_lag_14", F.lag("sales", 14).over(w_id))
df = df.withColumn("sales_lag_28", F.lag("sales", 28).over(w_id))

# Rolling means shifted by 28 (window ends 28 days before current row)
def shifted_mean(col, window):
    return F.avg(col).over(w_id.rowsBetween(-28 - window + 1, -28))

df = df.withColumn("rolling_mean_7",  shifted_mean("sales", 7))
df = df.withColumn("rolling_mean_14", shifted_mean("sales", 14))
df = df.withColumn("rolling_mean_28", shifted_mean("sales", 28))
```

- **Price/event features**: cast numeric types, compute pct-change on `sell_price` by id/store windows, encode events as binaries.
- **Interchange with Pandas**: Convert manageable subsets to Pandas with `toPandas()` for model training if not using Spark ML.
- **Resource tuning**: Adjust `spark.driver.memory`, parallelism; persist intermediate DataFrames to Parquet to limit re-computation.
