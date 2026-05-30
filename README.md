# Steam Video Game Recommender System

A collaborative filtering recommender system that suggests video games to Steam users based on their purchase and play behaviour. The system is built with **PySpark's Alternating Least Squares (ALS)** algorithm, tuned with cross-validation, tracked with **MLflow**, and evaluated against baselines using ranking metrics (NDCG@10).

The full workflow lives in [`recommender_system.ipynb`](recommender_system.ipynb).

Click [here](https://htmlpreview.github.io/?https://github.com/itsveence/steam_video_game_recommender_sys/blob/master/recommender_system.html) to preview the notebook.

## Overview

The dataset contains implicit feedback — users don't rate games explicitly; instead we observe whether they purchased a game and how long they played it. The pipeline turns this signal into a relevance score, factorises the sparse user–item matrix with ALS, and produces personalised top-N recommendations. It can also **explain** why a particular game was recommended to a particular user.

### Pipeline

1. **Data loading** — Steam interaction data (~200k rows of `purchase` / `play` events) is read into a Spark DataFrame.
2. **Exploratory data analysis** — inspects behaviour distribution, playtime skew, and computes matrix sparsity (~99.8%).
3. **Preprocessing** — normalises game names (lowercase, whitespace collapsed).
4. **Feature engineering**
   - A **relevance score** is computed per user–game pair. Playtime is log-transformed via `ln(1 + hours)` to tame the extreme skew (playtime ranges from 0.1 to 11,754 hours), then summed with the purchase signal.
   - Game names are encoded to integer `game_id` values using Spark's `StringIndexer`, since ALS requires integer item identifiers.
5. **Modeling** — an 80/20 train/test split, then ALS trained with 3-fold cross-validation over a hyperparameter grid. `implicitPrefs=True` switches Spark to the Weighted Regularised Matrix Factorisation objective for implicit feedback.
6. **Evaluation** — the tuned model is scored with **NDCG@10** and compared against a random baseline and an overall most-popular baseline.
7. **Recommendation & explanation** — generates top-N games for given users, and explains recommendations by decomposing the contribution of each item in a user's history via embedding dot products.

### Modeling notes

- **Tuned hyperparameters:** `rank` ∈ {10, 50, 100}, `regParam` ∈ {0.01, 0.05, 0.1}, `maxIter` ∈ {5, 10}.
- **Fixed parameters:** `coldStartStrategy="drop"` (avoids NaN predictions for unseen users/items), `implicitPrefs=True`.
- RMSE is used as a *proxy* metric during cross-validation (because Spark's `CrossValidator` is not directly compatible with `RankingEvaluator`), while **NDCG@10** is used for final evaluation since it reflects ranking quality, which is what actually matters for recommendations.

## Dataset

[Steam Video Games](https://www.kaggle.com/datasets/tamber/steam-video-games) (Tamber, Kaggle). Each row is a `user_id, game_name, behaviour, value` tuple, where `behaviour` is either `purchase` (value `1`) or `play` (value = hours played).

The notebook automatically downloads the zipped dataset and extracts it to `data/steam-200k.csv`.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- Java (required by PySpark / Spark 4.x)

Core dependencies (see [`pyproject.toml`](pyproject.toml)):

- `pyspark` — distributed data processing and the ALS model
- `mlflow` — experiment tracking and model registry
- `ipykernel` — running the notebook

## Setup

Install dependencies with uv:

```bash
uv sync
```

Optionally configure the log level by creating a `.env` file:

```bash
LOGGING_LEVEL=INFO   # or DEBUG
```

## Usage

### 1. Start the MLflow tracking server

Run this in a separate terminal before executing the notebook:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

The MLflow UI is then available at http://127.0.0.1:5000.

### 2. Run the notebook

```bash
uv run jupyter lab recommender_system.ipynb
```

Run the cells top to bottom. Runs, parameters, metrics, and the best model are logged to MLflow automatically.

## Project Structure

```
.
├── recommender_system.ipynb   # Main notebook: EDA -> feature engineering -> ALS -> evaluation
├── data/
│   └── steam-200k.csv         # Steam interaction dataset
├── utils/
│   └── logger.py              # Colored console logging setup
├── mlartifacts/               # MLflow artifact store (logged models)
├── mlflow.db                  # MLflow tracking backend (SQLite)
├── pyproject.toml             # Project metadata and dependencies
└── uv.lock                    # Locked dependency versions
```

## Results

The ALS model is evaluated with NDCG@10 against two baselines:

- **Random baseline** — 10 random games per user (lower bound, no personalisation).
- **Overall Top-K baseline** — the 10 most popular games to everyone (stronger, but still no personalisation).

The model demonstrates interpretability: for example, *The Elder Scrolls V: Skyrim* being recommended to a user can be traced back to their history with *Borderlands 2* — both action role-playing games — confirming the recommendation captures genuine taste similarity.

> Exact metric values depend on the run; see the MLflow UI after training.

## References

1. Apache Spark. *[Collaborative Filtering — Spark MLlib Programming Guide](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)*.
2. Hu, Koren & Volinsky. *[Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22)* (ICDM 2008).
3. DigitalSreeni. *[Alternating Least Squares (ALS) — Building your first production-ready recommender system](https://www.youtube.com/watch?v=E6n0-BNsDFE)* [Video].
