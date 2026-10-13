# 🛍️ Retail Sales Analytics Dashboard

A **complete, beginner-friendly data analytics project** built for B.Tech 1st-semester students. It demonstrates the **full analytics lifecycle** — from raw data cleaning to machine learning — using Python and Streamlit.

---

## 📌 Project Overview

This dashboard analyses an e-commerce behaviour dataset to answer real business questions about sales, customers, and products. It is structured as seven learning modules (M1–M7) that progress from basic data wrangling to predictive modelling, including a dedicated **Customer Purchase Prediction** module (M7) that classifies customer interactions as **Likely to Purchase** or **Not Likely to Purchase** using Logistic Regression and Random Forest.

---

## 📂 Dataset Description

**Ecommerce Behavior Dataset (Multi-Category Store)** — available on Kaggle.

| Column | Description |
|---|---|
| `event_time` | Timestamp of the event |
| `event_type` | view / cart / purchase |
| `product_id` | Unique product identifier |
| `category_id` | Unique category identifier |
| `category_code` | Hierarchical category (e.g. `electronics.smartphone`) |
| `brand` | Product brand |
| `price` | Product price |
| `user_id` | Unique user identifier |
| `user_session` | User session ID |

> **No Kaggle account?** The app automatically generates a realistic 50,000-row synthetic dataset so you can explore every feature immediately.

---

## ✨ Features

| Module | What You Learn |
|---|---|
| **M1 – Data Cleaning** | Loading, missing values, duplicates, type conversion |
| **M2 – Feature Engineering & EDA** | Time features, revenue, category extraction, business insights |
| **M3 – Probability Analysis** | Marginal & conditional probabilities, conversion funnel |
| **M4 – Statistical Inference** | Confidence intervals, T-Tests, hypothesis testing |
| **M5 – Regression Analysis** | Linear regression, MAE/RMSE/R², actual vs predicted plots |
| **M6 – Streamlit Dashboard** | Multi-page app, Plotly charts, KPI cards, interactive filters |
| **M7 – Customer Purchase Prediction** | Full pipeline: problem definition, EDA, outlier check, preprocessing, Logistic Regression vs Random Forest, Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrices, feature importance, model persistence, deployment guide, live prediction interface |

The home dashboard also provides a downloadable PDF EDA report containing dataset
quality checks, event and category pie charts, descriptive statistics, sales and
category performance, customer and product analysis, predictive model metrics with
an actual-vs-predicted chart, and actionable business recommendations.

---

## 🎯 Customer Purchase Prediction (M7)

### 1. Problem Definition

- **Problem:** predict whether a customer interaction will result in a **purchase** or **no purchase** (binary classification).
- **Why it matters:** most visitors browse without buying; identifying likely buyers focuses marketing spend and personalisation where it pays off.
- **Who uses it:** e-commerce marketing/growth teams, recommendation engines, CRM targeting.
- **Expected output:** `Likely to Purchase` / `Not Likely to Purchase` + a purchase probability (0–100%).

### 2–3. Dataset Collection & Data Understanding

Uses the same **Ecommerce Behavior Dataset** schema already in `data/raw/` (or the auto-generated synthetic fallback). Target column: `event_type` (`purchase` vs `view`/`cart`). Data Understanding stats (records, dtypes, missing values, duplicates, unique values, statistical summary) are computed live on the **Purchase Prediction** page via `classification.dataset_overview()` — no hardcoded numbers.

### 4. Data Preprocessing

- Missing values handled in `src/cleaning.py` (`brand`/`category_code` → `'unknown'`, rows missing `price` dropped).
- Duplicate rows removed.
- **Outlier detection** via IQR method on `price` (`classification.detect_outliers_iqr`) — outliers are reported but kept, since high-value products are legitimate catalogue items, not data errors.
- Categorical (`category_main`, `brand`) → One-Hot Encoded; Numerical (`price`, `hour`, `weekday`, `is_weekend`) → Standard Scaled — both via `ColumnTransformer`, **fit only on the training split**.

### 5. Exploratory Data Analysis

The Purchase Prediction page renders: price histogram, event-type count plot, price-by-event-type box plot (outlier view), price-vs-hour scatter plot, and a correlation heatmap of numeric features vs. the target.

### 6. Feature Selection / Engineering

- **Used:** `price`, `hour`, `weekday`, `is_weekend`, `category_main`, `brand` — all known at interaction time (no leakage).
- **Removed:** `product_id`, `category_id`, `user_id`, `user_session` — high-cardinality identifiers with no generalisable signal.
- **Engineered:** `is_weekend` (from `weekday`), `category_main` (top-level category parsed from `category_code`).

### Why interaction-level, not session-level?

The dataset's sessions average **~1.03 events per session** (each `user_session` almost always has exactly one interaction), so aggregating features per session collapses back to the row itself and creates a severe class imbalance if a session's only event is a purchase. This module therefore predicts purchase likelihood **per customer interaction**. This avoids target leakage and reflects the real structure of the data.

### 7. Train-Test Split

80/20 stratified split, `random_state=42`. Both models are evaluated on the identical test set.

### 8–9. Model Building, Evaluation & Comparison

`LogisticRegression` and `RandomForestClassifier` (`n_estimators=200, max_depth=10`), both with `class_weight="balanced"` so they don't just predict the majority "no purchase" class.

**Actual Results** (synthetic 50,000-row dataset, purchase rate ≈ 11.2%):

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression  | 0.505    | 0.107     | 0.467  | 0.174    | 0.491   |
| Random Forest        | 0.735    | 0.115     | 0.204  | 0.147    | 0.496   |

**Best model (by F1-Score): Logistic Regression.**

> ⚠️ Honest note: the sample dataset is randomly generated (event type is assigned independently of price/category/time), so there is very little genuine predictive signal — ROC-AUC ≈ 0.49–0.50 (≈ random guessing) confirms this honestly. This is expected and is a good talking point for the viva about *why data quality matters* and *why F1/AUC can look weak even with correct methodology*. Plugging in the real Kaggle multi-category-store CSV (which has genuine repeat-session behaviour) would likely produce stronger, more realistic metrics with the same code.

### 10. Save the Final Model

Every training run persists artifacts with `joblib`:

```
models/
├── preprocessing_pipeline.pkl   ← ColumnTransformer (scaler + one-hot encoder)
├── logistic_regression.pkl
├── random_forest.pkl
└── best_model.pkl
```

The app loads these fitted objects for prediction — preprocessing logic is never re-implemented in the UI.

### 11. Deployment

The app is deployment-ready as-is (single entry point `app.py`, pinned `requirements.txt`, no external services). To deploy on **Streamlit Community Cloud** (free):

1. Push this project to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo/branch, set main file to `app.py`, click **Deploy**.
4. Share the generated `https://<app-name>.streamlit.app` link.

For container platforms (Render, Railway, Hugging Face Spaces), the same `requirements.txt` plus `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` works unchanged.

### Files

- `src/classification.py` — data understanding, outlier detection, preprocessing pipeline, training, evaluation (incl. ROC-AUC), model persistence, prediction function
- `pages/6_Purchase_Prediction.py` — full UI covering every step above: EDA, preprocessing, metrics, ROC curve, confusion matrices, feature importance, live predictor, deployment guide

---

## 📁 Folder Structure

```
retail-sales-dashboard/
│
├── data/
│   ├── raw/               ← Place Kaggle CSV here (optional)
│   └── processed/         ← Auto-generated cleaned & featured data
│
├── src/
│   ├── cleaning.py        ← M1: Data cleaning pipeline
│   ├── feature_engineering.py  ← M2a: Feature creation
│   ├── eda.py             ← M2b: EDA helper functions
│   ├── probability.py     ← M3: Probability analysis
│   ├── statistics.py      ← M4: Statistical inference
│   ├── regression.py      ← M5: Linear regression model
│   ├── visualizations.py  ← M6: All Plotly chart builders
│   └── classification.py  ← M7: Purchase prediction (LogReg + Random Forest)
│
├── pages/
│   ├── 1_Sales_Analysis.py
│   ├── 2_Customer_Analysis.py
│   ├── 3_Product_Analysis.py
│   ├── 4_Probability_and_Statistics.py
│   ├── 5_Predictive_Analytics.py
│   └── 6_Purchase_Prediction.py
│
├── models/                 ← Persisted final models (joblib) for M7
│   ├── preprocessing_pipeline.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── best_model.pkl
│
├── requirements.txt
├── app.py                  ← Home page (entry point)
└── README.md
```

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.9 or later
- pip

### Step 1 – Clone / Download the Project

```bash
# If using git
git clone <your-repo-url>
cd retail-sales-dashboard
```

### Step 2 – (Optional) Add Kaggle Dataset

Download the dataset from Kaggle:
> https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Place any `.csv` file inside `data/raw/`:

```
data/raw/2019-Oct.csv
```

If no CSV is present, the app generates synthetic data automatically.

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📸 Dashboard Screenshots

| Page | Description |
|---|---|
| 🏠 Home | KPI cards + Revenue trend + Category breakdown |
| 📊 Sales Analysis | Revenue by category, time trend, monthly view with filters |
| 👥 Customer Analysis | Activity distribution, top spenders, repeat rate |
| 📦 Product Analysis | Top viewed/purchased products, category pie chart |
| 🎲 Probability & Stats | Event probabilities, conversion funnel, CI, T-Test |
| 🤖 Predictive Analytics | Model metrics, actual vs predicted, interactive predictor |
| 🛒 Purchase Prediction | Model comparison, confusion matrices, feature importance, live "will they buy?" predictor |

---

## 💡 Key Business Insights

1. **High browse-to-buy gap** — Only ~10% of events are purchases; improving the cart-to-purchase step yields the biggest ROI.
2. **Category concentration** — 2–3 categories drive ~80% of revenue (Pareto principle).
3. **VIP customers** — A small group of customers accounts for a disproportionate share of total spend; loyalty programmes are high-value.
4. **Weekend effects** — Statistical testing reveals whether weekend shoppers behave differently, informing promotion timing.
5. **Peak sales periods** — Daily trend analysis highlights high-traffic days for inventory and staffing planning.

---

## 🔮 Future Improvements

- [ ] Add cohort analysis (customer retention over time)
- [ ] Implement Random Forest or XGBoost for better price prediction
- [ ] Add market basket analysis (frequently bought together)
- [ ] Integrate real-time data via Kafka or API
- [ ] Add user authentication to the Streamlit dashboard
- [x] Export an EDA report as PDF from within the dashboard
- [ ] Deploy to Streamlit Community Cloud for public access

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Plotly | Interactive visualisations |
| Streamlit | Web dashboard framework |
| SciPy | Statistical tests |
| Scikit-Learn | Machine learning |
| ReportLab | PDF report generation |

---

## 📚 Learning Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Express Guide](https://plotly.com/python/plotly-express/)
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

*Built as a beginner data analytics portfolio project.*
