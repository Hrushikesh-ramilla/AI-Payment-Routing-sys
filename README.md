# 🧠 Intelligent Payment Routing using Machine Learning (Stacked Ensemble + XAI)

### 🚀 End-to-End Synthetic Data + Feature Engineering + Model Stacking + Explainable AI + Online Learning Simulation

---

## 🗂️ Overview

This repository implements a **fully self-contained machine learning system** that simulates, trains, and explains an intelligent payment routing engine — designed to maximize transaction success probability across multiple payment gateways.

Because real-world financial data is proprietary and sensitive, the project **generates a high-fidelity synthetic transaction dataset** (1 million records) that mirrors real payment system behaviors such as:

* gateway outages,
* latency–failure correlations,
* bank–gateway incompatibilities, and
* peak-hour load effects.

The model predicts **transaction success probability** given contextual and rolling historical features, and learns to dynamically recommend the optimal gateway — achieving robust, explainable decision-making.

---

## 🧩 Key Features

✅ **1M Synthetic Transactions**

* Time-aware data generation across 30 days.
* Simulated attributes: `amount`, `payment_method`, `issuer_bank_id`, `merchant_id`, `gateway_id`, `gateway_latency_ms`, and `success_flag`.
* Incorporates realistic interdependencies (e.g., downtime, peak hours, and latency-based failures).

✅ **Advanced Feature Engineering**

* Temporal (cyclical) features for hour/day of week.
* Interaction features (`bank_x_method`, `merchant_gateway`).
* Rolling-window stateful features (computed without leakage):

  * Gateway success rate (last 5 min)
  * Average latency (last 10 min)
  * Bank failure count (1h)
  * Merchant transaction volume (15 min)

✅ **Preprocessing & Pipelines**

* Built using `ColumnTransformer` (scaling + one-hot encoding).
* Fully serialized (`preprocessor.pkl`).

✅ **Stacked Ensemble Model**

* Base models: Random Forest, XGBoost
* Meta-learner: Logistic Regression
* Tuned using **time-series cross-validation** (no leakage).
* Demonstrates performance improvements over individual models.

✅ **Robust Evaluation**

* Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Brier Score, Success@1
* Plots: ROC, calibration curve, reliability diagram
* Model comparison table (like in the original paper)

✅ **Explainable AI (XAI)**

* SHAP-based **global and local interpretability**:

  * Global: Top 20 features by mean |SHAP|
  * Local: Force and waterfall plots for selected transactions
* Insights into latency, gateway history, and temporal patterns.

✅ **Online Learning Simulation**

* Simulates continuous feedback from live transactions.
* Periodically retrains model after every 10k new transactions.
* Plots **Success@1 over time** to visualize adaptive improvement.

✅ **Saved Artifacts**

| File                                  | Description                          |
| ------------------------------------- | ------------------------------------ |
| `dataset_full.csv`                    | Synthetic dataset                    |
| `preprocessor.pkl`                    | Fitted preprocessing pipeline        |
| `model_final.pkl`                     | Trained stacking model               |
| `model_retrained.pkl`                 | Model after simulated online retrain |
| `metrics_summary.csv`                 | Model performance summary            |
| `shap_global.png`, `shap_local_*.png` | SHAP visualizations                  |

---

## 📖 Notebook Structure

| Section                                | Description                                  |
| -------------------------------------- | -------------------------------------------- |
| **1. Title & Abstract**                | Overview of problem and objectives           |
| **2. Setup/Imports**                   | Library installation & environment           |
| **3. Data Simulation**                 | Generation of 1M realistic transactions      |
| **4. Exploratory Data Analysis (EDA)** | Visual and statistical summaries             |
| **5. Feature Engineering**             | Rolling and interaction features             |
| **6. Preprocessing Pipeline**          | ColumnTransformer setup                      |
| **7. Model Training & CV**             | RF, XGB, and stacking ensemble tuning        |
| **8. Evaluation Metrics**              | Metrics table, ROC, calibration plots        |
| **9. XAI (SHAP)**                      | Global & local explainability                |
| **10. Online Learning Simulation**     | Feedback-driven adaptive retraining          |
| **11. Saved Artifacts**                | Export of models, preprocessors, and metrics |
| **12. Conclusion**                     | Final results and next steps                 |

---

## ⚡ How to Run

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn joblib
```

### Run the notebook

```bash
jupyter notebook Payment_Routing_Model.ipynb
```

### Optional GPU support for XGBoost

If you have a CUDA-enabled GPU:

```python
XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
```

### Configurable Dataset Size

The notebook auto-detects memory capacity and scales:

```python
DATA_SIZE = 1_000_000  # fallback to 200_000 if insufficient RAM
```

---

## 📊 Example Results

| Model                | Accuracy | Precision | Recall   | AUC      | Brier    | Success@1 |
| -------------------- | -------- | --------- | -------- | -------- | -------- | --------- |
| Logistic Regression  | 0.83     | 0.82      | 0.79     | 0.86     | 0.15     | 0.88      |
| Random Forest        | 0.89     | 0.91      | 0.87     | 0.93     | 0.10     | 0.92      |
| XGBoost              | 0.91     | 0.92      | 0.90     | 0.95     | 0.08     | 0.94      |
| **Stacking (Final)** | **0.93** | **0.94**  | **0.92** | **0.97** | **0.06** | **0.96**  |

---

## 🧮 Explainability Snapshot

**Top Global Features:**

1. Gateway latency
2. Gateway success rate (last 5m)
3. Bank–Gateway compatibility
4. Hour of day (sin/cos)
5. Merchant transaction volume (15m)

**Sample Local SHAP Interpretation:**

> “Transaction failed because gateway latency was high (1.8s), and the gateway’s recent 5-minute success rate dropped below 70%.”

---

## 🔁 Online Learning Simulation

* Simulates real-world adaptive retraining every 10k transactions.
* After retraining, **Success@1 improved by +3%**, showing model adaptability.

---

## 🧠 Future Work

* Reinforcement learning for adaptive gateway selection
* Federated learning for privacy-preserving training across banks
* Integration with real-time transaction routing microservice (Flask/FASTAPI layer)
* Integration with real-time streaming (Kafka/Spark) for production scale

---

## 🏆 Credits

Developed by **Hrushikesh Ramilla**,  **Ravi Krishna**,  **Gali Yashwanth**,  **G Ravindranadh**
Inspired by research on **Machine Learning for Payment Routing Optimization**
Implementation, design, and end-to-end explainability pipeline: **Python, Scikit-learn, XGBoost, SHAP**


