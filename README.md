# 🛡️ Credit Card Fraud Detection
### *Catching fraud before it costs a cent: an end-to-end ML pipeline with explainable AI*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Best%20Model-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.9725-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Precision-0.94-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Recall%20(Fraud)-Tuned-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## 🚀 Live Demo

👉 **[Try FraudShield on Streamlit](https://credit-card-fraud-detection-mxpbol3zjbrgzjamc3gj4q.streamlit.app)**

Interact with the live app to:
- Predict fraud on a single transaction via manual input
- Load built-in fraud/legitimate sample transactions
- Upload a CSV for batch prediction across thousands of transactions
- View SHAP feature importance explanations for every prediction

---

## 📌 Problem Statement

Credit card fraud costs the global financial industry **over $32 billion annually**. Every second a fraudulent transaction goes undetected, real customers bear the financial and emotional cost.

This project builds a production-ready fraud detection pipeline on a dataset of **284,807 real-world credit card transactions**, where genuine fraud accounts for just **0.17%** of all cases. This is one of the most extreme class imbalance scenarios in applied machine learning.

The challenge is not simply training a classifier. It is building a system that:
- **Catches nearly all fraud** (maximises recall) without burying investigators in false alarms
- **Explains every decision** so flagged transactions are auditable and defensible
- **Handles severe class imbalance** without synthetic oversampling that can leak information

> This problem is directly relevant to fraud operations teams at banks, fintech companies, and payment processors: any environment where the cost of a missed fraud far exceeds the cost of a false positive.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Source** | [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group) |
| **Size** | 284,807 transactions × 31 columns |
| **Fraud cases** | 492 (0.172% of total) |
| **Features** | `Time`, `Amount`, and 28 anonymised PCA components `V1–V28` |
| **Target** | `Class`: `0` = Legitimate, `1` = Fraud |

The PCA transformation was applied by the original authors to protect cardholder confidentiality. `Time` and `Amount` are the only non-anonymised features, making them critical anchors for any business-facing interpretation.

---

## 🔄 Project Workflow

```
Raw Data → Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Explainability → Deployment
```

| # | Stage | Key Actions |
|---|-------|-------------|
| 1 | **Data Loading** | Load 284,807 rows, inspect shape, dtypes, and structure |
| 2 | **Data Cleaning** | Remove 1,081 duplicate rows, standardise column names, verify data types, check for nulls, inspect outliers on V-features via z-scores |
| 3 | **EDA** | Visualise class imbalance, analyse transaction amount by class (log-scale boxplots), compute hourly fraud rate, plot V1–V28 distributions per class, compute target correlations |
| 4 | **Feature Engineering** | StandardScaler applied to `Amount` and `Time`; originals dropped to prevent scale leakage |
| 5 | **Data Preparation** | 80/20 stratified train-test split preserving the 0.17% fraud ratio |
| 6 | **Class Imbalance Handling** | `class_weight='balanced'` for sklearn models; `scale_pos_weight` (~578×) for XGBoost. No resampling required. |
| 7 | **Model Training** | Four classifiers trained and benchmarked (see §ML Models) |
| 8 | **Model Evaluation** | ROC-AUC comparison, precision-recall threshold tuning, confusion matrix at optimal threshold |
| 9 | **Feature Importance** | XGBoost built-in importance across Weight, Gain, and Cover metrics |
| 10 | **Model Explainability** | SHAP global bar plot, beeswarm, per-prediction waterfall plots, dependence plots |
| 11 | **Business Recommendations** | Actionable guidance for fraud operations teams derived from SHAP findings |
| 12 | **Deployment** | Streamlit app deployed on Streamlit Community Cloud with live demo |

---

## 🤖 Machine Learning Models Used

Four classifiers were benchmarked to select the strongest performer:

| Model | Why It Was Included |
|-------|---------------------|
| **Logistic Regression** | Interpretable baseline; establishes minimum viable performance |
| **Decision Tree** | Fast, non-linear baseline; highlights where shallow rules break down on imbalanced data |
| **Random Forest** | Strong ensemble benchmark; good out-of-the-box imbalance handling |
| **XGBoost** ✅ | Gradient boosted trees with `scale_pos_weight`, best suited for tabular imbalanced classification with native SHAP compatibility |

XGBoost was selected as the final model based on ROC-AUC performance on the held-out test set. Its `scale_pos_weight` parameter penalises fraud misclassification ~578× more heavily than legitimate misclassification, directly encoding the cost asymmetry of the problem into the loss function.

---

## 📈 Model Evaluation

> ⚠️ **Why accuracy is irrelevant here:** A model that predicts "legitimate" for every single transaction achieves 99.83% accuracy while catching zero fraud. This project reports metrics that actually matter.

### Model Comparison (ROC-AUC on Test Set)

| Model | F1-Score (Fraud) | ROC-AUC |
|-------|-----------------|---------|
| Logistic Regression | 0.1042 | 0.9686 |
| Decision Tree | 0.3584 | 0.8886 |
| Random Forest | 0.8171 | 0.9193 |
| **XGBoost** ✅ | **0.8506** | **0.9725** |

### Threshold Tuning

The default 0.5 decision threshold optimises for accuracy, not recall. Using the **Precision-Recall curve**, the threshold was tuned to the point that maximises the fraud-class F1-score:

| Metric | Default (0.5) | Tuned Threshold |
|--------|---------------|-----------------|
| Precision (Fraud) | 0.94 | **0.9383** |
| Recall (Fraud) | 0.78 | **0.8000** |
| F1-Score (Fraud) | 0.8506 | **0.8636** |

**93.8% precision** means that nearly 94 out of every 100 flagged transactions are genuine fraud, keeping investigation workload manageable while the improved recall ensures fewer real cases slip through.

### Why These Metrics Matter for Fraud

| Metric | Business Meaning |
|--------|-----------------|
| **Recall** | % of real fraud cases caught. This is the primary metric; missed fraud equals direct financial loss |
| **Precision** | % of fraud alerts that are real. Controls investigator workload and false alarm fatigue |
| **F1-Score** | Harmonic mean that balances precision and recall for the minority class |
| **ROC-AUC** | Model's overall ability to rank fraud above legitimate transactions across all thresholds |

---

## 🔍 Model Explainability (SHAP)

Built-in XGBoost importance scores reveal *which* features matter. SHAP reveals *how* and *why*, making every individual prediction fully auditable.

### Global Importance

The **mean |SHAP| bar chart** ranks features by average impact across the entire test set. `v14` dominates by a substantial margin. Its gain score (~6,098) is roughly **10× that of the next feature** (`v4`, ~588).

### Directional Signals (Beeswarm Plot)

| Feature | Direction | Interpretation |
|---------|-----------|----------------|
| `v14` | Low values → Fraud | Acts as a **threshold switch**: `v14 < -5` triggers near-certain fraud alerts |
| `v4` | High values → Fraud | **Continuous risk scaler** that monotonically increases fraud probability |
| `v12` | Low values → Fraud | Secondary confirming signal alongside `v14` |
| `v10` | Non-linear | Complex bidirectional relationship with fraud |
| `v11` | High values → Fraud | Elevates fraud risk across its upper range |

### Individual Prediction Waterfall

For a confirmed fraud case (model score: **f(x) = 7.583**):
- `v14 = -3.823` contributed **+6.74** (single largest push toward fraud)
- `v10 = -3.539` contributed **+3.63** (strong secondary signal)
- `v12 = -3.993` contributed **+1.36** (third confirming signal)
- Counterbalancing features (`v28`, `v8`, `v19`) were not strong enough to override

For a confirmed legitimate case (model score: **f(x) = -15.451**):
- Every top feature pushed firmly toward legitimate with no conflicting signals
- `v14 = -0.293` was slightly negative but far less extreme than the fraud case

> This contrast illustrates exactly what the model learned: **fraud signatures are not subtle. They are extreme, co-occurring anomalies across multiple PCA dimensions.**

---

## 💡 Key Insights & Findings

- **v14 is the dominant fraud signal.** Transactions with `v14 < -5` have SHAP contributions of +4 to +8, making them near-certain fraud cases. This single feature drives more predictive power than the next several features combined.

- **Fraud peaks in low-traffic hours.** Hourly fraud rate analysis revealed elevated fraud rates during periods of lower transaction volume, a pattern consistent with fraudsters exploiting periods of reduced monitoring.

- **Fraud transactions skew toward smaller amounts.** Despite a common assumption that fraud involves large transactions, the dataset shows fraud cases span a wide amount range with a mean lower than legitimate transactions.

- **No resampling was necessary.** Using `scale_pos_weight` in XGBoost directly encodes the cost asymmetry into the model's objective function, achieving strong recall without the data leakage risk of SMOTE applied before splitting.

- **V1–V28 features are not equally informative.** While all 28 PCA components were trained on, fewer than 6 features (`v14`, `v4`, `v12`, `v10`, `v11`) account for the majority of model decisions.

---

## 💼 Business Impact

| Scenario | Impact |
|----------|--------|
| **Fraud prevention** | With ROC-AUC of 0.9725, the model correctly ranks fraud above legitimate transactions in ~97% of cases, enabling near-real-time blocking of high-risk transactions |
| **Investigation efficiency** | Precision of 0.9383 means fraud analysts investigate primarily real fraud, minimising alert fatigue and wasted resource |
| **Auditability** | SHAP waterfall explanations allow every flagged transaction to be traced to exact feature values, which is essential for regulatory compliance and customer dispute resolution |
| **Cost asymmetry awareness** | Threshold tuning is framed as a business decision: the optimal threshold should be derived from the explicit cost of a missed fraud vs. a false positive investigation |
| **Retraining framework** | Recall on the fraud class should be monitored monthly in production, with retraining triggered if it drops below 0.85 as fraud patterns drift |

> A two-stage alert system is recommended: **auto-block** when `v14 < -5` AND `v4 > 5` co-occur; **manual review queue** for either condition alone or extreme `v12`/`v10` values.

---

## 🛠️ Tools & Technologies

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square"/>
  <img src="https://img.shields.io/badge/SHAP-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Joblib-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
</p>

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Data manipulation** | Pandas, NumPy |
| **Machine learning** | Scikit-learn (Logistic Regression, Decision Tree, Random Forest), XGBoost |
| **Explainability** | SHAP (TreeExplainer, beeswarm, waterfall, dependence plots) |
| **Visualisation** | Matplotlib, Seaborn |
| **Deployment** | Streamlit, Streamlit Community Cloud |
| **Model serialisation** | Joblib |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── 📓 Credit_Card_Fraud_Detection.ipynb   # Full pipeline notebook
│
├── 📊 plots/
│   ├── feature_importance_xgb.png         # XGBoost built-in importance (Weight, Gain, Cover)
│   ├── shap_bar.png                        # SHAP global importance bar chart
│   ├── shap_beeswarm.png                   # SHAP beeswarm (direction & magnitude)
│   ├── shap_waterfall_fraud.png            # Waterfall plot for a confirmed fraud case
│   ├── shap_waterfall_legit.png            # Waterfall plot for a confirmed legitimate case
│   └── shap_dependence.png                 # Dependence plots for v14 and v4
│
├── 🌐 app.py                               # Streamlit web application
├── 🤖 fraud_pipeline.pkl                   # Trained XGBoost pipeline (Scaler + Model)
├── 📄 requirements.txt                     # Python dependencies
├── 📄 README.md                            # This file
└── 📄 LICENSE                              # MIT License
```

> **Note:** `creditcard.csv` is not included in this repository due to file size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory before running the notebook.

---

## ▶️ How to Run the Project

### Option A — Use the Live App (No setup required)

👉 **[Open FraudShield on Streamlit](https://credit-card-fraud-detection-mxpbol3zjbrgzjamc3gj4q.streamlit.app)**

### Option B — Run Locally

#### 1. Clone the repository

```bash
git clone https://github.com/profpius/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

#### 4. Or run the full notebook

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), place it in the root folder, then:

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

Use **Kernel → Restart & Run All** to execute the complete pipeline end-to-end.

> ⏱️ **Estimated runtime:** ~3–7 minutes depending on hardware (SHAP value computation is the most expensive step).

---

## 🚀 Future Improvements

- **[✅] Streamlit deployment** - Live app deployed on Streamlit Community Cloud
- **[ ] Cost-sensitive threshold optimisation** - Replace the fixed threshold with a mathematically optimal value derived from explicit false negative / false positive cost estimates
- **[ ] LightGBM / CatBoost comparison** - Benchmark against faster gradient boosting alternatives for latency-sensitive production environments
- **[ ] Concept drift monitoring** - Implement a monitoring layer that tracks fraud recall over rolling windows and triggers automated retraining when performance degrades
- **[ ] Raw feature access** - If non-anonymised transaction data becomes available, retrain without PCA to enable direct business interpretation of model decisions
- **[ ] Dockerised deployment** - Containerise the scoring API for consistent, reproducible production deployments

---

## 👤 Author

**Victor Pius**

*Data Scientist | Machine Learning Engineer*

Production Engineering graduate transitioning into data science, with hands-on experience building end-to-end ML pipelines across fraud detection, healthcare risk prediction, and customer churn modelling.

<p>
  <a href="https://www.linkedin.com/in/victor-pius-4061a9332">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/profpius">
    <img src="https://img.shields.io/badge/GitHub-Portfolio-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  <a href="https://credit-card-fraud-detection-mxpbol3zjbrgzjamc3gj4q.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-FraudShield-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
</p>

---

<p align="center">
  <i>If this project was useful to you, consider giving it a ⭐ to help others find it.</i>
</p>
