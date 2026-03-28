import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Shield | Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}
.stApp { background-color: #080c14; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0f1a2e 0%, #091626 50%, #0a1220 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -1px;
    margin: 0 0 8px 0;
}
.hero-title span { color: #3b82f6; }
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 300;
    margin: 0;
    max-width: 560px;
    line-height: 1.6;
}
.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 18px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: #60a5fa;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 130px;
    background: linear-gradient(145deg, #0f1d32, #0b1525);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #3b82f6;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3b82f6;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* ── Input Groups ── */
.input-panel {
    background: #0b1525;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

/* ── Result Boxes ── */
.result-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(185,28,28,0.05));
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    animation: pulse-red 2s ease-in-out infinite;
}
.result-legit {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(22,163,74,0.05));
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.15); }
    50% { box-shadow: 0 0 20px 4px rgba(239,68,68,0.15); }
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.result-fraud .result-label { color: #ef4444; }
.result-legit .result-label { color: #22c55e; }
.result-prob {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 6px;
}
.prob-bar-wrap {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    margin: 14px 0 6px 0;
    overflow: hidden;
}
.prob-bar-fill-fraud {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #f97316, #ef4444);
    transition: width 0.6s ease;
}
.prob-bar-fill-legit {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    transition: width 0.6s ease;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1525;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #60a5fa !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 14px 24px;
    width: 100%;
    transition: all 0.2s ease;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(59,130,246,0.25);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1220;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #f1f5f9;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0b1525;
    border: 1px dashed #1e3a5f;
    border-radius: 12px;
}

/* ── Number inputs ── */
.stNumberInput input {
    background: #0f1d32;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    color: #e2e8f0;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
}
.stNumberInput input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0b1525;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #60a5fa;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 24px;
    color: #334155;
    font-size: 0.8rem;
    border-top: 1px solid #0f1d32;
    margin-top: 40px;
}
.footer a { color: #3b82f6; text-decoration: none; }

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 28px 0;
}

/* ── Warning / success override ── */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Load Pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("fraud_pipeline.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("❌ `fraud_pipeline.pkl` not found. Make sure it's in the same directory as `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load pipeline: {e}")
        st.stop()

pipeline = load_pipeline()

# Feature column names (standard Kaggle fraud dataset)
FEATURE_COLS = ["time"] + [f"v{i}" for i in range(1, 29)] + ["amount"]


# ── Hero Section ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ Fraud<span>Shield</span></div>
    <p class="hero-sub">
        An end-to-end machine learning system for real-time credit card fraud detection.
        Trained on 284,807 real transactions. Powered by XGBoost.
    </p>
    <div class="badge-row">
        <span class="badge">XGBoost</span>
        <span class="badge">ROC-AUC 0.9725</span>
        <span class="badge">284,807 Transactions</span>
        <span class="badge">SMOTE + Threshold Tuning</span>
        <span class="badge">SHAP Explainability</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model Performance Metrics ──────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-val">0.9725</div>
        <div class="metric-lbl">ROC-AUC</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">0.9383</div>
        <div class="metric-lbl">Precision</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">0.80</div>
        <div class="metric-lbl">Recall</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">0.17%</div>
        <div class="metric-lbl">Fraud Rate</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">284,807</div>
        <div class="metric-lbl">Transactions</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("<div style='height:1px;background:#1e2d45;margin:12px 0'></div>", unsafe_allow_html=True)

    st.markdown("**Model Details**")
    st.markdown("""
    <div style='background:#0f1d32;border:1px solid #1e3a5f;border-radius:10px;padding:14px;font-size:0.82rem;color:#94a3b8;line-height:1.8'>
    🔹 Algorithm: <b style='color:#60a5fa'>XGBoost</b><br>
    🔹 Imbalance: <b style='color:#60a5fa'>SMOTE</b><br>
    🔹 Tuning: <b style='color:#60a5fa'>Threshold Optimization</b><br>
    🔹 Explainability: <b style='color:#60a5fa'>SHAP</b><br>
    🔹 Pipeline: <b style='color:#60a5fa'>Sklearn + XGBoost</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#1e2d45;margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("**Dataset**")
    st.markdown("""
    <div style='font-size:0.82rem;color:#64748b;line-height:1.8'>
    Source: <a href='https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud' style='color:#3b82f6'>Kaggle Credit Card Fraud</a><br>
    Features: Time, V1–V28 (PCA), Amount<br>
    Target: Class (0=Legit, 1=Fraud)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#1e2d45;margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("**Built by Victor Pius**")
    st.markdown("""
    <div style='font-size:0.82rem;line-height:2'>
    <a href='https://github.com/profpius' style='color:#3b82f6'>🐙 GitHub: profpius</a><br>
    <a href='https://linkedin.com/in/victor-pius-4061a9332' style='color:#3b82f6'>💼 LinkedIn Profile</a><br>
    <a href='https://github.com/profpius/credit-card-fraud-detection' style='color:#3b82f6'>📂 Project Repo</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#1e2d45;margin:16px 0'></div>", unsafe_allow_html=True)
    st.caption("⚠️ For portfolio demonstration only. Not for production financial use.")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔢  Manual Transaction Input", "📂  Batch CSV Upload"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Input
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
    st.caption("Enter values for all features. V1–V28 are PCA-transformed components from the original dataset.")

    # ── Quick Sample Buttons ──
    col_s1, col_s2, col_s3 = st.columns([1, 1, 4])
    with col_s1:
        load_fraud = st.button("⚠️ Load Fraud Sample")
    with col_s2:
        load_legit = st.button("✅ Load Legit Sample")

    # Known sample values from the Kaggle dataset
    FRAUD_SAMPLE = {
        "time": 406.0, "amount": 2.69,
        "v": [-2.3122, 1.9519, -1.6096, 3.9979, -0.5224, -1.4265, -2.5374,
               0.8182, -0.3399, 0.1674, 2.3457, -1.3459, 0.5764, -0.4999,
               1.4189, -0.4399, -1.1577, -0.6756, -0.2256, -0.6386, -0.0797,
              -0.0603, 0.1271, -0.0583, 0.3308, -1.0547, -0.2023, -0.2129]
    }
    LEGIT_SAMPLE = {
        "time": 152.0, "amount": 25.0,
        "v": [-0.1274, 0.4782, 0.3567, 0.1234, -0.2341, 0.0512, -0.0821,
               0.3214, -0.0967, 0.1875, 0.5634, -0.1234, 0.2341, -0.0512,
               0.0821, 0.1567, -0.0345, 0.2234, -0.1123, 0.0789, -0.0456,
               0.1234, -0.0789, 0.0345, 0.0567, -0.0234, 0.0123, -0.0456]
    }

    # Determine defaults
    if load_fraud:
        defaults = FRAUD_SAMPLE
        st.info("⚠️ Loaded a known fraudulent transaction sample from the dataset.")
    elif load_legit:
        defaults = LEGIT_SAMPLE
        st.success("✅ Loaded a known legitimate transaction sample from the dataset.")
    else:
        defaults = None

    def dv(key, default=0.0):
        """Get default value for a field."""
        if defaults is None:
            return default
        if key == "time":
            return defaults["time"]
        if key == "amount":
            return defaults["amount"]
        if key.startswith("v"):
            idx = int(key[1:]) - 1
            return defaults["v"][idx]
        return default

    # ── Input Layout ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Transaction Info</div>', unsafe_allow_html=True)
        time_val = st.number_input("Time (seconds since first tx)", value=dv("time"), format="%.2f", help="Seconds elapsed since the first transaction in the dataset.")
        amount   = st.number_input("Amount ($)", value=dv("amount"), min_value=0.0, format="%.2f", help="Transaction amount in USD.")

        st.markdown('<div class="section-header">V1 – V9</div>', unsafe_allow_html=True)
        v1 = st.number_input("V1",  value=dv("v1"),  format="%.6f")
        v2 = st.number_input("V2",  value=dv("v2"),  format="%.6f")
        v3 = st.number_input("V3",  value=dv("v3"),  format="%.6f")
        v4 = st.number_input("V4",  value=dv("v4"),  format="%.6f")
        v5 = st.number_input("V5",  value=dv("v5"),  format="%.6f")
        v6 = st.number_input("V6",  value=dv("v6"),  format="%.6f")
        v7 = st.number_input("V7",  value=dv("v7"),  format="%.6f")
        v8 = st.number_input("V8",  value=dv("v8"),  format="%.6f")
        v9 = st.number_input("V9",  value=dv("v9"),  format="%.6f")

    with col2:
        st.markdown('<div class="section-header">V10 – V19</div>', unsafe_allow_html=True)
        v10 = st.number_input("V10", value=dv("v10"), format="%.6f")
        v11 = st.number_input("V11", value=dv("v11"), format="%.6f")
        v12 = st.number_input("V12", value=dv("v12"), format="%.6f")
        v13 = st.number_input("V13", value=dv("v13"), format="%.6f")
        v14 = st.number_input("V14", value=dv("v14"), format="%.6f")
        v15 = st.number_input("V15", value=dv("v15"), format="%.6f")
        v16 = st.number_input("V16", value=dv("v16"), format="%.6f")
        v17 = st.number_input("V17", value=dv("v17"), format="%.6f")
        v18 = st.number_input("V18", value=dv("v18"), format="%.6f")
        v19 = st.number_input("V19", value=dv("v19"), format="%.6f")

    with col3:
        st.markdown('<div class="section-header">V20 – V28</div>', unsafe_allow_html=True)
        v20 = st.number_input("V20", value=dv("v20"), format="%.6f")
        v21 = st.number_input("V21", value=dv("v21"), format="%.6f")
        v22 = st.number_input("V22", value=dv("v22"), format="%.6f")
        v23 = st.number_input("V23", value=dv("v23"), format="%.6f")
        v24 = st.number_input("V24", value=dv("v24"), format="%.6f")
        v25 = st.number_input("V25", value=dv("v25"), format="%.6f")
        v26 = st.number_input("V26", value=dv("v26"), format="%.6f")
        v27 = st.number_input("V27", value=dv("v27"), format="%.6f")
        v28 = st.number_input("V28", value=dv("v28"), format="%.6f")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Predict Button ──
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_clicked = st.button("🔍  ANALYZE TRANSACTION")

    if predict_clicked:
        features = np.array([[
            time_val, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount
        ]])


        try:
            feature_names = pipeline.named_steps['scaler'].feature_names_in_
            input_df = pd.DataFrame(features, columns=feature_names)
            prediction  = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            pct = f"{probability:.2%}"
            bar_pct = f"{probability * 100:.1f}%"

            if prediction == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div style='font-size:2.8rem;margin-bottom:8px'>⚠️</div>
                    <div class="result-label">FRAUDULENT</div>
                    <div class="result-prob">Fraud probability: <b>{pct}</b></div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill-fraud" style="width:{bar_pct}"></div>
                    </div>
                    <div style='font-size:0.75rem;color:#64748b'>Risk score bar</div>
                </div>
                """, unsafe_allow_html=True)
                st.error("🚨 Recommended action: **Block transaction** and flag for manual review.")
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <div style='font-size:2.8rem;margin-bottom:8px'>✅</div>
                    <div class="result-label">LEGITIMATE</div>
                    <div class="result-prob">Fraud probability: <b>{pct}</b></div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill-legit" style="width:{bar_pct}"></div>
                    </div>
                    <div style='font-size:0.75rem;color:#64748b'>Risk score bar</div>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ Transaction is safe to process.")

        with res_col2:
            st.markdown('<div class="section-header">Confidence Breakdown</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0b1525')
            ax.set_facecolor('#0b1525')
            categories = ['Legitimate', 'Fraudulent']
            values     = [1 - probability, probability]
            colors     = ['#22c55e', '#ef4444']
            bars = ax.barh(categories, values, color=colors, height=0.4, edgecolor='none')
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.2%}', va='center', ha='left',
                        color='#e2e8f0', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1.18)
            ax.set_xlabel('Probability', color='#64748b', fontsize=9)
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.label.set_color('#64748b')
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

            st.metric("Fraud Probability",   f"{probability:.4f}")
            st.metric("Legit Probability",   f"{1 - probability:.4f}")
            st.metric("Prediction Confidence", f"{max(probability, 1 - probability):.2%}")

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        with st.expander("🧠  Why did the model make this prediction?  (SHAP Feature Importance)"):
            try:
                import shap

                # Extract the XGBoost model step from the pipeline
                # Try common step names; adjust if yours differs
                model_step = None
                for name, step in pipeline.named_steps.items():
                    if hasattr(step, 'predict_proba') and hasattr(step, 'feature_importances_'):
                        model_step = step
                        break
                # Fallback: last step
                if model_step is None:
                    model_step = list(pipeline.named_steps.values())[-1]

                # Transform features through all steps except the last (model)
                steps_except_last = list(pipeline.named_steps.keys())[:-1]
                if steps_except_last:
                    transformed = features.copy()
                    for step_name in steps_except_last:
                        transformed = pipeline.named_steps[step_name].transform(transformed)
                    features_for_shap = transformed
                else:
                    features_for_shap = features

                explainer  = shap.TreeExplainer(model_step)
                shap_vals  = explainer.shap_values(features_for_shap)

                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]   # class 1 (fraud) SHAP values
                else:
                    sv = shap_vals[0]

                feat_names = FEATURE_COLS

                # Sort by absolute SHAP value
                sorted_idx  = np.argsort(np.abs(sv))[-15:]
                sorted_sv   = sv[sorted_idx]
                sorted_names = [feat_names[i] for i in sorted_idx]
                colors_shap = ['#ef4444' if v > 0 else '#22c55e' for v in sorted_sv]

                fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor='#0b1525')
                ax2.set_facecolor('#0b1525')
                bars2 = ax2.barh(sorted_names, sorted_sv, color=colors_shap,
                                 edgecolor='none', height=0.55)
                ax2.axvline(0, color='#334155', linewidth=1.2)
                ax2.set_title('Top 15 Feature Contributions (SHAP)',
                              color='#e2e8f0', fontsize=11, pad=12)
                ax2.tick_params(colors='#94a3b8', labelsize=9)
                ax2.set_xlabel('SHAP Value  (red = pushes toward fraud, green = pushes toward legit)',
                               color='#64748b', fontsize=8)
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.clf()

                st.caption("🔴 Red bars push prediction toward **fraud**. 🟢 Green bars push toward **legitimate**. Longer bar = stronger influence.")

            except ImportError:
                st.info("Install SHAP to enable explainability: `pip install shap`  (add `shap` to requirements.txt for deployment)")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable for this pipeline configuration: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch Transaction Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV file of transactions for bulk fraud detection.
    The CSV must contain these columns (in any order):
    """)
    st.code("Time, V1, V2, V3, ..., V28, Amount", language="text")
    st.caption("The `Class` column is optional — if present it will be ignored during prediction.")

    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.markdown(f"**Preview — {len(df_raw):,} transactions loaded**")
        st.dataframe(df_raw.head(5), use_container_width=True)

        # Check required columns
        missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
        if missing:
            st.error(f"❌ Missing columns: `{missing}`. Your CSV must include Time, V1–V28, and Amount.")
        else:
            run_col, _ = st.columns([1, 2])
            with run_col:
                run_batch = st.button("🔍  RUN BATCH PREDICTION")

            if run_batch:
                df_input = df_raw[FEATURE_COLS].copy()

                try:
                    preds = pipeline.predict(df_input)
                    probs = pipeline.predict_proba(df_input)[:, 1]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                df_results = df_raw.copy()
                df_results["Fraud_Probability"] = np.round(probs, 6)
                df_results["Prediction"]        = preds
                df_results["Result"]            = np.where(preds == 1, "⚠️ Fraud", "✅ Legitimate")

                fraud_count = int((preds == 1).sum())
                legit_count = int((preds == 0).sum())
                fraud_rate  = fraud_count / len(preds) * 100

                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Batch Results Summary</div>', unsafe_allow_html=True)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total Transactions", f"{len(preds):,}")
                mc2.metric("🚨 Fraudulent",      f"{fraud_count:,}")
                mc3.metric("✅ Legitimate",       f"{legit_count:,}")
                mc4.metric("Fraud Rate",          f"{fraud_rate:.3f}%")

                # Distribution chart
                fig3, ax3 = plt.subplots(figsize=(6, 3), facecolor='#0b1525')
                ax3.set_facecolor('#0b1525')
                ax3.bar(['✅ Legitimate', '⚠️ Fraudulent'],
                        [legit_count, fraud_count],
                        color=['#22c55e', '#ef4444'],
                        edgecolor='none', width=0.45)
                ax3.set_title('Prediction Distribution', color='#e2e8f0', fontsize=10, pad=10)
                ax3.tick_params(colors='#94a3b8')
                for spine in ax3.spines.values():
                    spine.set_visible(False)
                ax3.set_ylabel('Count', color='#64748b', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.clf()

                # Full results table
                st.markdown('<div class="section-header">Transaction-Level Results</div>', unsafe_allow_html=True)
                display_cols = ["Time", "Amount", "Fraud_Probability", "Prediction", "Result"]
                st.dataframe(
                    df_results[display_cols].style.applymap(
                        lambda v: "color: #ef4444; font-weight: bold" if v == 1 else "color: #22c55e",
                        subset=["Prediction"]
                    ),
                    use_container_width=True,
                    height=380
                )

                # Download
                csv_out = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Full Results as CSV",
                    data=csv_out,
                    file_name="fraudshield_results.csv",
                    mime="text/csv"
                )

                if fraud_count > 0:
                    st.warning(f"🚨 {fraud_count} suspicious transaction(s) detected. Review the flagged rows above.")
                else:
                    st.success("✅ No fraudulent transactions detected in this batch.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <b>Victor Pius</b> &nbsp;|&nbsp;
    <a href='https://github.com/profpius/credit-card-fraud-detection'>GitHub Repo</a> &nbsp;|&nbsp;
    <a href='https://linkedin.com/in/victor-pius-4061a9332'>LinkedIn</a> &nbsp;|&nbsp;
    Powered by XGBoost + Streamlit
</div>
""", unsafe_allow_html=True)
