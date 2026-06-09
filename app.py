import os
import time
import traceback

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =========================
# GROQ SETUP (UPDATED & STABLE)
# =========================

import os
import streamlit as st

try:
    from groq import Groq

    # API KEY (Streamlit Secrets OR Environment Variable)
    GROQ_API_KEY = None

    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

except ImportError:
    groq_client = None


def get_ai_security_advice(attack_type: str, confidence: float) -> str:

    if groq_client is None:
        return """
        <h3>AI Security Copilot Offline</h3>
        <p>Groq API key not configured.</p>
        """

    prompt = f"""
You are a cybersecurity SOC analyst.

Attack Type: {attack_type}
Confidence Score: {confidence}%

IMPORTANT:

Return VALID HTML ONLY.

Do NOT write:
Attack Meaning
Severity Level
Impact

without HTML tags.

Strictly return:

<h3>Detection Result Analysis</h3>

<h4>Attack Meaning</h4>
<p>your answer</p>

<h4>Severity Level</h4>
<p>your answer</p>

<h4>Impact</h4>
<p>your answer</p>

<h4>Mitigation Steps</h4>
<ul>
<li>step 1</li>
<li>step 2</li>
<li>step 3</li>
</ul>

No markdown.
No code block.
No triple backticks.
Only pure HTML.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity SOC analyst. You ONLY output raw HTML. Never use markdown. Never use backticks. Never wrap in code blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=700,
        )

        # ✅ ASSIGN html FIRST, then clean it
        html = response.choices[0].message.content  # ← this line was missing!

        import re
        html = re.sub(r'```[\w]*', '', html)
        html = html.replace('`', '')
        html = html.strip()

        return html

    except Exception as e:
        return f"<p>⚠️ Groq Error: {str(e)}</p>"
# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="NIDS Dashboard", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a0033, #000000);
    color: #ffffff;
}
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    background: linear-gradient(90deg, #a855f7, #ff00cc, #00ffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #c084fc;
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #ec4899);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Network Intrusion Detection System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-Enhanced Real-time Security Analysis</div>',
    unsafe_allow_html=True,
)

# =========================
# MODEL SETUP
# =========================
MODEL_PATH = "model.pkl"

col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


def train_and_save_model():
    """Train an XGBoost model on KDDTrain+.txt and persist it."""
    train_path = "KDDTrain+.txt"
    if not os.path.exists(train_path):
        st.error(
            f"Training file `{train_path}` not found. "
            "Place it in the same directory as app.py."
        )
        st.stop()

    df = pd.read_csv(train_path, header=None)

    X = df.iloc[:, :41].copy()
    y = df.iloc[:, 41].apply(lambda x: 0 if str(x).strip() == "normal" else 1)

    encoders: dict = {}
    for col_idx in [1, 2, 3]:
        le = LabelEncoder()
        X[col_idx] = le.fit_transform(X[col_idx].astype(str))
        encoders[col_idx] = le

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X, y)
    joblib.dump((model, encoders), MODEL_PATH)
    return model, encoders


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    st.warning("Model not found — training now. This may take a minute…")
    model, encoders = train_and_save_model()
    st.success("Model trained and saved.")
    return model, encoders


model, encoders = load_model()

# =========================
# SESSION STATE INIT
# =========================
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "ai_response" not in st.session_state:
    st.session_state.ai_response = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# =========================
# UI — DATASET SELECTION
# =========================
st.subheader("Dataset Selection")

use_demo = st.button("📂 NSL-KDD Demo Dataset")
uploaded_file = st.file_uploader("Upload CSV/TXT Dataset", type=["csv", "txt"])

df_raw = None

# Demo dataset takes priority over upload
if use_demo:
    demo_path = "demo_dataset.csv"
    if not os.path.exists(demo_path):
        st.error(
            f"`{demo_path}` not found. Place it in the same directory as app.py."
        )
    else:
        df_raw = pd.read_csv(demo_path, header=None)
        st.success("✅ Demo dataset loaded.")

elif uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, header=None)
        st.success(f"✅ Uploaded file loaded: `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

# =========================
# VALIDATE & PREVIEW
# =========================
if df_raw is not None:
    n_cols = df_raw.shape[1]

    if n_cols < 41:
        st.error(
            f"Dataset has only {n_cols} columns — at least 41 feature columns are required."
        )
        st.stop()

    df_input = df_raw.iloc[:, :41].copy()
    df_input.columns = col_names

    st.markdown("**Preview (first 5 rows):**")
    st.dataframe(df_input.head())

    run = st.button("▶ Run Detection")

    if run:
        with st.spinner("Analysing network traffic…"):
            try:
                # ── Encode categorical columns ──────────────────────────
                cat_map = {"protocol_type": 1, "service": 2, "flag": 3}

                def safe_encode(val, encoder):
                    s = str(val).strip()
                    return encoder.transform([s])[0] if s in encoder.classes_ else -1

                for col_name, col_idx in cat_map.items():
                    df_input[col_name] = df_input[col_name].apply(
                        lambda x, ci=col_idx: safe_encode(x, encoders[ci])
                    )

                df_numeric = df_input.apply(pd.to_numeric, errors="coerce").fillna(0)

                # ── Predict ─────────────────────────────────────────────
                predictions = model.predict(df_numeric)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df_numeric)[:, 1]
                else:
                    probs = predictions.astype(float)

                df_numeric["Prediction"] = [
                    "Attack" if p == 1 else "Normal" for p in predictions
                ]
                df_numeric["Confidence"] = probs

                # ── Summary stats ────────────────────────────────────────
                total = len(df_numeric)
                attacks = int(sum(predictions))
                normal_count = total - attacks
                attack_ratio = (attacks / total) * 100
                avg_confidence = round(float(probs.mean()) * 100, 2)
                attack_label = "Attack" if attacks > 0 else "Normal"

                st.session_state.results_df = df_numeric
                st.session_state.summary = {
                    "total": total,
                    "attacks": attacks,
                    "normal": normal_count,
                    "attack_ratio": attack_ratio,
                    "avg_confidence": avg_confidence,
                    "attack_label": attack_label,
                }

                # ── Groq AI call ─────────────────────────────────────────
                with st.spinner("Getting AI security analysis…"):
                    st.session_state.ai_response = get_ai_security_advice(
                        attack_label, avg_confidence
                    )

            except Exception as e:
                st.error("An error occurred during detection.")
                st.text(traceback.format_exc())

# =========================
# RESULTS
# =========================
if st.session_state.results_df is not None and st.session_state.summary is not None:
    s = st.session_state.summary
    df_res = st.session_state.results_df

    st.success("✅ Detection Completed")

    st.subheader("Detection Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Connections", s["total"])
    c2.metric("Attacks Detected", s["attacks"])
    c3.metric("Normal Traffic", s["normal"])
    c4.metric("Attack Ratio", f"{s['attack_ratio']:.2f}%")

    # ── Pie chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(facecolor="#0f0c29")
    ax.set_facecolor("#0f0c29")
    wedge_colors = ["#22c55e", "#ef4444"]
    ax.pie(
        [s["normal"], s["attacks"]],
        labels=["Normal", "Attack"],
        autopct="%1.1f%%",
        colors=wedge_colors,
        textprops={"color": "white"},
    )
    st.pyplot(fig)
    risk_level = (
    "🔴 HIGH"
    if s["attack_ratio"] > 30
    else "🟡 MEDIUM"
    if s["attack_ratio"] > 10
    else "🟢 LOW"
    )

    st.subheader("🛡 Threat Assessment")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Threat Score",
        f"{s['avg_confidence']}%"
    )

    c2.metric(
        "Risk Level",
        risk_level
    )

    c3.metric(
        "Attack Ratio",
        f"{s['attack_ratio']:.2f}%"
    )

    # ── AI analysis ───────────────────────────────────────────────────────
    st.markdown("""
<style>

.ai-card{

    background:linear-gradient(
        135deg,
        rgba(88,28,135,0.95),
        rgba(15,23,42,0.98)
    );

    border:1px solid rgba(168,85,247,0.5);

    border-radius:22px;

    padding:28px;

    box-shadow:
        0 0 20px rgba(255,0,204,0.25),
        0 0 40px rgba(168,85,247,0.25),
        0 0 70px rgba(0,255,255,0.12);

    backdrop-filter: blur(12px);
}

.ai-header{

    font-size:34px;
    font-weight:800;

    background:linear-gradient(
        90deg,
        #ff00cc,
        #c084fc,
        #00ffff
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    

    margin-bottom:15px;
}

.ai-badge{
    display:inline-block;

    padding:8px 16px;

    background:rgba(168,85,247,0.25);

    border:1px solid #c084fc;

    border-radius:999px;

    color:white;

    margin-bottom:20px;
}

.ai-content{
    color:#f8fafc;
    line-height:1.9;
    font-size:16px;
}

.ai-card h3{

    font-size:32px;

    text-align:center;

    font-weight:700;

    background: linear-gradient(
        90deg,
        #a855f7,
        #ff00cc,
        #00ffff
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.ai-card h4{

    font-size:22px;

    font-weight:700;

    background: linear-gradient(
        90deg,
        #a855f7,
        #ff00cc,
        #00ffff
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.ai-card ul{
    padding-left:20px;
}

.ai-card li{
    margin-bottom:8px;
}

</style>
""", unsafe_allow_html=True)
    if st.session_state.ai_response:
        ai_content = st.session_state.ai_response.strip()
        
        # Clean any remaining markdown artifacts
        import re
        ai_content = re.sub(r'```[\w]*', '', ai_content)
        ai_content = ai_content.replace('`', '').strip()

        st.markdown(f"""
        <div class="ai-card">
            <div class="ai-header">🛡 AI Security Copilot</div>
            <div class="ai-badge">Powered by Groq AI</div>
            <div class="ai-content">{ai_content}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Suspicious records ────────────────────────────────────────────────
    st.subheader("⚠️ Suspicious Records (Top 20)")
    attack_rows = df_res[df_res["Prediction"] == "Attack"]
    if attack_rows.empty:
        st.info("No attack records found in this dataset.")
    else:
        st.dataframe(attack_rows.head(20))

    # ── Download ──────────────────────────────────────────────────────────
    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Results",
        data=csv_bytes,
        file_name="nids_output.csv",
        mime="text/csv",
    )