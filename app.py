import streamlit as st
import pandas as pd
import joblib
import os
import time
import traceback
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.set_page_config(page_title="NIDS Dashboard", layout="wide")

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 50px;
        font-weight: 700;
        background: linear-gradient(90deg, #8e2de2, #ff6ec4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }

    .subtitle {
        text-align: center;
        font-size: 25px;
        color: #6c757d;
        margin-bottom: 30px;
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time analysis of network traffic</div>', unsafe_allow_html=True)

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
    "dst_host_srv_rerror_rate"
]

def train_and_save_model():
    df = pd.read_csv("KDDTrain+.txt", header=None)

    X = df.iloc[:, :41]
    y = df.iloc[:, 41].apply(lambda x: 0 if x == "normal" else 1)

    encoders = {}
    for col in [1, 2, 3]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X, y)

    joblib.dump((model, encoders), MODEL_PATH)
    return model, encoders


if os.path.exists(MODEL_PATH):
    model, encoders = joblib.load(MODEL_PATH)
else:
    st.warning("Initializing model...")
    model, encoders = train_and_save_model()
    st.success("Model ready")

uploaded_file = st.file_uploader("📂Select CSV/TXT file", type=["csv", "txt"])

status_slot = st.empty()
button_slot = st.empty()

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None)

        if df.shape[1] < 41:
            st.error("Invalid input format. Expected at least 41 columns.")
            st.stop()

        df = df.iloc[:, :41]
        df.columns = col_names

        st.markdown("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        status_slot.info("Preparing detection engine... Please wait")
        time.sleep(1.5)
        status_slot.success("Ready for detection")

        run = button_slot.button("▶ Run Detection")

        if run:
            status_slot.empty()

            with st.spinner("Analyzing network traffic..."):
                def safe_encode(val, encoder):
                    val = str(val).strip()
                    return encoder.transform([val])[0] if val in encoder.classes_ else -1

                mapping = {"protocol_type": 1, "service": 2, "flag": 3}

                for col_name, idx in mapping.items():
                    df[col_name] = df[col_name].apply(lambda x: safe_encode(x, encoders[idx]))

                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

                predictions = model.predict(df)
                probs = model.predict_proba(df)[:, 1]

            df["Prediction"] = ["Attack" if p == 1 else "Normal" for p in predictions]
            df["Confidence"] = probs

            st.success("Detection Completed")

            total = len(df)
            attacks = int(sum(predictions))
            normal = total - attacks
            attack_ratio = (attacks / total) * 100

            st.subheader("Detection Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", total)
            c2.metric("Attacks", attacks)
            c3.metric("Normal", normal)

            st.metric("⚠️ Attack %", f"{attack_ratio:.2f}%")

            if attack_ratio > 30:
                st.error("HIGH RISK DETECTED")
            elif attack_ratio > 10:
                st.warning("Moderate Suspicious Activity")
            else:
                st.success("Traffic Appears Normal")

            st.subheader("Traffic Distribution")

            fig, ax = plt.subplots()
            ax.pie([normal, attacks], labels=["Normal", "Attack"], autopct='%1.1f%%')
            ax.set_title("Traffic Distribution")
            st.pyplot(fig)

            st.subheader("Suspicious Records")
            st.dataframe(df[df["Prediction"] == "Attack"].head(20))

            with st.expander("View Full Dataset"):
                st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "nids_output.csv", "text/csv")

    except Exception as e:
        st.error("Processing error")
        st.text(str(e))
        st.text(traceback.format_exc())