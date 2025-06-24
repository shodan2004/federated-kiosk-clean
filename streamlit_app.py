import streamlit as st
import pandas as pd
import os

LOG_FILE = "training_logs.csv"

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")
st.title("📡 Federated Learning Dashboard")

# Load CSV logs
def load_logs():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["round", "train_loss", "val_loss", "val_accuracy"])
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=["round", "train_loss", "val_loss", "val_accuracy"])

# Initialize session state
if "training" not in st.session_state:
    st.session_state.training = False

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Training"):
        st.session_state.training = True
        st.success("Training started!")

with col2:
    if st.button("⏹ Stop Training"):
        st.session_state.training = False
        st.warning("Training stopped.")

with col3:
    if st.button("🔄 Refresh Logs"):
        st.rerun()

df = load_logs()

if df.empty or "round" not in df.columns:
    st.info("No valid training logs available yet.")
else:
    # Clean up invalid rows
    df.dropna(subset=["round"], inplace=True)

    st.subheader("📈 Training Progress")
    try:
        st.line_chart(df.set_index("round")[["val_loss", "val_accuracy"]])
    except KeyError:
        st.warning("Missing columns for chart: 'val_loss' or 'val_accuracy'")

    latest = df.tail(1)
    if not latest.empty:
        val_loss = latest["val_loss"].values[0] if "val_loss" in df.columns else None
        val_acc = latest["val_accuracy"].values[0] if "val_accuracy" in df.columns else None

        if val_loss is not None:
            st.metric("📉 Latest Validation Loss", f"{val_loss:.4f}")
        if val_acc is not None:
            st.metric("🎯 Latest Validation Accuracy", f"{val_acc * 100:.2f}%")

    st.subheader("📋 Log Table")
    st.dataframe(df)

if st.session_state.training and not df.empty:
    round_num = int(df["round"].max())
    st.info(f"🔁 Training in progress... Round {round_num}")
elif not st.session_state.training:
    st.info("⏸ Training is paused. You can start it above.")

st.caption("🔁 Refresh the page or use 'Refresh Logs' to update charts.")
