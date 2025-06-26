import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime

# Streamlit page config
st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")
st.title("📡 Federated Learning Dashboard")

# Supabase connection URI (Session Pooler - IPv4 compatible)
SUPABASE_DB_URL = "postgresql://postgres.xjvteeaqbttimgcyjzyu:shodhanAdmin123@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

# Load data function
def load_data():
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        df = pd.read_sql("SELECT * FROM training_logs ORDER BY timestamp ASC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Sidebar: manual refresh and timestamp
st.sidebar.markdown("### 🔁 Refresh Options")
if st.sidebar.button("🔄 Refresh Now"):
    st.experimental_rerun()

last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"🕒 Last updated at: `{last_updated}`")

# Load training logs from Supabase
df_all = load_data()

# Sidebar: select client
client_ids = ["All Clients"]
if not df_all.empty and "client_id" in df_all.columns:
    client_ids += sorted(df_all["client_id"].dropna().unique())
selected_client = st.sidebar.selectbox("🔍 Select Kiosk (Client ID):", client_ids)

# Filter by selected client
if selected_client != "All Clients":
    df = df_all[df_all["client_id"] == selected_client]
else:
    df = df_all

if df.empty:
    st.warning("No training data available.")
    st.stop()

# Kiosk health indicator
def get_status_color(loss):
    if loss < 0.3:
        return "🟢 Healthy"
    elif loss < 0.6:
        return "🟡 Degrading"
    else:
        return "🔴 Alert"

latest = df.iloc[-1]
status = get_status_color(latest["val_loss"])
st.metric(label="📟 Kiosk Status", value=status)
st.markdown("---")

# Training Progress
st.subheader("📈 Training Progress")
st.line_chart(df[["train_loss", "val_loss"]])

# Accuracy Monitoring
st.subheader("🎯 Validation Accuracy")
st.line_chart(df["val_accuracy"])

# Data Table & Export
# Data Table & Export
st.subheader("📄 Training Log")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Export CSV", data=csv, file_name="training_logs.csv", mime="text/csv")
