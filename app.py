import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")

# load the saved model, scaler and column order
model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")

# sidebar: user chooses the fraud threshold
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.01)
st.sidebar.write("Lower threshold = catch more fraud, but more false alarms.")

# choose the data
source = st.radio("Data", ["Demo transactions", "Upload CSV"], horizontal=True)

if source == "Demo transactions":
    data = pd.read_csv("model/demo_transactions.csv")
else:
    file = st.file_uploader("Upload a CSV with Time, V1 to V28 and Amount", type="csv")
    if file is None:
        st.stop()
    data = pd.read_csv(file)

# check that all needed columns are there
missing = [c for c in columns if c not in data.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

# scale Time and Amount the same way as training, then predict
X = data[columns].copy()
X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
data["Fraud probability"] = model.predict_proba(X)[:, 1].round(4)
data["Flagged"] = data["Fraud probability"] >= threshold

# summary numbers
col1, col2, col3 = st.columns(3)
col1.metric("Transactions", len(data))
col2.metric("Flagged as fraud", int(data["Flagged"].sum()))
col3.metric("Threshold", threshold)

# if the file has the real labels, show how well the model did
if "Class" in data.columns:
    caught = int((data["Flagged"] & (data["Class"] == 1)).sum())
    missed = int((~data["Flagged"] & (data["Class"] == 1)).sum())
    false_alarms = int((data["Flagged"] & (data["Class"] == 0)).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Frauds caught", caught)
    c2.metric("Frauds missed", missed)
    c3.metric("False alarms", false_alarms)

# show the table, highest fraud probability first
st.subheader("Transactions (highest risk first)")
show = ["Time", "Amount", "Fraud probability", "Flagged"]
if "Class" in data.columns:
    show.append("Class")
st.dataframe(data[show].sort_values("Fraud probability", ascending=False),
             use_container_width=True)