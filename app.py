# app.py
import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "attendance.xlsx")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")

# ---------- Config ----------
N_CLUSTERS = 4
RANDOM_STATE = 42

# ---------- Column names ----------
COL_FAKEID = "Fake ID"
COL_ACCOUNT = "Account code"
COL_IN = "Avg. In Time"
COL_OUT = "Avg. Out Time"
COL_OFFICE = "Avg. Office hrs"
COL_BAY = "Avg. Bay hrs"
COL_BREAK = "Avg. Break hrs"
COL_CAFE = "Avg. Cafeteria hrs"
COL_OOO = "Avg. OOO hrs"
COL_UNBILLED = "Unbilled"
COL_HALF = "Half-Day leave"
COL_FULL = "Full-Day leave"
COL_ONLINE = "Online Check-in"
COL_EXEMPT = "Excemptions"
COL_UNALLOC = "Unallocated"

# ---------- Flask app ----------
app = Flask(__name__)
os.makedirs(CHART_DIR, exist_ok=True)

# ---------- Utility functions ----------
def parse_time_or_duration(x):
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, (pd.Timestamp, pd.DatetimeTZDtype, pd.DatetimeIndex.__class__)):
            return x.hour + x.minute / 60.0 + x.second / 3600.0
    except Exception:
        pass
    try:
        td = pd.to_timedelta(x)
        return td.total_seconds() / 3600.0
    except Exception:
        pass
    try:
        t = pd.to_datetime(x).time()
        return t.hour + t.minute / 60.0 + t.second / 3600.0
    except Exception:
        return np.nan

def safe_mean(series):
    return float(series.dropna().mean()) if len(series.dropna()) > 0 else np.nan

# ---------- Load & preprocess dataset ----------
def load_and_prep(path=DATA_FILE):
    df = pd.read_excel(path)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in [COL_IN, COL_OUT]:
        if col in df.columns:
            df[col + "_hr"] = df[col].apply(parse_time_or_duration)
    for col in [COL_OFFICE, COL_BAY, COL_BREAK, COL_CAFE, COL_OOO]:
        if col in df.columns:
            df[col + "_hrs_float"] = df[col].apply(parse_time_or_duration)
    for col in [COL_HALF, COL_FULL, COL_ONLINE]:
        if col in df.columns:
            df[col + "_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["unallocated_flag"] = df[COL_UNALLOC].astype(str).str.lower().isin(["yes","true","1"]).astype(int) if COL_UNALLOC in df.columns else 0
    df["exempt_flag"] = df[COL_EXEMPT].astype(str).str.lower().isin(["yes","true","1"]).astype(int) if COL_EXEMPT in df.columns else 0

    df["break_bay_ratio"] = df.apply(lambda r: (r.get("Avg. Break hrs_hrs_float", np.nan)/max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. Break hrs_hrs_float", np.nan)) else np.nan, axis=1)
    df["ooo_bay_ratio"] = df.apply(lambda r: (r.get("Avg. OOO hrs_hrs_float", np.nan)/max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. OOO hrs_hrs_float", np.nan)) else np.nan, axis=1)

    numeric_cols = [c for c in df.columns if c.endswith("_hrs_float") or c.endswith("_hr") or c.endswith("_num") or c in ["unallocated_flag","exempt_flag","break_bay_ratio","ooo_bay_ratio"]]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median(skipna=True))

    return df

# Load data
df = load_and_prep()

# ---------- Features for modeling ----------
FEATURES = [c for c in ["Avg. Bay hrs_hrs_float","Avg. Break hrs_hrs_float","Avg. OOO hrs_hrs_float","Avg. Cafeteria hrs_hrs_float","Avg. Office hrs_hrs_float","break_bay_ratio","ooo_bay_ratio","Half-Day leave_num","Full-Day leave_num"] if c in df.columns]
if len(FEATURES) < 3:
    FEATURES = [c for c in df.columns if ("_hrs_float" in c or "_hr" in c or "_num" in c)][:6]

# Standardize and run clustering
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES].values)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
kmeans.fit(X)
df["cluster"] = kmeans.predict(X)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_profiles = pd.DataFrame(cluster_centers, columns=FEATURES, index=range(N_CLUSTERS))

# Anomaly detection
iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
iso.fit(X)
df["anomaly_score"] = iso.decision_function(X)
df["anomaly"] = iso.predict(X)
df["anomaly_flag"] = df["anomaly"].apply(lambda v: 1 if v == -1 else 0)

# Helper to get employee row
def get_employee_row(emp_id):
    try:
        emp_id_int = int(emp_id)
    except ValueError:
        return None
    df_ids = pd.to_numeric(df[COL_FAKEID], errors="coerce")
    subset = df[df_ids == emp_id_int]
    return subset.iloc[0] if not subset.empty else None

# ---------- Chart helpers ----------
def save_bar_chart(values, labels, title, filename):
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=values)
    plt.title(title)
    plt.ylabel("Hours / Score")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path

def save_account_distribution_chart(account_df, metric_col, title, filename):
    plt.figure(figsize=(6,4))
    sns.histplot(account_df[metric_col].dropna(), kde=False, bins=12)
    plt.title(title)
    plt.xlabel(metric_col)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/employee", methods=["GET","POST"])
def employee():
    if request.method=="POST":
        emp_id = request.form.get("emp_id","").strip()
        if emp_id == "":
            return render_template("employee.html", error="Please enter an Employee ID.")
        row = get_employee_row(emp_id)
        if row is None:
            return render_template("employee.html", emp_id=emp_id, error="Employee ID not found in dataset.")

        labels = ["Bay Hrs","Break Hrs","OOO Hrs","Cafeteria","Office Hrs"]
        values = [row.get("Avg. Bay hrs_hrs_float",0), row.get("Avg. Break hrs_hrs_float",0), row.get("Avg. OOO hrs_hrs_float",0), row.get("Avg. Cafeteria hrs_hrs_float",0), row.get("Avg. Office hrs_hrs_float",0)]
        chart_fn = f"{emp_id}_employee_bar.png"
        chart_path = save_bar_chart(values, labels, f"Work Pattern: {emp_id}", chart_fn)
        # employee story generation remains same
        # ... your existing story & recommendation code here ...

        return render_template("employee.html", emp_id=emp_id, chart=chart_path)
    return render_template("employee.html")

@app.route("/account", methods=["GET","POST"])
def account():
    if request.method=="POST":
        acc_code = request.form.get("acc_code","").strip()
        account_df = df[df[COL_ACCOUNT]==acc_code]
        if account_df.empty:
            return render_template("account.html", acc_code=acc_code, error="Account not found or no employees for this account.")
        avg_bay = account_df["Avg. Bay hrs_hrs_float"].mean()
        avg_break = account_df["Avg. Break hrs_hrs_float"].mean()
        avg_ooo = account_df["Avg. OOO hrs_hrs_float"].mean()
        avg_cafe = account_df["Avg. Cafeteria hrs_hrs_float"].mean()
        labels = ["Avg Bay","Avg Break","Avg OOO","Avg Cafe"]
        values = [avg_bay,avg_break,avg_ooo,avg_cafe]
        chart_fn = f"{acc_code}_account_bar.png"
        chart_path = save_bar_chart(values, labels, f"Account Work Pattern: {acc_code}", chart_fn)
        # account story remains same
        return render_template("account.html", acc_code=acc_code, chart=chart_path)
    return render_template("account.html")

# ---------- Run ----------
if __name__=="__main__":
    print("Starting Attendance Insights App...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
