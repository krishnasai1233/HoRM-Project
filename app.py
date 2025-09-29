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
from io import BytesIO
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------- Configuration / expected column names ----------
DATA_FILE = r"C:\Users\krishna.sai\OneDrive - Mu Sigma Business Solutions Pvt. Ltd\Documents\attendance_app\data\attendance.xlsx"

CHART_DIR = "static/charts"
N_CLUSTERS = 4  # can be tuned
RANDOM_STATE = 42

# column names (adjust if your excel has slightly different names)
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

# ---------- App setup ----------
app = Flask(__name__)

if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# ---------- Utility functions for parsing ----------
def parse_time_or_duration(x):
    """
    Accepts values like '08:12:26' (time-of-day) or '12:32:03' as durations.
    Returns float hours where meaningful (time -> hour + minute/60).
    """
    if pd.isna(x):
        return np.nan
    try:
        # If it's a pandas Timestamp or datetime-like, extract hour + minute
        if isinstance(x, (pd.Timestamp, pd.DatetimeTZDtype, pd.DatetimeIndex.__class__)):
            return x.hour + x.minute / 60.0 + x.second / 3600.0
    except Exception:
        pass

    # Try parse as timedelta (duration)
    try:
        td = pd.to_timedelta(x)
        hours = td.total_seconds() / 3600.0
        return hours
    except Exception:
        pass

    # Try parse as time-of-day
    try:
        t = pd.to_datetime(x).time()
        return t.hour + t.minute / 60.0 + t.second / 3600.0
    except Exception:
        return np.nan

def safe_mean(series):
    return float(series.dropna().mean()) if len(series.dropna())>0 else np.nan

# ---------- Load & preprocess dataset ----------
def load_and_prep(path=DATA_FILE):
    df = pd.read_excel(path)
    df = df.copy()

    # Normalize column names trimming spaces
    df.columns = [c.strip() for c in df.columns]

    # Parse numeric/time columns into float hours
    for col in [COL_IN, COL_OUT]:
        if col in df.columns:
            df[col + "_hr"] = df[col].apply(parse_time_or_duration)

    for col in [COL_OFFICE, COL_BAY, COL_BREAK, COL_CAFE, COL_OOO]:
        if col in df.columns:
            df[col + "_hrs_float"] = df[col].apply(parse_time_or_duration)

    # Numeric leave/checkin features
    for col in [COL_HALF, COL_FULL, COL_ONLINE]:
        if col in df.columns:
            df[col + "_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Binarize some columns (Unallocated, Exemptions) if present
    if COL_UNALLOC in df.columns:
        df["unallocated_flag"] = df[COL_UNALLOC].astype(str).str.lower().isin(["yes", "true", "1"]).astype(int)
    else:
        df["unallocated_flag"] = 0

    if COL_EXEMPT in df.columns:
        df["exempt_flag"] = df[COL_EXEMPT].astype(str).str.lower().isin(["yes", "true", "1"]).astype(int)
    else:
        df["exempt_flag"] = 0

    # Derived features
    # Break/Bay ratio and OOO/Bay ratio
    df["break_bay_ratio"] = df.apply(lambda r: (r.get("Avg. Break hrs_hrs_float", np.nan) / max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. Break hrs_hrs_float", np.nan)) else np.nan, axis=1)
    df["ooo_bay_ratio"] = df.apply(lambda r: (r.get("Avg. OOO hrs_hrs_float", np.nan) / max(0.001, r.get("Avg. Bay hrs_hrs_float", np.nan))) if not pd.isna(r.get("Avg. OOO hrs_hrs_float", np.nan)) else np.nan, axis=1)

    # Fill NAs with medians for modeling
    numeric_cols = [c for c in df.columns if c.endswith("_hrs_float") or c.endswith("_hr") or c.endswith("_num") or c in ["unallocated_flag", "exempt_flag", "break_bay_ratio", "ooo_bay_ratio"]]
    for c in numeric_cols:
        if c in df.columns:
            median = df[c].median(skipna=True)
            df[c] = df[c].fillna(median)

    return df

# Load data
df = load_and_prep()

# ---------- Features for modeling ----------
FEATURES = []
for c in ["Avg. Bay hrs_hrs_float", "Avg. Break hrs_hrs_float", "Avg. OOO hrs_hrs_float", "Avg. Cafeteria hrs_hrs_float",
          "Avg. Office hrs_hrs_float", "break_bay_ratio", "ooo_bay_ratio", "Half-Day leave_num", "Full-Day leave_num"]:
    if c in df.columns:
        FEATURES.append(c)

# Safety: if not enough features, fallback to some alternatives
if len(FEATURES) < 3:
    # take any numeric processed columns
    FEATURES = [c for c in df.columns if ("_hrs_float" in c or "_hr" in c or "_num" in c)][:6]

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES].values)

# ---------- Clustering model ----------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
kmeans.fit(X)
df["cluster"] = kmeans.predict(X)

# Cluster profiles (centers in original feature space)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_profiles = pd.DataFrame(cluster_centers, columns=FEATURES)
cluster_profiles.index.name = "cluster"

# ---------- Anomaly detector ----------
iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
iso.fit(X)
df["anomaly_score"] = iso.decision_function(X)
df["anomaly"] = iso.predict(X)  # -1 for anomaly, 1 for normal
df["anomaly_flag"] = df["anomaly"].apply(lambda v: 1 if v == -1 else 0)

# Helper to get row by Fake ID
def get_employee_row(emp_id):
    try:
        # Convert input to integer
        emp_id_int = int(emp_id)
    except ValueError:
        return None  # invalid input, e.g. text entered

    # Ensure Fake ID column is numeric
    df_ids = pd.to_numeric(df[COL_FAKEID], errors="coerce")

    subset = df[df_ids == emp_id_int]
    if subset.empty:
        return None
    return subset.iloc[0]


# ---------- Charting helpers ----------
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

# ---------- Story generation helpers ----------
def generate_employee_story(emp_row):
    """
    Generate insight story and single recommendation for an employee row.
    Uses cluster profile, z-scores vs company mean, and anomaly flags.
    """
    emp_id = emp_row[COL_FAKEID]
    cluster = int(emp_row["cluster"])
    anom = bool(emp_row["anomaly_flag"] == 1)

    # basic numbers
    bay = emp_row.get("Avg. Bay hrs_hrs_float", np.nan)
    brk = emp_row.get("Avg. Break hrs_hrs_float", np.nan)
    ooo = emp_row.get("Avg. OOO hrs_hrs_float", np.nan)
    cafe = emp_row.get("Avg. Cafeteria hrs_hrs_float", np.nan)
    office = emp_row.get("Avg. Office hrs_hrs_float", np.nan)
    half = emp_row.get("Half-Day leave_num", 0)
    full = emp_row.get("Full-Day leave_num", 0)

    # z-scores relative to company
    z = {}
    for col in ["Avg. Bay hrs_hrs_float", "Avg. Break hrs_hrs_float", "Avg. OOO hrs_hrs_float", "Avg. Cafeteria hrs_hrs_float", "Avg. Office hrs_hrs_float"]:
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std(ddof=0) if df[col].std(ddof=0) != 0 else 1.0
            z[col] = (emp_row.get(col, mu) - mu) / sigma

    # Compose lines
    lines = []
    lines.append(f"Employee {emp_id} belongs to cluster #{cluster} (cluster summary: {cluster_profiles.loc[cluster].round(2).to_dict()}).")

    # Interpret z-scores (human readable)
    if z.get("Avg. Break hrs_hrs_float", 0) > 1:
        lines.append(f"Break time is significantly higher than company average ({brk:.1f} hrs).")
    elif z.get("Avg. Break hrs_hrs_float", 0) < -1:
        lines.append(f"Break time is significantly lower than company average ({brk:.1f} hrs).")

    if z.get("Avg. Bay hrs_hrs_float", 0) < -1:
        lines.append(f"Focused workstation hours (Bay hrs) are below average ({bay:.1f} hrs).")
    elif z.get("Avg. Bay yrs_hrs_float", 0) > 1:
        lines.append(f"Bay hours are significantly above average ({bay:.1f} hrs).")

    if z.get("Avg. OOO hrs_hrs_float", 0) > 1:
        lines.append(f"Out-of-office hours are higher than peers ({ooo:.1f} hrs).")

    if office > 11:
        lines.append(f"Total office time is long ({office:.1f} hrs) — check for overwork or poor focus.")

    if half + full > 10:
        lines.append(f"Frequent leaves observed (Half+Full = {int(half + full)}).")

    if anom:
        lines.append("This employee is flagged as an anomaly by the model (behaviour deviates from peers).")

    if len(lines) == 1:
        lines.append("No strong deviations detected; behavior is within expected bounds.")

    # Recommendation logic combining cluster profile and anomalies
    rec = "Continue monitoring."
    # basic recommendations using cluster characteristics
    # examine cluster center for high break or low bay:
    center = cluster_profiles.loc[cluster].to_dict() if cluster in cluster_profiles.index else {}
    # heuristics
    if center.get("Avg. Break hrs_hrs_float", 0) > center.get("Avg. Bay hrs_hrs_float", 0) * 0.3 or z.get("Avg. Break hrs_hrs_float", 0) > 1:
        rec = "Recommend a coaching session on focused work and review break patterns (time management)."
    if z.get("Avg. OOO hrs_hrs_float", 0) > 1 or center.get("Avg. OOO hrs_hrs_float", 0) > 1.5:
        rec = "Recommend reviewing meeting/travel schedules — consider reducing external meetings or shifting them to asynchronous updates."
    if (office > 11 and bay < 7) or (z.get("Avg. Bay hrs_hrs_float", 0) < -1 and office > 10):
        rec = "Recommend reviewing task allocation and workload — employee works long hours but has low focused time; consider task re-alignment."
    if not anom and (half + full) < 3 and z.get("Avg. Bay hrs_hrs_float", 0) > 1:
        rec = "Good discipline observed — consider recognition/positive reinforcement."

    return lines, rec

def generate_account_story(account_code):
    """
    Generate account-level story and recommendation
    """
    account_df = df[df[COL_ACCOUNT] == account_code]
    if account_df.empty:
        return ["No data for account."], "No recommendation."

    # aggregated metrics
    avg_bay = safe_mean(account_df.get("Avg. Bay hrs_hrs_float", pd.Series()))
    avg_break = safe_mean(account_df.get("Avg. Break hrs_hrs_float", pd.Series()))
    avg_ooo = safe_mean(account_df.get("Avg. OOO hrs_hrs_float", pd.Series()))
    unbilled_pct = 0
    if COL_UNBILLED in account_df.columns:
        unbilled_pct = len(account_df[account_df[COL_UNBILLED].astype(str).str.lower().str.contains("unbilled", na=False)]) / max(1, len(account_df)) * 100

    cluster_counts = account_df["cluster"].value_counts(normalize=True).round(3).to_dict()
    anomaly_rate = account_df["anomaly_flag"].mean() * 100

    lines = []
    lines.append(f"Account {account_code} has {len(account_df)} employees. Cluster distribution: {cluster_counts}.")
    lines.append(f"Avg Bay hrs: {avg_bay:.2f} hrs; Avg Break hrs: {avg_break:.2f} hrs; Avg OOO hrs: {avg_ooo:.2f} hrs.")
    lines.append(f"Unbilled proportion: {unbilled_pct:.1f}%. Model-flagged anomaly rate: {anomaly_rate:.1f}%.")

    # recommendations heuristics
    rec = "No major action required."
    if avg_break > df["Avg. Break hrs_hrs_float"].mean() + df["Avg. Break hrs_hrs_float"].std():
        rec = "Account shows higher-than-usual break time. Recommend time-discipline coaching and structured breaks."
    if avg_bay < df["Avg. Bay hrs_hrs_float"].mean() - df["Avg. Bay hrs_hrs_float"].std():
        rec = "Account shows lower focused-work hours. Recommend task alignment and productivity workshops."
    if unbilled_pct > 30:
        rec = "Large fraction of unbilled resources — prioritize reallocation to billable projects or billing review."
    if anomaly_rate > 7:
        rec = "High anomaly rate — run deeper review of attendance data / exceptions for the account."

    return lines, rec

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/employee", methods=["GET", "POST"])
def employee():
    if request.method == "POST":
        emp_id = request.form.get("emp_id", "").strip()
        if emp_id == "":
            return render_template("employee.html", error="Please enter an Employee ID.")

        row = get_employee_row(emp_id)
        if row is None:
            return render_template("employee.html", emp_id=emp_id, error="Employee ID not found in dataset.")

        # generate chart for employee: Bay / Break / OOO / Cafe / Office
        labels = ["Bay Hrs", "Break Hrs", "OOO Hrs", "Cafeteria", "Office Hrs"]
        values = [
            row.get("Avg. Bay hrs_hrs_float", 0),
            row.get("Avg. Break hrs_hrs_float", 0),
            row.get("Avg. OOO hrs_hrs_float", 0),
            row.get("Avg. Cafeteria hrs_hrs_float", 0),
            row.get("Avg. Office hrs_hrs_float", 0)
        ]
        chart_fn = f"{emp_id}_employee_bar.png"
        chart_path = save_bar_chart(values, labels, f"Work Pattern: {emp_id}", chart_fn)

        # employee story & recommendation
        story_lines, recommendation = generate_employee_story(row)

        # small distribution chart comparing employee bay vs company
        # We'll create a histogram of company bay hours and mark employee position
        hist_fn = f"{emp_id}_bay_hist.png"
        plt.figure(figsize=(6,4))
        sns.histplot(df["Avg. Bay hrs_hrs_float"].dropna(), bins=15, kde=False)
        plt.axvline(x=row.get("Avg. Bay hrs_hrs_float", 0), color="red", linestyle="--", label=f"{emp_id} Bay hrs")
        plt.legend()
        plt.title("Company Bay Hours Distribution")
        plt.tight_layout()
        hist_path = os.path.join(CHART_DIR, hist_fn)
        plt.savefig(hist_path)
        plt.close()

        return render_template("employee.html",
                               emp_id=emp_id,
                               chart=chart_path,
                               hist=hist_path,
                               story=story_lines,
                               recommendation=recommendation)
    return render_template("employee.html")

@app.route("/account", methods=["GET", "POST"])
def account():
    if request.method == "POST":
        acc_code = request.form.get("acc_code", "").strip()
        if acc_code == "":
            return render_template("account.html", error="Please enter an account code.")
        account_df = df[df[COL_ACCOUNT] == acc_code]
        if account_df.empty:
            return render_template("account.html", acc_code=acc_code, error="Account not found or no employees for this account.")

        # create account level chart: avg Bay/Break/OOO/Cafe
        avg_bay = account_df["Avg. Bay hrs_hrs_float"].mean()
        avg_break = account_df["Avg. Break hrs_hrs_float"].mean()
        avg_ooo = account_df["Avg. OOO hrs_hrs_float"].mean()
        avg_cafe = account_df["Avg. Cafeteria hrs_hrs_float"].mean()
        labels = ["Avg Bay", "Avg Break", "Avg OOO", "Avg Cafe"]
        values = [avg_bay, avg_break, avg_ooo, avg_cafe]
        chart_fn = f"{acc_code}_account_bar.png"
        chart_path = save_bar_chart(values, labels, f"Account Work Pattern: {acc_code}", chart_fn)

        # distribution charts
        hist_fn = f"{acc_code}_office_hist.png"
        hist_path = save_account_distribution_chart(account_df, "Avg. Office hrs_hrs_float", "Office Hours Distribution", hist_fn)

        # story & rec
        story_lines, recommendation = generate_account_story(acc_code)

        return render_template("account.html",
                               acc_code=acc_code,
                               chart=chart_path,
                               hist=hist_path,
                               story=story_lines,
                               recommendation=recommendation)
    return render_template("account.html")

# ---------- Run ----------
if __name__ == "__main__":
    print("Starting Attendance Insights App...")
    app.run(debug=True)
