import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import features
from chart_of_accounts import ALTMAN_VARIANT, ALTMAN_VARIANTS, BENEISH_THRESHOLD

DB_PATH = "data/red_shift.db"
MODEL_PATH = "data/red_shift_risk_model.pkl"
INDEX_ORDER = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]

st.set_page_config(page_title="Red Shift | Forensic Analytics Platform", page_icon="🔍", layout="wide")

st.title("Red Shift: Automated Forensic Analytics")
st.caption("Forensic Red-Flag Detection for Listed Indian Companies")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


conn = sqlite3.connect(DB_PATH)

companies_df = pd.read_sql_query("SELECT company_id, symbol, name, sector, is_financial FROM COMPANY", conn)
if companies_df.empty:
    st.error("No companies found in database. Run load_bulk_data.py first.")
    st.stop()

selected_symbol = st.selectbox("Select Target Company to Inspect:", options=companies_df["symbol"].tolist(), index=0)
company_row = companies_df[companies_df["symbol"] == selected_symbol].iloc[0]
company_id = int(company_row["company_id"])
PRICES_CSV = f"data/{selected_symbol}.NS_prices.csv"          # per-company price file

# Latest screening run = latest FISCAL YEAR (not latest run_id)
run_df = pd.read_sql_query(
    """SELECT run_id, fiscal_year, m_score, z_score, f_score, composite_score, severity_band, model_version
       FROM SCORE_RUN WHERE company_id = ?
       ORDER BY fiscal_year DESC, run_id DESC LIMIT 1;""", conn, params=(company_id,))
if run_df.empty:
    st.warning("No screening run data available. Run quant_scorer.py and report_generator.py first.")
    st.stop()

latest_run = run_df.iloc[0]
run_id = int(latest_run["run_id"])
run_year = int(latest_run["fiscal_year"])

# Altman cut-offs for the variant this run used
parts = str(latest_run["model_version"]).split("/")
variant = parts[1] if len(parts) > 1 and parts[1] in ALTMAN_VARIANTS else ALTMAN_VARIANT
z_distress = ALTMAN_VARIANTS[variant]["cutoffs"][0]

# ----------------- SECTION 1: HEADER METRICS -----------------
st.subheader(f"Screening Summary: {company_row['name']} ({company_row['symbol']}) | FY{run_year}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    cs = latest_run["composite_score"]
    st.metric("Composite Risk Score", f"{cs} / 100" if pd.notnull(cs) else "N/A")
with col2:
    sb = latest_run["severity_band"]
    st.metric("Severity Band", str(sb).upper() if pd.notnull(sb) else "N/A")
with col3:
    m_val = latest_run["m_score"]
    if pd.notnull(m_val):
        st.metric("Beneish M-Score", f"{m_val:.2f}",
                  delta=f"Manipulator (> {BENEISH_THRESHOLD})" if m_val > BENEISH_THRESHOLD else "Safe",
                  delta_color="inverse")
    else:
        st.metric("Beneish M-Score", "N/A")
with col4:
    z_val = latest_run["z_score"]
    if pd.notnull(z_val):
        st.metric(f"Altman Z-Score ({variant})", f"{z_val:.2f}",
                  delta=f"Distress (< {z_distress})" if z_val < z_distress else "Not in distress")
    else:
        st.metric("Altman Z-Score", "N/A")

if int(company_row["is_financial"]) == 1:
    st.info("Beneish and Altman models are not valid for banks, NBFCs and insurers (FR-20), so no score is reported.")

st.divider()

# ----------------- SECTION 1.5: EXPERIMENTAL ML BASELINE -----------------
st.subheader("🤖 Machine Learning Baseline (experimental)")
st.caption("A Random Forest trained on 24 company-years from 8 companies to reproduce a rule-based risk label. "
           "It has not been validated on unseen companies and can disagree with the scores above.")
try:
    model = load_model()
    feats = features.build_features(company_id=company_id)
    yr = feats["years"].get(run_year)
    if yr and yr["beneish"].get("m_score") is not None:
        idx = yr["beneish"]["indices"]
        pred = model.predict([[idx[k] for k in INDEX_ORDER]])[0]
        if pred == 1:
            st.warning("Baseline model output: **risk pattern** (resembles company-years the rule labelled risky).")
        else:
            st.success("Baseline model output: **no risk pattern** (resembles company-years the rule labelled safe).")
    else:
        st.info("Baseline model not run: no Beneish indices for this company-year (e.g. financial sector or missing data).")
except FileNotFoundError:
    st.caption("Model file not found. Run train_model.py to create data/red_shift_risk_model.pkl.")
except Exception as e:
    st.caption(f"Baseline model unavailable: {e}")

st.divider()

# ----------------- SECTION 2: RED FLAGS & EVIDENCE -----------------
st.subheader("Forensic Red Flags & Evidence Reference")
flags_df = pd.read_sql_query(
    "SELECT category, flag_type, severity, explanation, evidence_ref FROM RED_FLAG WHERE run_id = ?;",
    conn, params=(run_id,))
if not flags_df.empty:
    st.dataframe(flags_df, width="stretch", column_config={
        "category": "Category", "flag_type": "Flag Type", "severity": "Severity",
        "explanation": "Plain Language Finding", "evidence_ref": "Underlying Traceable Evidence"})
else:
    st.success("No red flags raised for this reporting period.")

# ----------------- SECTION 3: QUALITATIVE AUDIT LOG -----------------
st.subheader("Filing Narrative Findings (Grounding Guard Verified)")
qual_df = pd.read_sql_query(
    "SELECT theme, severity, finding_text, quotation, verification_status FROM QUAL_FINDING WHERE run_id = ?;",
    conn, params=(run_id,))
if not qual_df.empty:
    for _, q in qual_df.iterrows():
        with st.expander(f"[{q['severity']}] {q['theme']} — Status: {q['verification_status']}"):
            st.write(f"**Finding:** {q['finding_text']}")
            st.info(f"**Verbatim Source Quotation:** \"{q['quotation']}\"")
else:
    st.info("No narrative disclosures recorded for this company-year.")

# ----------------- SECTION 4: MARKET BEHAVIOUR CHART -----------------
st.subheader("Market Behaviour & Price Action")
try:
    price_df = pd.read_csv(PRICES_CSV, header=[0, 1], index_col=0)
    price_df.columns = price_df.columns.get_level_values(0)
    price_df.index = pd.to_datetime(price_df.index, errors="coerce")
    price_df = price_df.dropna(subset=["Close"])
    fig = px.line(price_df, x=price_df.index, y="Close", title=f"{selected_symbol} Historical Daily Closing Price",
                  labels={"x": "Date", "Close": "Closing Price (INR)"})
    st.plotly_chart(fig, width="stretch")
except Exception:
    st.caption(f"Price chart unavailable: no price file found at {PRICES_CSV}.")

conn.close()

# Mandatory analytical disclaimer (FR-48)
st.caption("Disclaimer: This report constitutes forensic analytical information for screening purposes and does not "
           "represent investment advice. No assertion of statutory fraud is made or implied.")