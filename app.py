import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils import load_data
from model import prepare_features, train_model, predict

st.set_page_config(page_title="AgriFlow AI", layout="wide", page_icon="🌾")

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

body { background: #0e1117; color: #dde3f0; }

[data-testid="stSidebar"] {
    background: #161b25 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #dde3f0 !important; }

/* Hide default streamlit header padding */
.block-container { padding-top: 1.2rem !important; }

/* ── HEADER ── */
.agri-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}
.agri-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.agri-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #00e676, #00b4ff);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.agri-logo-title {
    font-size: 20px; font-weight: 700; color: #fff;
    letter-spacing: -0.3px;
}
.agri-logo-sub { font-size: 11px; color: #4a5268; }
.hackathon-badge {
    background: linear-gradient(135deg, #00e676 0%, #00b4ff 100%);
    color: #0e1117 !important;
    font-size: 11px; font-weight: 700;
    padding: 5px 14px; border-radius: 20px;
    letter-spacing: 1px;
}

/* ── MAIN ANALYSIS CARD ── */
.main-card {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
}
.main-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 16px;
}
.main-card-left h4 {
    font-size: 13px; color: #7a8499; margin-bottom: 4px; font-weight: 500;
}
.main-card-left h1 {
    font-size: 42px; font-weight: 700; color: #fff; margin: 0; line-height: 1.1;
}
.change-down { color: #ff4d6d; font-size: 13px; margin-top: 6px; }
.change-up   { color: #00e676; font-size: 13px; margin-top: 6px; }

.main-stats {
    display: flex; gap: 28px; flex-wrap: wrap; align-items: center;
}
.stat-item { text-align: center; }
.stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #4a5268; margin-bottom: 3px; }
.stat-value { font-size: 15px; font-weight: 600; color: #dde3f0; }

/* ── SECTION TITLE ── */
.section-title {
    font-size: 15px; font-weight: 600; color: #dde3f0;
    margin: 20px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}

/* ── FORECAST CARDS ── */
.forecast-card {
    background: #1a2030;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    height: 100%;
}
.fc-green  { border: 2px solid rgba(0,230,118,0.5); }
.fc-amber  { border: 2px solid rgba(255,179,0,0.5); }
.fc-red    { border: 2px solid rgba(255,77,109,0.5); }

.fc-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; color: #7a8499; margin-bottom: 10px; }
.fc-status-green { font-size: 20px; font-weight: 700; color: #00e676; }
.fc-status-amber { font-size: 20px; font-weight: 700; color: #ffb300; }
.fc-status-red   { font-size: 20px; font-weight: 700; color: #ff4d6d; }
.fc-conf { font-size: 12px; color: #7a8499; margin-top: 6px; }
.fc-yield { font-size: 12px; color: #4a5268; margin-top: 3px; }

/* ── METRIC CARDS ── */
.metric-card {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.mc-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #4a5268; margin-bottom: 8px; }
.mc-value-green  { font-size: 22px; font-weight: 700; color: #00e676; }
.mc-value-amber  { font-size: 22px; font-weight: 700; color: #ffb300; }
.mc-value-blue   { font-size: 22px; font-weight: 700; color: #00b4ff; }
.mc-value-white  { font-size: 22px; font-weight: 700; color: #dde3f0; }

/* ── DATA COVERAGE (sidebar) ── */
.coverage-box {
    background: rgba(0,180,255,0.06);
    border: 1px solid rgba(0,180,255,0.2);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12px;
    color: #7ab8d4;
    margin-top: 10px;
    line-height: 1.7;
}
.coverage-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #00b4ff; font-weight: 700; margin-bottom: 6px;
}
.sidebar-help {
    font-size: 12px; color: #4a5268; line-height: 1.6; margin-top: 14px;
}

/* ── PLOTLY CHART CONTAINER ── */
.chart-wrap {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── LOAD DATA ───────────────────────────
df = load_data()

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:20px'>
        <div style='width:32px;height:32px;background:linear-gradient(135deg,#00e676,#00b4ff);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;'>🌾</div>
        <div>
            <div style='font-size:15px;font-weight:700;color:#fff;'>AgriFlow AI</div>
            <div style='font-size:10px;color:#4a5268;'>Agricultural Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**CONTROL PANEL**")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    country = st.selectbox("Country", sorted(df["Country"].unique()))
    crop    = st.selectbox("Crop", df[df["Country"] == country]["Crop"].unique())

    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    period   = st.selectbox("Analysis Period", [
        f"Full Period ({year_min}–{year_max})",
        f"Last 5 Years ({year_max-4}–{year_max})",
        f"Last 3 Years ({year_max-2}–{year_max})",
    ])

    run = st.button("▶  Run Analysis", use_container_width=True,
                    type="primary")

    # Data Coverage
    n_countries = df["Country"].nunique()
    n_crops     = df["Crop"].nunique()
    n_records   = len(df)
    n_combos    = df.groupby(["Country","Crop"]).ngroups

    st.markdown(f"""
    <div class="coverage-box">
        <div class="coverage-title">DATA COVERAGE</div>
        {n_countries} countries · {n_crops} crops · {year_min}–{year_max}<br>
        {n_records:,} agricultural records · {n_combos:,} combinations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-help">
        Select a country and crop to get yield trends, food security forecasts,
        trade analysis, and AI-powered recommendations.
    </div>
    """, unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div class="agri-header">
        <div class="agri-logo">
            <div class="agri-logo-icon">🌾</div>
            <div>
                <div class="agri-logo-title">AgriFlow <span style='color:#00e676'>AI</span></div>
                <div class="agri-logo-sub">Agricultural Intelligence Platform</div>
            </div>
        </div>
        <span class="hackathon-badge">HACKATHON 2025</span>
    </div>
    <div style='text-align:center;padding:80px 0;color:#4a5268;'>
        <div style='font-size:48px;margin-bottom:16px;'>🌾</div>
        <div style='font-size:18px;color:#7a8499;'>Select a country and crop, then click <b>Run Analysis</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────── FILTER ───────────────────────────
data = df[(df["Country"] == country) & (df["Crop"] == crop)].sort_values("Year")

if len(data) < 3:
    st.error("Not enough data for this crop/country combination. Please select another.")
    st.stop()

# Apply period filter
if "Last 5" in period:
    data = data[data["Year"] >= year_max - 4]
elif "Last 3" in period:
    data = data[data["Year"] >= year_max - 2]

# ─────────────────────────── MODEL ───────────────────────────
data_model         = prepare_features(data)
model, features    = train_model(data_model)
pred, prob         = predict(model, data_model, features)

latest = data.iloc[-1]
prev   = data.iloc[-2] if len(data) >= 2 else latest

yield_change    = latest["Yield"] - prev["Yield"]
yield_change_pct = (yield_change / prev["Yield"] * 100) if prev["Yield"] != 0 else 0
arrow = "▼" if yield_change < 0 else "▲"
change_class = "change-down" if yield_change < 0 else "change-up"

# ─────────────────────────── HEADER ───────────────────────────
st.markdown(f"""
<div class="agri-header">
    <div class="agri-logo">
        <div class="agri-logo-icon">🌾</div>
        <div>
            <div class="agri-logo-title">AgriFlow <span style='color:#00e676'>AI</span></div>
            <div class="agri-logo-sub">Agricultural Intelligence Platform</div>
        </div>
    </div>
    <span class="hackathon-badge">HACKATHON 2025</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── MAIN CARD ───────────────────────────
prod_val    = f"{latest['Production']/1000:.0f}K MT" if latest['Production'] > 1000 else f"{latest['Production']:.0f} MT"
rain_val    = f"{latest['Rainfall']:.0f}mm"
temp_val    = f"{latest['Temperature']:.1f}°C"
fsi_val     = latest['FSI']
year_val    = int(latest["Year"])
price_val   = f"${latest['Price']:.0f}/MT" if 'Price' in latest.index else "N/A"
trade_val   = f"${latest['TradeBalance']:.0f}M" if 'TradeBalance' in latest.index else "N/A"
drought_val = latest['DroughtRisk'] if 'DroughtRisk' in latest.index else "N/A"
fs_status   = latest['FSStatus'] if 'FSStatus' in latest.index else ("Food Secure" if fsi_val > 60 else "At Risk")
fs_color    = "#00e676" if "Secure" in str(fs_status) else "#ffb300" if "Moderate" in str(fs_status) else "#ff4d6d"

st.markdown(f"""
<div class="main-card">
    <div class="main-card-top">
        <div class="main-card-left">
            <h4>Current Analysis: {country} — {crop} ({year_val})</h4>
            <h1>{latest['Yield']:.2f} MT/ha</h1>
            <div class="{change_class}">{arrow} {abs(yield_change_pct):.2f}% vs prev year &nbsp;|&nbsp;
                Region: {country} &nbsp;|&nbsp; Category: {crop}</div>
        </div>
        <div class="main-stats">
            <div class="stat-item">
                <div class="stat-label">PRICE</div>
                <div class="stat-value">{price_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">PRODUCTION</div>
                <div class="stat-value">{prod_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">TRADE BALANCE</div>
                <div class="stat-value">{trade_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">FSI STATUS</div>
                <div class="stat-value" style="color:{fs_color};">{fs_status}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">DROUGHT</div>
                <div class="stat-value">{drought_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">RAINFALL</div>
                <div class="stat-value">{rain_val}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── FORECAST CARDS ───────────────────────────
st.markdown('<div class="section-title">📊 Forecast Summary</div>', unsafe_allow_html=True)

def forecast_label(p):
    return "IMPROVING" if p == 1 else "DECLINING"

def forecast_color(p):
    return "green" if p == 1 else "amber"

short_label  = forecast_label(pred)
short_color  = forecast_color(pred)
# Medium / long term: use slightly shifted versions (simulate)
medium_pred  = 1 if prob > 0.4 else 0
long_pred    = 1 if prob > 0.35 else 0
medium_label = "STABLE" if abs(prob - 0.5) < 0.15 else forecast_label(medium_pred)
long_label   = "STABLE" if abs(prob - 0.5) < 0.2  else forecast_label(long_pred)
medium_color = "amber"
long_color   = "amber"

short_conf  = prob * 100
medium_conf = max(30, prob * 85)
long_conf   = max(30, prob * 82)

est_yield       = latest["Yield"] * (1.03 if pred == 1 else 0.97)
est_yield_med   = latest["Yield"] * 1.005
est_yield_long  = latest["Yield"] * 1.01

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="forecast-card fc-{short_color}">
        <div class="fc-label">SHORT-TERM</div>
        <div class="fc-status-{short_color}">{short_label}</div>
        <div class="fc-conf">Confidence: {short_conf:.1f}%</div>
        <div class="fc-yield">Est. yield: {est_yield:.2f} MT/ha</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="forecast-card fc-{medium_color}">
        <div class="fc-label">MEDIUM-TERM</div>
        <div class="fc-status-{medium_color}">{medium_label}</div>
        <div class="fc-conf">Confidence: {medium_conf:.1f}%</div>
        <div class="fc-yield">Est. yield: {est_yield_med:.2f} MT/ha</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="forecast-card fc-{long_color}">
        <div class="fc-label">LONG-TERM</div>
        <div class="fc-status-{long_color}">{long_label}</div>
        <div class="fc-conf">Confidence: {long_conf:.1f}%</div>
        <div class="fc-yield">Est. yield: {est_yield_long:.2f} MT/ha</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── METRICS ───────────────────────────
st.markdown('<div class="section-title">📈 Key Metrics</div>', unsafe_allow_html=True)

avg_yield     = data["Yield"].mean()
profit_margin = data["ProfitMargin"].mean() if "ProfitMargin" in data.columns else 0
loss_rate     = data["LossRate"].mean()     if "LossRate"     in data.columns else 0
avg_fsi       = data["FSI"].mean()
mechanization = latest["Mechanization"]     if "Mechanization" in latest.index else 0
irrigation    = latest["Irrigation"]        if "Irrigation"    in latest.index else 0
soil_health   = latest["SoilHealth"]        if "SoilHealth"    in latest.index else 0
avg_temp      = data["Temperature"].mean()

cols = st.columns(8)

metric_data = [
    ("AVG YIELD",      f"{avg_yield:.2f} MT/ha", "green"),
    ("PROFIT MARGIN",  f"{profit_margin:.1f}%",  "green"),
    ("LOSS RATE",      f"{loss_rate:.1f}%",       "amber"),
    ("AVG FSI",        f"{avg_fsi:.1f}",          "blue"),
    ("MECHANIZATION",  f"{mechanization:.0f}%",   "blue"),
    ("IRRIGATION",     f"{irrigation:.0f}%",      "amber"),
    ("SOIL HEALTH",    f"{soil_health:.0f}/100",  "green"),
    ("AVG TEMP",       f"{avg_temp:.1f}°C",       "white"),
]

for i, (label, value, color) in enumerate(metric_data):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">{label}</div>
            <div class="mc-value-{color}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────── CHART ───────────────────────────
st.markdown(f'<div class="section-title">📉 Yield Performance Chart</div>', unsafe_allow_html=True)
st.markdown(f"<div style='font-size:13px;color:#7a8499;margin-bottom:10px;'>{country} — {crop} Yield History ({int(data['Year'].min())}–{int(data['Year'].max())})</div>", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["Year"], y=data["Yield"],
    mode="lines+markers",
    name="Yield (MT/ha)",
    line=dict(color="#00e676", width=2.5),
    marker=dict(size=6, color="#00e676"),
    fill="tozeroy",
    fillcolor="rgba(0,230,118,0.06)"
))

# Add rolling average
if len(data) >= 3:
    roll = data["Yield"].rolling(3, center=True).mean()
    fig.add_trace(go.Scatter(
        x=data["Year"], y=roll,
        mode="lines",
        name="3-Year Avg",
        line=dict(color="#00b4ff", width=1.8, dash="dash"),
        opacity=0.8
    ))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1a2030",
    plot_bgcolor="#1a2030",
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(
        orientation="h", x=0, y=1.08,
        font=dict(size=12, color="#7a8499"),
        bgcolor="rgba(0,0,0,0)"
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#7a8499")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#7a8499"),
               title=dict(text="MT/ha", font=dict(color="#7a8499"))),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
