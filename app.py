import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import date

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AMSES — Smart Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (light theme — minimal)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }

    .header-banner {
        background: linear-gradient(135deg, #1a3c6e 0%, #2563eb 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-banner h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .header-banner p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.85; }

    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #2563eb;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1a3c6e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    .result-normal  { background:#dcfce7; border:2px solid #16a34a; border-radius:12px; padding:1.5rem; }
    .result-warning { background:#fef9c3; border:2px solid #ca8a04; border-radius:12px; padding:1.5rem; }
    .result-danger  { background:#fee2e2; border:2px solid #dc2626; border-radius:12px; padding:1.5rem; }
    .result-title   { font-size:1.5rem; font-weight:700; margin-bottom:0.4rem; }
    .result-subtitle{ font-size:0.95rem; opacity:0.8; }

    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #1a3c6e, #2563eb);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 3rem;
        border-radius: 10px;
        border: none;
        width: 100%;
    }

    .ind-card {
        background: white;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .ind-label { font-size: 0.78rem; color: #64748b; margin-bottom: 2px; }
    .ind-value { font-size: 1.05rem; font-weight: 700; color: #1e293b; }
    .ind-badge {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ANOMALY LABEL MAP
# ─────────────────────────────────────────────
ANOMALY_LABELS = {
    0: ("✅ Normal – No Anomaly",         "normal",  "The meter is operating within expected parameters. No suspicious activity detected."),
    1: ("⚠️ Meter Bypass / Tampering",    "danger",  "Possible meter tampering or bypass detected. Immediate field inspection recommended."),
    2: ("🔶 Unusual Consumption Pattern", "warning", "Consumption pattern is abnormal compared to historical norms. May indicate appliance malfunction or theft."),
    3: ("🌩️ Grid Outage Impact",          "warning", "Anomaly is likely linked to a grid outage event affecting meter readings."),
    4: ("🔴 Overload / High Usage",       "danger",  "Extremely high energy usage detected. Risk of overload or illegal commercial activity."),
}

# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("anomaly_model.pkl"), joblib.load("scaler.pkl"), True
    except FileNotFoundError:
        return None, None, False

model, scaler, model_loaded = load_model()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>⚡ AMSES — Smart Energy Intelligence Platform</h1>
    <p>AI-powered Meter Surveillance and Energy Anomaly Detection System · Random Forest · 92% Accuracy</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ **Model files not found.** Please place `anomaly_model.pkl` and `scaler.pkl` in the same folder as this app.")

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
1. Fill in the meter details across all sections  
2. Click **Predict Anomaly**  
3. View the result and recommendation  
    """)
    st.markdown("---")
    st.markdown("## 🏷️ Anomaly Types")
    for code, (label, _, desc) in ANOMALY_LABELS.items():
        st.markdown(f"**{label}**  \n_{desc}_\n")
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("Built with **Random Forest Classifier** trained on Kerala smart meter data. Feature engineering runs automatically in the background.")

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
SEASON_ENCODE   = {"Summer": 2, "Monsoon": 1, "Winter": 0}
DWELLING_ENCODE = {"Apartment": 0, "Commercial": 1, "Independent House": 2, "Villa": 3}

def get_season(month):
    if month in [3, 4, 5]:    return "Summer"
    if month in [6, 7, 8, 9]: return "Monsoon"
    return "Winter"

# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Section 1 ─────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">🏠 Meter & Property Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        dwelling_type  = st.selectbox("Dwelling Type", ["Apartment", "Villa", "Independent House", "Commercial"],
                                       help="Type of property where the meter is installed")
        num_occupants  = st.number_input("Number of Occupants", min_value=1, max_value=8, value=3,
                                          help="Total number of people living/working in the premises")
    with col2:
        house_area     = st.number_input("House Area (sq. ft.)", min_value=400, max_value=5000, value=1200, step=50,
                                          help="Total floor area of the property")
        connected_load = st.number_input("Connected Load (kW)", min_value=1.5, max_value=15.0, value=5.0, step=0.5,
                                          help="Sanctioned/connected electrical load in kilowatts")
    with col3:
        meter_age  = st.number_input("Meter Age (Years)", min_value=1, max_value=19, value=5,
                                      help="Age of the installed smart meter in years")
        has_solar  = st.selectbox("Has Solar Panel?", ["No", "Yes"],
                                   help="Whether the property has a rooftop solar panel")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2 ─────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📅 Date & Environmental Conditions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        reading_date = st.date_input("Reading Date", value=date.today(),
                                      help="Date of the meter reading")
    with col2:
        temperature  = st.slider("Temperature (°C)", min_value=22.0, max_value=38.0, value=28.0, step=0.5,
                                  help="Ambient temperature on the day of reading")
    with col3:
        humidity     = st.slider("Humidity (%)", min_value=45.0, max_value=95.0, value=65.0, step=1.0,
                                  help="Relative humidity percentage")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3 ─────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">⚡ Energy Consumption Readings</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        expected_energy = st.number_input("Expected Energy (kWh)", min_value=0.0, max_value=60.0, value=25.0, step=0.5,
                                           help="The energy consumption that was expected/estimated for this period")
        actual_energy   = st.number_input("Actual Energy Consumed (kWh)", min_value=1.0, max_value=260.0, value=30.0, step=0.5,
                                           help="The actual energy consumption recorded by the meter")
    with col2:
        peak_usage    = st.number_input("Peak Hour Usage (kWh)", min_value=0.0, max_value=165.0, value=18.0, step=0.5,
                                         help="Energy consumed during peak hours (typically 6 PM – 10 PM)")
        offpeak_usage = st.number_input("Off-Peak Usage (kWh)", min_value=0.0, max_value=90.0, value=10.0, step=0.5,
                                         help="Energy consumed during off-peak hours")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4 ─────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">🔧 Meter Health & Electrical Parameters</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        voltage      = st.number_input("Voltage (V)", min_value=195.0, max_value=260.0, value=230.0, step=0.5,
                                        help="Voltage reading at the meter point")
        power_factor = st.slider("Power Factor", min_value=0.65, max_value=0.99, value=0.85, step=0.01,
                                  help="Ratio of real power to apparent power (ideal ≥ 0.85)")
    with col2:
        appliance_score = st.number_input("Appliance Score", min_value=1, max_value=25, value=8,
                                           help="Score based on number and type of registered appliances (1=low, 25=high load)")
        grid_outage_hrs = st.number_input("Grid Outage Hours", min_value=0.0, max_value=8.0, value=0.0, step=0.25,
                                           help="Number of hours the grid was down during this period (0 = no outage)")
    with col3:
        bypass_signal = st.selectbox("Meter Bypass Signal", ["No (0)", "Yes (1)"],
                                      help="Has the meter detected a bypass/tampering signal?")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Anomaly")

# ─────────────────────────────────────────────
#  PREDICTION + VISUALISATIONS
# ─────────────────────────────────────────────
if submitted:

    # ── Feature Engineering ────────────────────
    solar_val  = 1 if has_solar == "Yes" else 0
    bypass_val = 1 if bypass_signal.startswith("Yes") else 0
    month      = reading_date.month
    dow        = reading_date.weekday()
    season_str = get_season(month)

    eps = 1e-5
    energy_deviation_ratio = (actual_energy - expected_energy) / (expected_energy + eps)
    energy_per_occupant    = actual_energy / (num_occupants + eps)
    peak_to_total_ratio    = peak_usage / (actual_energy + eps)
    load_utilization_pct   = actual_energy / (connected_load * 24 + eps)
    energy_per_sqft        = actual_energy / (house_area + eps)
    anomaly_risk_score     = bypass_val*3 + int(grid_outage_hrs > 0) + int(energy_deviation_ratio > 1)

    dwelling_enc = DWELLING_ENCODE[dwelling_type]
    season_enc   = SEASON_ENCODE[season_str]

    feature_vector = np.array([[
        dwelling_enc, num_occupants, house_area, connected_load, meter_age,
        temperature, humidity, grid_outage_hrs, appliance_score,
        bypass_val, solar_val, voltage, power_factor,
        expected_energy, actual_energy, peak_usage, offpeak_usage,
        month, dow, season_enc,
        energy_deviation_ratio, energy_per_occupant, peak_to_total_ratio,
        load_utilization_pct, energy_per_sqft, anomaly_risk_score
    ]])

    st.markdown("---")

    # ── Predict ────────────────────────────────
    if model_loaded:
        scaled     = scaler.transform(feature_vector)
        pred       = int(model.predict(scaled)[0])
        proba      = model.predict_proba(scaled)[0]
        confidence = round(float(max(proba)) * 100, 1)
    else:
        pred, proba, confidence = 0, [0.7, 0.1, 0.1, 0.05, 0.05], 70.0

    label, style, explanation = ANOMALY_LABELS[pred]

    # ── Result Banner ──────────────────────────
    st.markdown(
        f'<div class="result-{style}">'
        f'<div class="result-title">{label}</div>'
        f'<div class="result-subtitle">{explanation}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Chart color follows anomaly type
    CHART_COLORS = {
        0: "#16a34a", 1: "#dc2626", 2: "#ca8a04", 3: "#7c3aed", 4: "#ea580c"
    }
    main_color = CHART_COLORS[pred]

    # ══════════════════════════════════════════
    #  ROW 1 — Gauge · Bar · Donut
    # ══════════════════════════════════════════
    st.markdown("#### 📊 Prediction Analysis")
    c1, c2, c3 = st.columns([1, 1.1, 0.95])

    # Confidence Gauge
    with c1:
        st.markdown("**Confidence Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 34, "color": main_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8",
                         "tickfont": {"size": 10, "color": "#94a3b8"}},
                "bar":  {"color": main_color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0,  50], "color": "#fef2f2"},
                    {"range": [50, 75], "color": "#fefce8"},
                    {"range": [75,100], "color": "#f0fdf4"},
                ],
                "threshold": {
                    "line": {"color": "#2563eb", "width": 2},
                    "thickness": 0.8,
                    "value": 92
                }
            }
        ))
        fig_gauge.update_layout(
            height=230, margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="white", font={"color": "#64748b"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Class Probability Bar Chart
    with c2:
        st.markdown("**Probability Across All Anomaly Classes**")
        short_labels   = ["Normal", "Bypass", "Unusual", "Outage", "Overload"]
        all_colors     = ["#16a34a", "#dc2626", "#ca8a04", "#7c3aed", "#ea580c"]
        # highlight predicted, dim others
        final_colors = []
        for i, c in enumerate(all_colors):
            if i == pred:
                final_colors.append(c)
            else:
                r, g2, b3 = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
                final_colors.append(f"rgba({r},{g2},{b3},0.28)")

        fig_bar = go.Figure(go.Bar(
            x=short_labels,
            y=[round(p * 100, 1) for p in proba],
            marker_color=final_colors,
            marker_line_color=all_colors,
            marker_line_width=1.5,
            text=[f"{round(p*100,1)}%" for p in proba],
            textposition="outside",
            textfont={"size": 11, "color": "#475569"},
        ))
        fig_bar.update_layout(
            height=230, margin=dict(l=10, r=10, t=15, b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis={"tickfont": {"size": 11, "color": "#475569"}, "gridcolor": "#f1f5f9"},
            yaxis={"tickfont": {"size": 10, "color": "#94a3b8"}, "gridcolor": "#f1f5f9",
                   "range": [0, max(p*100 for p in proba) * 1.3]},
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Energy Donut
    with c3:
        st.markdown("**Energy Distribution**")
        rest = max(0, actual_energy - peak_usage - offpeak_usage)
        fig_donut = go.Figure(go.Pie(
            labels=["Peak Hours", "Off-Peak", "Other"],
            values=[peak_usage, offpeak_usage, rest],
            hole=0.62,
            marker={"colors": ["#2563eb", "#06b6d4", "#cbd5e1"],
                    "line": {"color": "white", "width": 2}},
            textfont={"size": 11},
            hovertemplate="<b>%{label}</b><br>%{value:.1f} kWh · %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{round(actual_energy, 1)}</b><br>kWh",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 14, "color": "#1e293b"}, align="center"
        )
        fig_donut.update_layout(
            height=230, margin=dict(l=5, r=5, t=15, b=5),
            paper_bgcolor="white",
            legend={"font": {"size": 11, "color": "#475569"}, "bgcolor": "white"},
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════
    #  ROW 2 — Radar · Key Indicators
    # ══════════════════════════════════════════
    st.markdown("#### 🔬 Feature Analysis")
    c4, c5 = st.columns([1.3, 1])

    # Radar Chart
    with c4:
        st.markdown("**Engineered Feature Radar**")

        def norm(v, lo, hi):
            return max(0, min(1, (v - lo) / (hi - lo + 1e-9)))

        radar_vals = [
            norm(energy_deviation_ratio, -2, 5),
            norm(energy_per_occupant, 0, 50),
            norm(peak_to_total_ratio, 0, 1),
            norm(load_utilization_pct, 0, 1),
            norm(energy_per_sqft, 0, 0.1),
            norm(anomaly_risk_score, 0, 4),
            norm(abs(voltage - 230) / 230, 0, 0.15),
            norm(1 - power_factor, 0, 0.35),
        ]
        radar_labels = [
            "Energy Deviation", "Energy/Person", "Peak Ratio",
            "Load Utilization", "Energy/SqFt", "Risk Score",
            "Voltage Anomaly", "Low Power Factor"
        ]
        rv_c = radar_vals + [radar_vals[0]]
        rl_c = radar_labels + [radar_labels[0]]

        r, g2, b3 = int(main_color[1:3],16), int(main_color[3:5],16), int(main_color[5:7],16)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=rv_c, theta=rl_c, fill="toself",
            fillcolor=f"rgba({r},{g2},{b3},0.12)",
            line={"color": main_color, "width": 2},
            marker={"size": 6, "color": main_color},
            name="Current Reading"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.5]*9, theta=rl_c, fill="toself",
            fillcolor="rgba(37,99,235,0.05)",
            line={"color": "#2563eb", "width": 1, "dash": "dot"},
            name="Baseline (0.5)"
        ))
        fig_radar.update_layout(
            height=310, margin=dict(l=55, r=55, t=25, b=25),
            paper_bgcolor="white",
            polar={
                "bgcolor": "#f8fafc",
                "radialaxis": {"visible": True, "range": [0, 1],
                               "tickfont": {"size": 8, "color": "#94a3b8"},
                               "gridcolor": "#e2e8f0", "linecolor": "#e2e8f0"},
                "angularaxis": {"tickfont": {"size": 10, "color": "#475569"},
                                "gridcolor": "#e2e8f0", "linecolor": "#e2e8f0"},
            },
            legend={"font": {"size": 10, "color": "#64748b"}, "bgcolor": "white",
                    "x": 0.82, "y": 0.05},
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    # Key Indicators
    with c5:
        st.markdown("**Key Indicators**")

        STATUS_BG = {
            "Normal": "#dcfce7", "Good": "#dcfce7", "Low Risk": "#dcfce7",
            "Moderate": "#fef9c3", "Elevated": "#fef9c3", "High": "#fef9c3",
            "Fair": "#fef9c3", "Medium": "#fef9c3",
            "Very High": "#fee2e2", "Overloaded": "#fee2e2",
            "Poor": "#fee2e2", "High Risk": "#fee2e2",
        }

        def indicator(label, value, unit, status, color):
            bg = STATUS_BG.get(status, "#f1f5f9")
            st.markdown(f"""
            <div class="ind-card" style="border-left-color:{color};">
                <div>
                    <div class="ind-label">{label}</div>
                    <div class="ind-value">{value}
                        <span style="font-size:0.75rem;color:#94a3b8;font-weight:400;"> {unit}</span>
                    </div>
                </div>
                <div class="ind-badge" style="background:{bg};color:{color};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        dv   = round(energy_deviation_ratio * 100, 1)
        indicator("Energy Deviation Ratio", f"{dv}", "%",
                  "High" if abs(dv)>50 else ("Moderate" if abs(dv)>20 else "Normal"),
                  "#dc2626" if abs(dv)>50 else ("#ca8a04" if abs(dv)>20 else "#16a34a"))

        ptu  = round(peak_to_total_ratio * 100, 1)
        indicator("Peak-to-Total Ratio", f"{ptu}", "%",
                  "Very High" if ptu>80 else ("Elevated" if ptu>60 else "Normal"),
                  "#dc2626" if ptu>80 else ("#ca8a04" if ptu>60 else "#16a34a"))

        lu   = round(load_utilization_pct * 100, 1)
        indicator("Load Utilization", f"{lu}", "%",
                  "Overloaded" if lu>80 else ("High" if lu>50 else "Normal"),
                  "#dc2626" if lu>80 else ("#ca8a04" if lu>50 else "#16a34a"))

        indicator("Power Factor", f"{power_factor}", "",
                  "Good" if power_factor>=0.85 else ("Fair" if power_factor>=0.75 else "Poor"),
                  "#16a34a" if power_factor>=0.85 else ("#ca8a04" if power_factor>=0.75 else "#dc2626"))

        indicator("Anomaly Risk Score", f"{anomaly_risk_score}", "/ 5",
                  "High Risk" if anomaly_risk_score>=3 else ("Medium" if anomaly_risk_score>=1 else "Low Risk"),
                  "#dc2626" if anomaly_risk_score>=3 else ("#ca8a04" if anomaly_risk_score>=1 else "#16a34a"))

    # ══════════════════════════════════════════
    #  Metrics Row
    # ══════════════════════════════════════════
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Confidence",   f"{confidence}%")
    m2.metric("Detected Anomaly",   f"Type {pred}")
    m3.metric("Anomaly Risk Score", anomaly_risk_score)

    # ══════════════════════════════════════════
    #  Auto-Computed Features Table
    # ══════════════════════════════════════════
    st.markdown("### 🔧 Auto-Computed Backend Features")
    st.markdown("These values were **automatically derived** from your inputs and fed to the model:")

    fe_data = {
        "Feature": [
            "Month", "Day of Week", "Season",
            "Energy Deviation Ratio", "Energy Per Occupant",
            "Peak to Total Ratio", "Load Utilization %",
            "Energy Per Sq.ft", "Anomaly Risk Score"
        ],
        "Value": [
            month,
            f"{dow} ({'Mon Tue Wed Thu Fri Sat Sun'.split()[dow]})",
            season_str,
            round(energy_deviation_ratio, 4),
            round(energy_per_occupant, 4),
            round(peak_to_total_ratio, 4),
            round(load_utilization_pct, 4),
            round(energy_per_sqft, 6),
            anomaly_risk_score
        ],
        "What It Means": [
            "Month number extracted from reading date",
            "Day of week (0=Monday, 6=Sunday)",
            "Climate season (Summer / Monsoon / Winter)",
            "(Actual − Expected) ÷ Expected  |  High value = suspicious",
            "Actual kWh per person  |  Unusually high = overconsumption",
            "Peak hours share of total usage  |  Very high = abnormal load timing",
            "How much of the sanctioned load is being used",
            "Consumption density by floor area",
            "Composite risk signal (Bypass × 3 + Outage + Deviation)"
        ]
    }
    st.dataframe(pd.DataFrame(fe_data), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    #  Recommended Action
    # ══════════════════════════════════════════
    actions = {
        0: ("No Action Required",            "#16a34a", "📋",
            "The meter is functioning normally. Continue regular monthly monitoring."),
        1: ("Immediate Field Inspection",     "#dc2626", "🚨",
            "Dispatch a field engineer to physically inspect the meter for tampering or bypass wiring. File an FIR if confirmed."),
        2: ("Flag for Follow-up Audit",       "#ca8a04", "⚠️",
            "Schedule a detailed energy audit. Cross-check appliance registration against actual usage."),
        3: ("Review Outage Compensation",     "#7c3aed", "🌩️",
            "Verify grid outage records. Apply appropriate energy credit if outage is confirmed."),
        4: ("Overload Notice & Safety Check", "#ea580c", "🔴",
            "Issue an overload warning. Inspect premises for undeclared high-load equipment."),
    }
    at, ac, ai, atxt = actions[pred]
    st.markdown(f"""
    <div style="margin-top:1rem;padding:1.2rem 1.5rem;background:white;
                border-radius:12px;border-left:5px solid {ac};
                box-shadow:0 2px 8px rgba(0,0,0,0.07);">
        <div style="font-size:1rem;font-weight:700;color:{ac};margin-bottom:0.4rem;">
            {ai} Recommended Action: {at}
        </div>
        <div style="font-size:0.88rem;color:#475569;">{atxt}</div>
    </div>
    """, unsafe_allow_html=True)