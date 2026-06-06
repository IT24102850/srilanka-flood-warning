# app.py — Sri Lanka Flood & Landslide Early Warning System

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import geopandas as gpd
import pydeck as pdk

st.set_page_config(
    page_title="Sri Lanka Flood Early Warning",
    page_icon="🌧️",
    layout="wide"
)

# ── Premium UI / Custom CSS ───────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
    
    /* Main background and font adjustments */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
        color: #FAFAFA;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        animation: fadeInDown 0.8s ease-out;
    }
    /* Metric Cards Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 15px;
        transition: transform 0.2s ease-in-out;
        animation: fadeInUp 0.8s ease-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 12px 40px 0 rgba(0, 201, 255, 0.2);
        border: 1px solid rgba(0, 201, 255, 0.4);
    }
    /* Smooth animations */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "xgb_v2.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_geodata():
    return gpd.read_file(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LKA_1.json"
    )

model, best_thresh, FEATURES = load_model()
lka = load_geodata()

# ── Terrain risk ──────────────────────────────────────────
terrain = {
    'Ratnapura':1.0,'Kandy':0.95,'Badulla':0.95,'Kegalle':0.90,
    'Nuwara Eliya':0.90,'Matale':0.80,'Kalutara':0.75,'Galle':0.70,
    'Matara':0.70,'Kurunegala':0.60,'Colombo':0.55,'Gampaha':0.55,
    'Hambantota':0.45,'Trincomalee':0.50,'Batticaloa':0.50,
    'Ampara':0.55,'Polonnaruwa':0.40,'Anuradhapura':0.40,
    'Puttalam':0.45,'Mannar':0.35,'Vavuniya':0.40,'Mullaitivu':0.45,
    'Kilinochchi':0.35,'Jaffna':0.30,'Monaragala':0.65
}
DISTRICTS = list(terrain.keys())

# ── Header ────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>🌧️ Sri Lanka Flood & Landslide Early Warning</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem;'>"
    "AI-powered 3D risk predictions updated in real time using NASA GPM satellite data.</p>",
    unsafe_allow_html=True
)

# ── Sidebar inputs ────────────────────────────────────────
st.sidebar.header("Rainfall Inputs (mm)")
st.sidebar.markdown("Enter rainfall for Sri Lanka this month and previous months.")

rain_now   = st.sidebar.slider("This month (mm)",   0, 600, 150)
rain_prev1 = st.sidebar.slider("Last month (mm)",   0, 600, 120)
rain_prev2 = st.sidebar.slider("2 months ago (mm)", 0, 600, 100)
rain_prev3 = st.sidebar.slider("3 months ago (mm)", 0, 600, 80)
month_num  = st.sidebar.selectbox("Current month", range(1,13),
                                   format_func=lambda x: [
                                       'Jan','Feb','Mar','Apr','May','Jun',
                                       'Jul','Aug','Sep','Oct','Nov','Dec'
                                   ][x-1], index=10)

# Monthly average rainfall for anomaly calculation
monthly_avg = {1:100,2:73,3:82,4:139,5:128,6:62,
               7:54,8:71,9:111,10:249,11:308,12:237}

# ── Predict for all districts ─────────────────────────────
rows = []
for district in DISTRICTS:
    tr = terrain[district]
    rows.append({
        'district':       district,
        'rainfall_mm':    rain_now,
        'rain_prev1':     rain_prev1,
        'rain_prev2':     rain_prev2,
        'rain_prev3':     rain_prev3,
        'rain_3mo':       rain_now + rain_prev1 + rain_prev2,
        'rain_6mo':       rain_now + rain_prev1 + rain_prev2 + rain_prev3,
        'rain_anomaly':   rain_now - monthly_avg.get(month_num, 150),
        'rain_x_terrain': rain_now * tr,
        'terrain_risk':   tr,
        'month_sin':      np.sin(2 * np.pi * month_num / 12),
        'month_cos':      np.cos(2 * np.pi * month_num / 12),
        'is_monsoon':     int(month_num in [4,5,10,11])
    })

pred_df = pd.DataFrame(rows)
pred_df['risk_score'] = model.predict_proba(
    pred_df[FEATURES])[:,1]
pred_df['alert'] = pred_df['risk_score'].apply(
    lambda x: '🔴 HIGH' if x >= 0.6 else ('🟡 MEDIUM' if x >= 0.35 else '🟢 LOW')
)

# ── Metrics (Moved to top for mobile UX) ──────────────────
high   = (pred_df['risk_score'] >= 0.6).sum()
medium = ((pred_df['risk_score'] >= 0.35) & 
          (pred_df['risk_score'] < 0.6)).sum()
low    = (pred_df['risk_score'] < 0.35).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 High Risk",   high)
c2.metric("🟡 Medium Risk", medium)
c3.metric("🟢 Low Risk",    low)
c4.metric("Model AUC-ROC", "0.822")

st.divider()

# ── Layout: map + table ───────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Interactive 3D Risk Map")

    lka2 = lka.copy()
    lka2.columns = [c.upper() for c in lka2.columns]
    lka2 = lka2.set_geometry("GEOMETRY")
    lka2 = lka2.merge(pred_df[['district','risk_score']],
                      left_on='NAME_1', right_on='district', how='left')
    lka2['risk_score'] = lka2['risk_score'].fillna(0.2)
    
    # Calculate 3D elevation and color for PyDeck
    lka2['elevation'] = (lka2['risk_score'] ** 2) * 150000  # Exponential scale for dramatic 3D effect
    # Create Neon Cyberpunk colors based on risk
    lka2['fill_color'] = lka2['risk_score'].apply(
        lambda x: [255, 51, 102, 220] if x >= 0.6 else ([255, 204, 0, 200] if x >= 0.35 else [0, 229, 255, 180])
    )

    geojson = json.loads(lka2.to_json())
    
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="properties.elevation",
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255, 50],
        pickable=True,
        auto_highlight=True # Interactive glowing hover effect
    )

    view_state = pdk.ViewState(latitude=7.8, longitude=80.7, zoom=6.5, pitch=55, bearing=15)
    
    m = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "{NAME_1}\nRisk Score: {risk_score}"}
    )
    st.pydeck_chart(m, use_container_width=True)

with col2:
    st.subheader("District Risk Scores")
    display = pred_df[['district','risk_score','alert']]\
        .sort_values('risk_score', ascending=False)\
        .reset_index(drop=True)
    display.columns = ['District','Risk Score','Alert Level']
    display['Risk Score'] = display['Risk Score'].round(3)
    st.dataframe(display, use_container_width=True, height=500, hide_index=True)

st.divider()
st.caption(
    "Model: XGBoost trained on NASA GPM IMERG rainfall (2000–2023) + "
    "DesInventar disaster records (1974–2020). "
    "AUC-ROC: 0.822 | Test period: 2018–2020 | "
    "MADE BY HASIRU CHAMIKA"
)