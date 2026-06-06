# app.py — Sri Lanka Flood & Landslide Early Warning System

import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
import geopandas as gpd
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Sri Lanka Flood Early Warning",
    page_icon="🌧️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_v2.pkl")

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
st.title("🌧️ Sri Lanka Flood & Landslide Early Warning")
st.markdown(
    "Enter current and recent rainfall to predict flood/landslide risk "
    "across all 25 districts. Built using NASA GPM satellite data + "
    "50 years of DesInventar disaster records."
)

st.divider()

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

# ── Layout: map + table ───────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("District Risk Map")

    lka2 = lka.copy()
    lka2.columns = [c.upper() for c in lka2.columns]
    lka2 = lka2.set_geometry("GEOMETRY")
    lka2 = lka2.merge(pred_df[['district','risk_score']],
                      left_on='NAME_1', right_on='district', how='left')
    lka2['risk_score'] = lka2['risk_score'].fillna(0.2)

    m = folium.Map(location=[7.9, 80.7], zoom_start=7,
                   tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=lka2.__geo_interface__,
        data=lka2,
        columns=['NAME_1','risk_score'],
        key_on='feature.properties.NAME_1',
        fill_color='RdYlGn_r',
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name='Risk Score (0–1)',
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ).add_to(m)

    for _, row in lka2.iterrows():
        if row.GEOMETRY:
            c = row.GEOMETRY.centroid
            folium.CircleMarker(
                location=[c.y, c.x], radius=4,
                color='black', fill=True,
                tooltip=f"{row['NAME_1']}: {row['risk_score']:.2f}"
            ).add_to(m)

    st_folium(m, width=520, height=500)

with col2:
    st.subheader("District Risk Scores")
    display = pred_df[['district','risk_score','alert']]\
        .sort_values('risk_score', ascending=False)\
        .reset_index(drop=True)
    display.columns = ['District','Risk Score','Alert Level']
    display['Risk Score'] = display['Risk Score'].round(3)
    st.dataframe(display, use_container_width=True, height=500)

# ── Metrics ───────────────────────────────────────────────
st.divider()
high   = (pred_df['risk_score'] >= 0.6).sum()
medium = ((pred_df['risk_score'] >= 0.35) & 
          (pred_df['risk_score'] < 0.6)).sum()
low    = (pred_df['risk_score'] < 0.35).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 High Risk Districts",   high)
c2.metric("🟡 Medium Risk Districts", medium)
c3.metric("🟢 Low Risk Districts",    low)
c4.metric("Model AUC-ROC", "0.822")

st.divider()
st.caption(
    "Model: XGBoost trained on NASA GPM IMERG rainfall (2000–2023) + "
    "DesInventar disaster records (1974–2020). "
    "AUC-ROC: 0.822 | Test period: 2018–2020 | "
    "Built by [Your Name] · github.com/YOUR_USERNAME/srilanka-flood-warning"
)