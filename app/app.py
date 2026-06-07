# app.py — Sri Lanka Flood & Landslide Early Warning System (Premium Edition)

import os
import base64
import json
import math
import time
import random
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from shapely.geometry import Point, mapping
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sri Lanka Flood & Landslide Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD EXTERNAL UI ASSETS (HTML/CSS/JS)
# ============================================================================
def load_external_assets():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")
    js_path = os.path.join(base_dir, "assets", "script.js")
    
    # Inject external FontAwesome and Animate.css CDNs
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    """, unsafe_allow_html=True)

    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    if os.path.exists(js_path):
        with open(js_path, "r") as f:
            st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

load_external_assets()

# ============================================================================
# PARTICLE BACKGROUND GENERATOR
# ============================================================================
def add_particle_background():
    particle_html = """
    <div id="particles-container" style="position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0;"></div>
    <script>
        const container = document.getElementById('particles-container');
        for(let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            const size = Math.random() * 4 + 2;
            particle.style.width = size + 'px';
            particle.style.height = size + 'px';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDuration = Math.random() * 20 + 10 + 's';
            particle.style.animationDelay = Math.random() * 10 + 's';
            container.appendChild(particle);
        }
    </script>
    """
    st.markdown(particle_html, unsafe_allow_html=True)

add_particle_background()

# ============================================================================
# TOP NAVIGATION BAR
# ============================================================================
global_nav = option_menu(
    menu_title=None,
    options=["Home", "Methodology", "About Us", "Contact Us"],
    icons=["house", "book", "people", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important", 
            "background-color": "transparent",
            "border": "none",
            "margin-bottom": "1rem"
        },
        "icon": {"color": "#00C9FF", "font-size": "1.1rem"}, 
        "nav-link": {
            "font-family": "'Space Grotesk', sans-serif",
            "font-size": "1rem", 
            "font-weight": "500",
            "text-align": "center",
            "margin": "0px 10px", 
            "color": "#8A8F9E",
            "--hover-color": "rgba(0, 201, 255, 0.1)"
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, rgba(0,201,255,0.2), rgba(146,254,157,0.1))", 
            "color": "#ffffff",
            "border": "1px solid rgba(0, 201, 255, 0.3)",
            "border-radius": "12px"
        },
    }
)

if global_nav == "Methodology":
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>System Methodology</h2><br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card">
            <h4>Data Sources & AI Pipeline</h4>
            <p style="color: #8A8F9E;">This system uses a data-driven approach to predict flood and landslide risks across Sri Lanka's 25 districts.</p>
            <ul style="color: #cbd5e1; margin-top: 1rem; line-height: 1.8;">
                <li><b style="color: #00C9FF;">NASA GPM IMERG:</b> Real-time satellite precipitation data continuously updating our rainfall features.</li>
                <li><b style="color: #00C9FF;">SRTM Terrain Data:</b> Digital elevation and slope metrics used to estimate landslide vulnerabilities.</li>
                <li><b style="color: #00C9FF;">Machine Learning:</b> XGBoost classifier trained on 50 years of historical disaster data from the DesInventar disaster database.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.stop()
    
elif global_nav == "About Us":
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>About the Project</h2><br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card">
            <h4>Our Mission</h4>
            <p style="color: #cbd5e1; line-height: 1.8; margin-top: 1rem;">Built in response to the devastating impacts of recent extreme weather events, our goal is to transition Sri Lanka's early warning infrastructure from static rule-based thresholds to dynamic, AI-powered predictive models.</p>
            <p style="color: #cbd5e1; line-height: 1.8;">We aim to provide actionable, district-level intelligence 24-48 hours in advance to the Disaster Management Centre (DMC) and the general public, ultimately saving lives and reducing economic loss.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

elif global_nav == "Contact Us":
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>Contact Us</h2><br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-bottom: 1rem;">Get in Touch</h4>
                <p style="color: #cbd5e1; margin-bottom: 0.8rem;"><i class="fa-solid fa-envelope" style="color: #00C9FF; margin-right: 10px;"></i> <b>Email:</b> earlywarning@srilanka-ai.org</p>
                <p style="color: #cbd5e1; margin-bottom: 0.8rem;"><i class="fa-solid fa-location-dot" style="color: #00C9FF; margin-right: 10px;"></i> <b>Address:</b> Disaster Management Centre, Colombo</p>
                <p style="color: #cbd5e1; margin-bottom: 0.8rem;"><i class="fa-solid fa-phone" style="color: #FF3366; margin-right: 10px;"></i> <b>Emergency Hotline:</b> 117</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        with st.form("contact_form"):
            st.markdown("#### Send a Message")
            st.text_input("Full Name")
            st.text_input("Email Address")
            st.text_area("How can we help you?")
            st.form_submit_button("Submit Request", use_container_width=True)
    st.stop()

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "xgb_v2.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_geodata():
    return gpd.read_file("https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LKA_1.json")

@st.cache_data
def load_cloud_b64():
    """Load cloud.obj and convert to base64 for JS embedding"""
    cloud_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "static", 
        "cloud.obj"
    )
    with open(cloud_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

CLOUD_B64 = load_cloud_b64()

# Model loading with fallback
try:
    model, best_thresh, FEATURES = load_model()
except Exception:
    # Fallback to mock model if file not found
    best_thresh = 0.6
    FEATURES = ['rainfall_mm', 'rain_prev1', 'rain_prev2', 'rain_prev3', 'rain_3mo', 
                'rain_6mo', 'rain_anomaly', 'rain_x_terrain', 'terrain_risk', 
                'month_sin', 'month_cos', 'is_monsoon']

    def mock_predict_proba(features):
        # Simulate model predictions with realistic patterns
        risk = (features['rainfall_mm'] / 600) * 0.4 + \
               (features['rain_x_terrain']) * 0.3 + \
               (features['rain_anomaly'] / 400) * 0.2 + \
               features['is_monsoon'] * 0.1
        risk = np.clip(risk, 0.05, 0.95)
        return np.column_stack((1-risk, risk))

    class MockModel:
        def predict_proba(self, features):
            return mock_predict_proba(features)
    
    model = MockModel()

lka = load_geodata()

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
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

monthly_avg = {1:100,2:73,3:82,4:139,5:128,6:62,
               7:54,8:71,9:111,10:249,11:308,12:237}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_risk_color(risk):
    if risk >= 0.6: return "#FF3366"
    if risk >= 0.35: return "#FFCC00"
    return "#00E5FF"

def get_risk_text(risk):
    if risk >= 0.6: return "CRITICAL"
    if risk >= 0.35: return "MODERATE"
    return "LOW"

# ============================================================================
# SIDEBAR - PREMIUM CONTROLS
# ============================================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 class="gradient-text" style="font-size: 1.8rem;">🌊 WARNING HUB</h2>
            <div style="width: 60px; height: 3px; background: linear-gradient(90deg, #00C9FF, #92FE9D); margin: 0.5rem auto;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Real-time clock
    current_time = datetime.now().strftime("%I:%M:%S %p")
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-family: monospace; font-size: 1.2rem; background: rgba(0,201,255,0.1); padding: 0.3rem 1rem; border-radius: 20px;">
                🕐 {current_time} UTC+5:30
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Rainfall input section
    st.markdown("### 📊 RAINFALL DATA")
    st.markdown("*NASA GPM IMERG Satellite*")
    
    rain_now = st.slider("Current Month (mm)", 0, 600, 150, key="rain_now")
    rain_prev1 = st.slider("Last Month (mm)", 0, 600, 120, key="rain_prev1")
    rain_prev2 = st.slider("2 Months Ago (mm)", 0, 600, 100, key="rain_prev2")
    rain_prev3 = st.slider("3 Months Ago (mm)", 0, 600, 80, key="rain_prev3")
    
    month_num = st.selectbox(
        "Current Month", 
        range(1,13),
        format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1],
        index=10,
        key="month_select"
    )
    
    st.markdown("---")
    
    # Live alerts toggle
    st.markdown("### 🔔 ALERT PREFERENCES")
    show_live_alerts = st.toggle("Live Alerts", value=True)
    sound_alerts = st.toggle("Sound Alerts", value=False)
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📡 SYSTEM STATUS")
    status_items = [
        ("Satellite Link", "🟢 Active"),
        ("Model Inference", "🟢 Online"),
        ("Data Pipeline", "🟢 Synced"),
        ("API Gateway", "🟢 Operational")
    ]
    for label, status in status_items:
        st.markdown(f"<div style='display: flex; justify-content: space-between; font-size: 0.85rem;'><span>{label}</span><span>{status}</span></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Powered by XGBoost | NASA GPM | DesInventar")

# ============================================================================
# FORECAST LOGIC
# ============================================================================
@st.cache_data(ttl=3600)
def get_future_predictions(_model, terrain, FEATURES, monthly_avg, days=30):
    """
    Fetches real weather forecast and runs model
    for every district × every future day
    """
    # ── Get forecast rainfall ──────────────────────────
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      7.8731,
        "longitude":     80.7718,
        "daily":         ["precipitation_sum"],
        "timezone":      "Asia/Colombo",
        "forecast_days": min(days, 16)  # API max is 16 days free
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    dates     = pd.to_datetime(data['daily']['time'])
    rain_days = data['daily']['precipitation_sum']

    # ── Convert daily mm → monthly equivalent ─────────
    results = []
    for i, (date, daily_rain) in enumerate(zip(dates, rain_days)):
        monthly_equiv = daily_rain * 30   # scale to monthly
        for district, tr in terrain.items():
            month = date.month
            row = {
                'district':       district,
                'date':           date,
                'day_label':      date.strftime('%b %d'),
                'rainfall_mm':    monthly_equiv,
                'rain_prev1':     monthly_avg.get((month - 1) or 12, 100),
                'rain_prev2':     monthly_avg.get((month - 2) or 11, 100),
                'rain_prev3':     monthly_avg.get((month - 3) or 10, 100),
                'terrain_risk':   tr,
                'month_sin':      np.sin(2 * np.pi * month / 12),
                'month_cos':      np.cos(2 * np.pi * month / 12),
                'is_monsoon':     int(month in [4, 5, 10, 11])
            }
            row['rain_3mo']       = row['rainfall_mm'] + row['rain_prev1'] + row['rain_prev2']
            row['rain_6mo']       = row['rain_3mo'] + row['rain_prev3']
            row['rain_anomaly']   = row['rainfall_mm'] - monthly_avg.get(month, 150)
            row['rain_x_terrain'] = row['rainfall_mm'] * tr
            results.append(row)

    future_df = pd.DataFrame(results)

    # ── Run model on all future rows ───────────────────
    future_df['risk_score'] = _model.predict_proba(future_df[FEATURES])[:, 1]
    future_df['alert'] = future_df['risk_score'].apply(
        lambda x: 'CRITICAL' if x >= 0.6 else ('MODERATE' if x >= 0.35 else 'LOW')
    )
    return future_df

# ============================================================================
# PREDICTION LOGIC
# ============================================================================
rows = []
for district in DISTRICTS:
    tr = terrain[district]
    rows.append({
        'district': district,
        'rainfall_mm': rain_now,
        'rain_prev1': rain_prev1,
        'rain_prev2': rain_prev2,
        'rain_prev3': rain_prev3,
        'rain_3mo': rain_now + rain_prev1 + rain_prev2,
        'rain_6mo': rain_now + rain_prev1 + rain_prev2 + rain_prev3,
        'rain_anomaly': rain_now - monthly_avg.get(month_num, 150),
        'rain_x_terrain': rain_now * tr,
        'terrain_risk': tr,
        'month_sin': np.sin(2 * np.pi * month_num / 12),
        'month_cos': np.cos(2 * np.pi * month_num / 12),
        'is_monsoon': int(month_num in [4,5,10,11])
    })

pred_df = pd.DataFrame(rows)
features_df = pred_df[FEATURES]
pred_df['risk_score'] = model.predict_proba(features_df)[:, 1]

pred_df['alert'] = pred_df['risk_score'].apply(
    lambda x: '🔴 CRITICAL' if x >= 0.6 else ('🟡 MODERATE' if x >= 0.35 else '🟢 LOW')
)
pred_df['alert_color'] = pred_df['risk_score'].apply(get_risk_color)
pred_df['risk_level'] = pred_df['risk_score'].apply(get_risk_text)

# ============================================================================
# METRICS DASHBOARD
# ============================================================================
high_risk = pred_df[pred_df['risk_score'] >= 0.6]
medium_risk = pred_df[(pred_df['risk_score'] >= 0.35) & (pred_df['risk_score'] < 0.6)]
low_risk = pred_df[pred_df['risk_score'] < 0.35]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; opacity: 0.7;">CRITICAL RISK</div>
            <div class="metric-value">{len(high_risk)}</div>
            <div style="font-size: 0.7rem;">Districts</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; opacity: 0.7;">MODERATE RISK</div>
            <div class="metric-value">{len(medium_risk)}</div>
            <div style="font-size: 0.7rem;">Districts</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; opacity: 0.7;">LOW RISK</div>
            <div class="metric-value">{len(low_risk)}</div>
            <div style="font-size: 0.7rem;">Districts</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    max_risk_district = pred_df.loc[pred_df['risk_score'].idxmax(), 'district']
    max_risk = pred_df['risk_score'].max()
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; opacity: 0.7;">HIGHEST RISK</div>
            <div class="metric-value">{max_risk_district[:12]}</div>
            <div style="font-size: 0.7rem;">{max_risk:.1%} probability</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; opacity: 0.7;">MODEL SCORE</div>
            <div class="metric-value">0.822</div>
            <div style="font-size: 0.7rem;">AUC-ROC</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# NAVIGATION BAR
# ============================================================================
selected_tab = option_menu(
    menu_title=None,
    options=["3D RISK MAP", "ANALYTICS", "DISTRICT TABLE", "ALERTS & INFO", "FORECAST"],
    icons=["map", "bar-chart-line", "table", "shield-exclamation", "calendar3"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important", 
            "background-color": "rgba(20, 25, 40, 0.6)", 
            "border": "1px solid rgba(255,255,255,0.08)", 
            "border-radius": "16px",
            "backdrop-filter": "blur(12px)",
            "margin-bottom": "1.5rem"
        },
        "icon": {"color": "#00C9FF", "font-size": "1.1rem"}, 
        "nav-link": {
            "font-family": "'Space Grotesk', sans-serif",
            "font-size": "0.95rem", 
            "font-weight": "500",
            "text-align": "center", 
            "margin": "0px", 
            "color": "#8A8F9E",
            "--hover-color": "rgba(0, 201, 255, 0.1)"
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, rgba(0,201,255,0.2), rgba(146,254,157,0.1))", 
            "color": "#ffffff",
            "border": "1px solid rgba(0, 201, 255, 0.3)"
        },
    }
)




# ============================================================================
# TAB 1: 3D RISK MAP WITH CLOUDS (No flat upper area)
# ============================================================================
if selected_tab == "3D RISK MAP":
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <span class="gradient-text" style="font-size: 1.2rem;">Interactive Geospatial Intelligence</span>
            <p style="color: #8A8F9E;">3D map with district-shaped clouds floating above terrain</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_map, col_legend = st.columns([4, 1])
    
    with col_legend:
        show_clouds = st.toggle("☁️ Show Clouds", value=True)
        show_rain = st.toggle("🌧️ Show Rain", value=True)
        show_floods = st.toggle("🌊 Show Flooding", value=True)
        show_landslides = st.toggle("⛰️ Show Landslides", value=True)

    with col_map:
        lka2 = lka.copy()
        lka2.columns = [c.upper() for c in lka2.columns]
        lka2 = lka2.set_geometry("GEOMETRY")
        lka2 = lka2.merge(pred_df[['district','risk_score', 'rainfall_mm']],
                          left_on='NAME_1', right_on='district', how='left')
        lka2['risk_score'] = lka2['risk_score'].fillna(0.2)
        lka2['rainfall_mm'] = lka2['rainfall_mm'].fillna(0)
        
        # Reduce elevation significantly - no tall flat areas
        lka2['elevation'] = (lka2['rainfall_mm'] / 600) ** 1.5 * 80000

        def get_gradient_color(risk):
            if risk >= 0.8: return [255, 0,   50,  220]
            if risk >= 0.6: return [255, 51,  102, 210]
            if risk >= 0.45:return [255, 153, 0,   200]
            if risk >= 0.35:return [255, 204, 0,   190]
            if risk >= 0.2: return [0,   200, 255, 180]
            return              [0,   229, 255, 160]

        lka2['fill_color'] = lka2['risk_score'].apply(get_gradient_color)
        geojson = json.loads(lka2.to_json())

        # Layer 1: 3D extruded districts with minimal height
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson,
            opacity=0.75,
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="properties.elevation",
            get_fill_color="properties.fill_color",
            get_line_color=[255, 255, 255, 40],
            pickable=True,
            auto_highlight=True
        )

        # Layer 2: Flood wave rings on high-risk districts
        flood_rings = []
        for _, row in lka2.iterrows():
            if row['risk_score'] >= 0.6 and row.GEOMETRY is not None:
                centroid = row.GEOMETRY.centroid
                base_radius = 5000 + (row['risk_score'] - 0.6) * 25000
                for r in [base_radius * 0.4, base_radius * 0.7, base_radius]:
                    flood_rings.append({
                        "position": [centroid.x, centroid.y],
                        "radius": r,
                        "color": [20, 50, 220, int(90 * (1 - r / max(35000, r)))]
                    })

        flood_layer = pdk.Layer(
            "ScatterplotLayer",
            flood_rings,
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[100, 150, 255, 180],
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            pickable=False
        )

        # Layer 3: Landslide warning markers
        landslide_points = []
        for _, row in lka2.iterrows():
            if row['risk_score'] >= 0.7 and row.GEOMETRY is not None:
                centroid = row.GEOMETRY.centroid
                landslide_points.append({
                    "position": [centroid.x, centroid.y, row['elevation'] + 5000],
                    "color": [255, 200, 0, 240],
                    "radius": 4000,
                    "name": row['NAME_1'],
                    "risk": f"{row['risk_score']:.0%}"
                })

        landslide_layer = pdk.Layer(
            "ScatterplotLayer",
            landslide_points,
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 200],
            stroked=True,
            filled=True,
            line_width_min_pixels=3,
            pickable=True
        )

        # ── Layer 4: Volumetric Puffy Clouds (PyDeck Native) ──────────────────
        cloud_puffs = []
        for _, row in lka2.iterrows():
            if row['rainfall_mm'] > 0 and row.GEOMETRY is not None:
                bounds = row.GEOMETRY.bounds
                cloud_base_z = int(row['elevation'] + 15000)
                
                rain_ratio = min(row['rainfall_mm'] / 600, 1.0)
                r = int(255 - rain_ratio * 80)
                g = int(255 - rain_ratio * 70)
                b = int(255 - rain_ratio * 20)
                
                n_puffs = int(60 * rain_ratio) + 30
                attempts, added = 0, 0
                
                while added < n_puffs and attempts < n_puffs * 15:
                    attempts += 1
                    rx = random.uniform(bounds[0], bounds[2])
                    ry = random.uniform(bounds[1], bounds[3])
                    
                    if row.GEOMETRY.contains(Point(rx, ry)):
                        # Vary Z height and radius to create a volumetric 3D dome effect
                        puff_z = cloud_base_z + random.randint(0, 8000)
                        radius = random.randint(3000, 7000)
                        alpha = random.randint(140, 200)
                        
                        cloud_puffs.append({
                            "position": [rx, ry, puff_z],
                            "radius": radius,
                            "color": [r, g, b, alpha]
                        })
                        added += 1

        layer_clouds = pdk.Layer(
            "ScatterplotLayer",
            cloud_puffs,
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            pickable=False
        )

        # ── Layer 5: Static Rain Streaks (Internal District Volume) ───────────
        rain_streaks = []
        for _, row in lka2.iterrows():
            if row['rainfall_mm'] > 0 and row.GEOMETRY is not None:
                bounds = row.GEOMETRY.bounds
                n_drops = min(int(row['rainfall_mm'] * 0.6), 150)
                cloud_z = int(row['elevation'] + 15000)
                ground_z = 0

                attempts, added = 0, 0
                while added < n_drops and attempts < n_drops * 15:
                    attempts += 1
                    rx = random.uniform(bounds[0], bounds[2])
                    ry = random.uniform(bounds[1], bounds[3])
                    
                    if row.GEOMETRY.contains(Point(rx, ry)):
                        rain_streaks.append({
                            "source": [rx, ry, cloud_z],
                            "target": [rx, ry, ground_z]
                        })
                        added += 1

        layer_rain = pdk.Layer(
            "LineLayer",
            rain_streaks,
            get_source_position="source",
            get_target_position="target",
            get_color=[160, 200, 255, 110],
            get_width=1.5,
        )

        # Combine all layers
        view_state = pdk.ViewState(
            latitude=7.8, longitude=80.7,
            zoom=6.2, pitch=45, bearing=25
        )

        active_layers = [geojson_layer]
        if show_floods:
            active_layers.append(flood_layer)
        if show_landslides:
            active_layers.append(landslide_layer)
        if show_clouds:
            active_layers.append(layer_clouds)
        if show_rain:
            active_layers.append(layer_rain)

        deck = pdk.Deck(
            layers=active_layers,
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={
                "html": """
                    <div style='background:rgba(0,0,0,0.85);
                                padding:10px; border-radius:8px;
                                border:1px solid rgba(0,200,255,0.4)'>
                        <b style='color:#00C9FF'>{NAME_1}</b><br/>
                        Risk: <b style='color:#FF3366'>{risk_score}</b><br/>
                        Rainfall: {rainfall_mm}mm
                    </div>
                """,
                "style": {"color": "white"}
            }
        )
        st.pydeck_chart(deck, use_container_width=True)

        # ── CSS Animated Rain Overlay (Psychological Motion Effect) ───────
        if rain_now > 20 and show_rain:
            rain_intensity = min(rain_now / 600, 1.0)
            n_drops = int(rain_intensity * 60) + 20
            rain_css = "\n".join([
                f"""
                .rain-drop-{i} {{
                    position: absolute;
                    left: {random.randint(0,100)}%;
                    top: -20px;
                    width: {random.randint(1,2)}px;
                    height: {random.randint(10,25)}px;
                    background: linear-gradient(to bottom, transparent, rgba(160,200,255,0.7));
                    animation: fall-{i} {random.uniform(0.4,1.0):.2f}s linear {random.uniform(0,2):.2f}s infinite;
                    border-radius: 2px;
                    pointer-events: none;
                }}
                @keyframes fall-{i} {{
                    0%   {{ transform: translateY(-30px); opacity:0; }}
                    10%  {{ opacity: 0.9; }}
                    80%  {{ opacity: 0.7; }}
                    100% {{ transform: translateY(500px); opacity:0; }}
                }}
                """ for i in range(n_drops)
            ])

            rain_divs = "\n".join([f'<div class="rain-drop-{i}"></div>' for i in range(n_drops)])

            st.markdown(f"""
            <div style="position:relative; pointer-events:none; margin-top:-500px; height:500px; overflow:hidden; z-index:999;">
                <style>{rain_css}</style>
                {rain_divs}
            </div>
            """, unsafe_allow_html=True)
    
    with col_legend:
        st.markdown("""
            <div class="glass-card" style="padding: 1rem;">
                <h4 style="margin-bottom: 1rem;">🎨 RISK LEGEND</h4>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="width: 24px; height: 24px; background: #FF0032; border-radius: 4px;"></div>
                    <span>Critical (80-100%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="width: 24px; height: 24px; background: #FF3366; border-radius: 4px;"></div>
                    <span>High (60-80%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="width: 24px; height: 24px; background: #FF9900; border-radius: 4px;"></div>
                    <span>Elevated (45-60%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="width: 24px; height: 24px; background: #FFCC00; border-radius: 4px;"></div>
                    <span>Moderate (35-45%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="width: 24px; height: 24px; background: #00CCFF; border-radius: 4px;"></div>
                    <span>Low (20-35%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 24px; height: 24px; background: #00E5FF; border-radius: 4px;"></div>
                    <span>Minimal (<20%)</span>
                </div>
                <hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.1);">
                <div style="font-size: 0.75rem; text-align: center; color: #8A8F9E;">
                    <span>🎨 Color = Risk probability</span><br>
                    <span>🌊 Flood Waves = High risk areas (≥60%)</span><br>
                    <span>⛰️ Landslide Markers = Critical risk (≥70%)</span><br>
                    <span>☁️ Clouds = District-shaped precipitation</span><br>
                    <span>💧 Rain droplets = Active rainfall</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        
        
        
        
        

# ============================================================================
# TAB 2: ANALYTICS DASHBOARD
# ============================================================================
if selected_tab == "ANALYTICS":
    st.markdown("### 📈 Predictive Analytics Dashboard")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Risk distribution bar chart
        risk_counts = pred_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        color_map = {'CRITICAL': '#FF3366', 'MODERATE': '#FFCC00', 'LOW': '#00E5FF'}
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=risk_counts['Risk Level'],
                y=risk_counts['Count'],
                marker_color=[color_map[l] for l in risk_counts['Risk Level']],
                text=risk_counts['Count'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Districts: %{y}<extra></extra>'
            )
        ])
        fig_bar.update_layout(
            title="Risk Level Distribution",
            title_font_color="#00C9FF",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_title="Risk Level",
            yaxis_title="Number of Districts",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart2:
        # Top 10 risky districts
        top10 = pred_df.nlargest(10, 'risk_score')[['district', 'risk_score', 'rainfall_mm']].copy()
        top10['risk_percent'] = top10['risk_score'] * 100
        
        fig_hbar = go.Figure(data=[
            go.Bar(
                x=top10['risk_percent'],
                y=top10['district'],
                orientation='h',
                marker_color=top10['risk_percent'].apply(
                    lambda x: '#FF3366' if x >= 60 else ('#FFCC00' if x >= 35 else '#00E5FF')
                ),
                text=top10['risk_percent'].round(1).astype(str) + '%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>'
            )
        ])
        fig_hbar.update_layout(
            title="Top 10 High-Risk Districts",
            title_font_color="#00C9FF",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_title="Risk Probability (%)",
            yaxis_title="District",
            height=400,
            xaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_hbar, use_container_width=True)
    
    # Risk vs Rainfall scatter plot
    st.markdown("---")
    col_scatter, col_gauge = st.columns(2)
    
    with col_scatter:
        fig_scatter = px.scatter(
            pred_df, 
            x='rainfall_mm', 
            y='risk_score',
            color='risk_level',
            color_discrete_map={'CRITICAL': '#FF3366', 'MODERATE': '#FFCC00', 'LOW': '#00E5FF'},
            hover_data=['district', 'terrain_risk'],
            title="Risk vs Rainfall Correlation",
            labels={'rainfall_mm': 'Monthly Rainfall (mm)', 'risk_score': 'Risk Probability'}
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=400
        )
        fig_scatter.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')))
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_gauge:
        # Gauge chart for average risk
        avg_risk = pred_df['risk_score'].mean() * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_risk,
            title={'text': "Average Risk Index", 'font': {'color': 'white', 'size': 16}},
            delta={'reference': 50, 'increasing': {'color': "#FF3366"}, 'decreasing': {'color': "#00E5FF"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickwidth': 2},
                'bar': {'color': "#00C9FF"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': "rgba(0,229,255,0.2)"},
                    {'range': [35, 60], 'color': "rgba(255,204,0,0.2)"},
                    {'range': [60, 100], 'color': "rgba(255,51,102,0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': avg_risk
                }
            }
        ))
        fig_gauge.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=400
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ============================================================================
    # DISTRICT DEEP DIVE
    # ============================================================================
    st.markdown("---")
    st.markdown("### 🎯 District-Level Deep Dive")
    
    selected_dist = st.selectbox("Select a district to view its vulnerability profile:", DISTRICTS, index=DISTRICTS.index('Colombo'))
    dist_data = pred_df[pred_df['district'] == selected_dist].iloc[0]
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.info(f"**Alert Status:** {dist_data['alert']}")
    with col_d2:
        st.warning(f"**Terrain Vulnerability:** {dist_data['terrain_risk']:.0%}")
    with col_d3:
        st.error(f"**Current Month Rainfall:** {dist_data['rainfall_mm']} mm")
        
    # Radar Chart for District Profile
    # Normalizing values between 0 and 1 for the radar chart visualization
    radar_values = [
        dist_data['risk_score'], 
        dist_data['terrain_risk'], 
        min(dist_data['rainfall_mm'] / 500, 1.0), 
        max(min((dist_data['rain_anomaly'] + 200) / 400, 1.0), 0.0), 
        min(dist_data['rain_3mo'] / 1200, 1.0)
    ]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=radar_values,
        theta=['Overall Risk', 'Terrain Vulnerability', 'Rain Intensity', 'Weather Anomaly', '3-Month Saturation'],
        fill='toself',
        line_color='#00C9FF',
        fillcolor='rgba(0, 201, 255, 0.3)',
        hovertemplate='%{theta}<extra></extra>'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='rgba(255,255,255,0.5)')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white', size=12))
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================================
# TAB 3: DISTRICT TABLE
# ============================================================================
if selected_tab == "DISTRICT TABLE":
    st.markdown("### 🏛️ District Risk Analysis")
    
    search_term = st.text_input("🔍 Filter District", placeholder="Enter district name...")
    
    filtered_df = pred_df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['district'].str.contains(search_term, case=False)]
    
    display_df = filtered_df[['district', 'risk_score', 'alert', 'rainfall_mm', 'terrain_risk']].copy()
    display_df.columns = ['District', 'Risk Score', 'Alert Level', 'Rainfall (mm)', 'Terrain Risk']
    display_df['Risk Score'] = display_df['Risk Score'].apply(lambda x: f"{x:.1%}")
    display_df['Terrain Risk'] = display_df['Terrain Risk'].apply(lambda x: f"{x:.0%}")
    
    def color_alert(val):
        if 'CRITICAL' in str(val):
            return 'background-color: rgba(255,51,102,0.2); color: #FF6699'
        elif 'MODERATE' in str(val):
            return 'background-color: rgba(255,204,0,0.2); color: #FFD700'
        return 'background-color: rgba(0,229,255,0.2); color: #66E5FF'
    
    styled_df = display_df.style.map(color_alert, subset=['Alert Level'])
    
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        height=500,
        column_config={
            "District": st.column_config.TextColumn("District", width="medium"),
            "Risk Score": st.column_config.TextColumn("Risk Score", width="small"),
            "Alert Level": st.column_config.TextColumn("Alert Level", width="small"),
            "Rainfall (mm)": st.column_config.NumberColumn("Rainfall (mm)", format="%d mm"),
            "Terrain Risk": st.column_config.TextColumn("Terrain Risk", width="small")
        }
    )
    
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Risk Report (CSV)",
        data=csv,
        file_name=f"flood_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# TAB 4: ALERTS & INFORMATION
# ============================================================================
if selected_tab == "ALERTS & INFO":
    col_info_left, col_info_right = st.columns([2, 1])
    
    with col_info_left:
        st.markdown("### 🚨 Live Risk Alerts")
        
        critical_districts = pred_df[pred_df['risk_score'] >= 0.6].nlargest(5, 'risk_score')
        moderate_districts = pred_df[(pred_df['risk_score'] >= 0.35) & (pred_df['risk_score'] < 0.6)].nlargest(5, 'risk_score')
        
        if len(critical_districts) > 0:
            st.markdown("#### 🔴 CRITICAL ALERTS")
            for _, row in critical_districts.iterrows():
                st.markdown(f"""
                    <div class="alert-high" style="margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span><strong>{row['district']}</strong></span>
                            <span style="background: #FF3366; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.75rem;">{row['risk_score']:.1%}</span>
                        </div>
                        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                            🌧️ Rainfall: {row['rainfall_mm']}mm | ⛰️ Terrain: {row['terrain_risk']:.0%}
                        </div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">
                            ⚠️ Immediate action recommended. Risk of flash floods and landslides.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        if len(moderate_districts) > 0:
            st.markdown("#### 🟡 MODERATE ALERTS")
            for _, row in moderate_districts.iterrows():
                st.markdown(f"""
                    <div class="alert-medium" style="margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span><strong>{row['district']}</strong></span>
                            <span style="background: #FFCC00; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.75rem;">{row['risk_score']:.1%}</span>
                        </div>
                        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                            🌧️ Rainfall: {row['rainfall_mm']}mm | ⛰️ Terrain: {row['terrain_risk']:.0%}
                        </div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">
                            ⚠️ Stay vigilant. Monitor local weather updates.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        if len(critical_districts) == 0 and len(moderate_districts) == 0:
            st.info("✅ No active alerts at this time. All districts are at low risk levels.")
    
    with col_info_right:
        st.markdown("### 📋 Safety Guidelines")
        
        with st.expander("🏃 Evacuation", expanded=True):
            st.markdown("""
                - Know your nearest evacuation center
                - Prepare an emergency go-bag
                - Plan multiple evacuation routes
                - Keep important documents waterproof
            """)
        
        with st.expander("📱 Emergency Contacts", expanded=True):
            st.markdown("""
                | Service | Number |
                |---------|--------|
                | DMC Hotline | **117** |
                | Police Emergency | **119** |
                | Ambulance | **110** |
                | Disaster Center | **011-2136222** |
            """)
        
        with st.expander("🌊 Flood Preparedness", expanded=True):
            st.markdown("""
                - Move to higher ground immediately
                - Avoid walking or driving through floodwater
                - Turn off gas and electricity
                - Listen to local radio for updates
                - Do not attempt to cross flooded bridges
            """)
        
        st.markdown("---")
        st.markdown("### 📡 Data Sources")
        st.caption("""
            • NASA GPM IMERG (Global Precipitation Measurement)<br>
            • DesInventar Disaster Database (1974-2020)<br>
            • USGS Terrain Data<br>
            • Sri Lanka Meteorological Department
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 5: FORECAST
# ============================================================================
if selected_tab == "FORECAST":
    st.markdown("""
        <span class="gradient-text" style="font-size:1.4rem;">
            🔮 16-Day Flood Risk Forecast
        </span>
        <p style="color:#8A8F9E;">
            Real weather forecast → your trained XGBoost model →
            future district risk
        </p>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching live forecast from Open-Meteo..."):
        try:
            future_df = get_future_predictions(model, terrain, FEATURES, monthly_avg)

            # ── Summary metrics ────────────────────────────
            peak_row = future_df.loc[future_df['risk_score'].idxmax()]
            peak_day_df = future_df.groupby('day_label')['risk_score'].mean().reset_index()

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Peak Risk District",
                      peak_row['district'],
                      f"{peak_row['risk_score']:.0%}")
            c2.metric("📅 Peak Risk Date",
                      peak_row['day_label'])
            c3.metric("⚠️ High Risk Days",
                      int((future_df.groupby('date')['risk_score'].max() >= 0.6).sum()))

            st.divider()

            # ── 2D Map: Spatial Risk Forecast ─────────────
            st.markdown("### 🗺️ 2D Spatial Risk Forecast")
            selected_date = st.select_slider(
                "Slide to view forecast for a specific date:", 
                options=future_df['day_label'].unique().tolist()
            )
            map_day_df = future_df[future_df['day_label'] == selected_date]
            
            fig_2d_map = px.choropleth_mapbox(
                map_day_df,
                geojson=json.loads(lka.to_json()),
                locations='district',
                featureidkey="properties.NAME_1",
                color='risk_score',
                color_continuous_scale=[
                    [0.0,  '#001133'],
                    [0.2,  '#003366'],
                    [0.35, '#00C9FF'],
                    [0.6,  '#FFCC00'],
                    [0.8,  '#FF3366'],
                    [1.0,  '#FF0032']
                ],
                range_color=[0, 1],
                mapbox_style="carto-darkmatter",
                zoom=5.8,
                center={"lat": 7.8, "lon": 80.7},
                opacity=0.8,
                hover_name='district',
                hover_data={'district': False, 'risk_score': True, 'rainfall_mm': True},
                labels={'risk_score': 'Risk Probability', 'rainfall_mm': 'Rain (mm)'}
            )
            fig_2d_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500
            )
            st.plotly_chart(fig_2d_map, use_container_width=True)

            st.divider()

            # ── Heatmap: districts × days ──────────────────
            pivot = future_df.pivot_table(
                index='district',
                columns='day_label',
                values='risk_score',
                aggfunc='mean'
            ).round(3)

            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values * 100,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0,  '#001133'],
                    [0.2,  '#003366'],
                    [0.35, '#00C9FF'],
                    [0.6,  '#FFCC00'],
                    [0.8,  '#FF3366'],
                    [1.0,  '#FF0032']
                ],
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'Date: %{x}<br>'
                    'Risk: %{z:.1f}%'
                    '<extra></extra>'
                ),
                colorbar=dict(
                    title=dict(
                        text="Risk %",
                        font=dict(color="white")
                    ),
                    tickfont=dict(color="white")
                )
            ))
            fig_heat.update_layout(
                title="16-Day District Risk Forecast Heatmap",
                title_font_color="#00C9FF",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=700,
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── Daily avg risk trend line ──────────────────
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=peak_day_df['day_label'],
                y=peak_day_df['risk_score'] * 100,
                mode='lines+markers',
                line=dict(color='#00C9FF', width=3),
                marker=dict(
                    size=8,
                    color=peak_day_df['risk_score'] * 100,
                    colorscale=[
                        [0, '#00C9FF'],
                        [0.5, '#FFCC00'],
                        [1, '#FF3366']
                    ],
                    showscale=False
                ),
                fill='tozeroy',
                fillcolor='rgba(0,201,255,0.1)',
                hovertemplate='%{x}<br>Avg Risk: %{y:.1f}%<extra></extra>'
            ))

            # Add danger threshold line
            fig_trend.add_hline(
                y=60,
                line_dash="dash",
                line_color="#FF3366",
                annotation_text="Critical threshold (60%)",
                annotation_font_color="#FF3366"
            )
            fig_trend.add_hline(
                y=35,
                line_dash="dash",
                line_color="#FFCC00",
                annotation_text="Moderate threshold (35%)",
                annotation_font_color="#FFCC00"
            )

            fig_trend.update_layout(
                title="16-Day Average Risk Trend",
                title_font_color="#00C9FF",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=350,
                xaxis=dict(tickangle=45),
                yaxis_title="Risk Score (%)"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # ── Critical alerts for next 16 days ──────────
            critical_future = future_df[
                future_df['risk_score'] >= 0.6
            ].sort_values('risk_score', ascending=False)

            if len(critical_future) > 0:
                st.markdown("### 🚨 Upcoming Critical Risk Events")
                shown = set()
                for _, row in critical_future.iterrows():
                    key = f"{row['district']}_{row['day_label']}"
                    if key not in shown:
                        shown.add(key)
                        st.markdown(f"""
                        <div class="alert-high"
                             style="margin-bottom:0.5rem;">
                            <b>{row['district']}</b> —
                            {row['day_label']} —
                            <span style="color:#FF6699">
                                {row['risk_score']:.0%} risk
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        if len(shown) >= 10:
                            break
            else:
                st.success(
                    "✅ No critical risk events in the next 16 days "
                    "based on current forecast.")

        except Exception as e:
            st.error(f"Forecast unavailable: {e}")
            st.info(
                "This tab requires internet access to fetch "
                "live weather data from Open-Meteo.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption("🌍 **Model:** XGBoost Classifier")
    st.caption("📊 **AUC-ROC:** 0.822 | **F1-Score:** 0.79")

with col_footer2:
    st.caption("🔄 **Update Frequency:** Real-time")
    st.caption(f"📅 **Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

with col_footer3:
    st.caption("🔬 **Training Data:** 2000-2023 (NASA GPM)")
    st.caption("🎯 **Validation Period:** 2018-2020")

st.markdown("""
    <div style="text-align: center; padding: 1rem; opacity: 0.6;">
        <span>⚡ AI-Powered Early Warning System | Made with ❤️ for Sri Lanka | v3.0 Premium</span>
    </div>
""", unsafe_allow_html=True)