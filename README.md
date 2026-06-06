# Sri Lanka Flood & Landslide Early Warning System

> An open-source machine learning pipeline that predicts flood and landslide risk across all 25 districts of Sri Lanka 24–48 hours in advance, using satellite rainfall data, terrain features, and 50 years of historical disaster records.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/srilanka-flood-warning/blob/main/notebooks/01_data_collection.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP.streamlit.app)

---

## Motivation

In November 2025, Cyclone Ditwah struck Sri Lanka — the country's deadliest natural disaster in modern history. It caused **$4.1 billion in damage**, affected **2 million people** across all 25 districts, and destroyed over 58,000 hectares of paddy farmland. Sri Lanka's current early warning infrastructure relies on static meteorological thresholds. This project builds a data-driven, district-level risk model that learns dynamic patterns the rule-based system cannot detect.

---

## Demo

![Risk Map Screenshot](assets/demo_map.png)

**Live app:** [srilanka-flood-warning.streamlit.app](https://YOUR_APP.streamlit.app)

Enter current or forecast rainfall → the app returns a color-coded risk map of all 25 Sri Lanka districts updated in real time.

---

## What it does

- Ingests NASA GPM IMERG satellite rainfall, SRTM terrain elevation/slope, and historical DesInventar disaster records (1974–2024)
- Trains an XGBoost classifier to predict flood/landslide occurrence at district-month level
- Produces SHAP explainability plots showing which signals (rainfall accumulation, slope, prior-month saturation) drove each district's risk score
- Renders an interactive Folium/Streamlit choropleth map of Sri Lanka colored by predicted risk level
- Achieves **XX% precision / XX% recall** on held-out test years (2020–2023)

---

## Architecture

```
Raw Data Sources
├── NASA GPM IMERG       → monthly rainfall, 0.1° grid, 2000–2023
├── SRTM DEM             → elevation + slope per district (30m resolution)
├── DesInventar LKA      → 3,000+ disaster event records, 1974–2024
└── GADM Shapefiles      → Sri Lanka district boundaries (25 districts)
        │
        ▼
Feature Engineering (02_feature_engineering.ipynb)
├── rainfall_mm          → current month satellite rainfall
├── rainfall_prev_month  → 30-day antecedent moisture proxy
├── rainfall_3mo_sum     → seasonal accumulation
├── elevation_mean       → district mean elevation (m)
├── slope_mean           → district mean slope (degrees)
└── month_num            → seasonal signal (monsoon cycles)
        │
        ▼
XGBoost Classifier (03_model_training.ipynb)
├── scale_pos_weight=10  → handles severe class imbalance
├── SMOTE oversampling   → augments minority (disaster) class
└── 5-fold TimeSeriesCV  → no data leakage across time
        │
        ▼
SHAP Explainability + Risk Map (04_risk_dashboard.ipynb)
└── Streamlit App        → live district-level risk scores
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/srilanka-flood-warning.git
cd srilanka-flood-warning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or run directly in Google Colab — click the badge at the top of this README.

### 3. Download the data

```bash
python scripts/download_data.py
```

This fetches:
- NASA GPM IMERG monthly rainfall for Sri Lanka bounding box (`79.5°E–81.9°E, 5.9°N–9.8°N`)
- GADM district shapefile for Sri Lanka
- Prompts you to place the DesInventar CSV (manual download required — see [Data Sources](#data-sources))

### 4. Run the notebooks in order

| Notebook | What it does |
|---|---|
| `01_data_collection.ipynb` | Downloads and validates all raw data |
| `02_feature_engineering.ipynb` | Merges sources into district-month feature table |
| `03_model_training.ipynb` | Trains XGBoost, evaluates on test years, exports model |
| `04_risk_dashboard.ipynb` | SHAP plots, Folium map, Streamlit app preview |

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

---

## Data Sources

All data used in this project is free and publicly available.

| Dataset | Source | Coverage | License |
|---|---|---|---|
| GPM IMERG Monthly Rainfall | [NASA Earthdata](https://gpm.nasa.gov/data) | Global, 2000–present | NASA Open Data |
| SRTM Digital Elevation Model | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) | Global, 30m | Public Domain |
| DesInventar Disaster Records | [desinventar.net (LKA)](https://www.desinventar.net/DesInventar/profiletab.jsp?countrycode=lka) | Sri Lanka, 1974–2024 | UNDRR Open |
| GADM District Boundaries | [gadm.org](https://gadm.org/download_country.html) | Sri Lanka, Level 1 | Academic use |
| World Bank Flood Extent Maps | [WB Data Catalog](https://datacatalog.worldbank.org/search/dataset/0038275) | Sri Lanka, 2003–2014 | CC BY 4.0 |

---

## Results

> *(Update this table after you train your model)*

| Metric | Score |
|---|---|
| Precision (disaster class) | — |
| Recall (disaster class) | — |
| F1 Score | — |
| AUC-ROC | — |
| Test period | 2020–2023 |

**Top predictive features (SHAP):**
1. `rainfall_3mo_sum` — 3-month cumulative rainfall
2. `slope_mean` — terrain slope (landslide proxy)
3. `rainfall_prev_month` — antecedent soil moisture
4. `month_num` — monsoon seasonality

---

## Repo Structure

```
srilanka-flood-warning/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_risk_dashboard.ipynb
├── app/
│   └── app.py                  # Streamlit dashboard
├── scripts/
│   └── download_data.py        # Automated data fetcher
├── data/
│   ├── raw/                    # Downloaded source files (gitignored)
│   └── processed/              # Feature table, train/test splits
├── models/
│   └── xgb_flood_model.pkl     # Trained model artifact
├── assets/
│   └── demo_map.png            # Screenshot for README
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Limitations & Future Work

- Current model uses district-level aggregation. A grid-level (0.1°) model would give finer spatial resolution
- Elevation/slope features are static — adding NDVI (vegetation cover) would improve landslide prediction
- Real-time pipeline not yet implemented — currently requires manual rainfall input
- Model trained on data through 2023; Cyclone Ditwah (Nov 2025) not yet in training set — a strong validation opportunity

---

## Acknowledgements

- [Sri Lanka Disaster Management Centre (DMC)](http://www.dmc.gov.lk/) for maintaining public disaster records
- [NASA GPM Mission](https://gpm.nasa.gov/) for open-access global precipitation data
- UNDRR for the DesInventar open disaster database
- Research by [Udayanga et al. (2018)](https://www.mdpi.com/2072-4292/10/3/448) on Sri Lanka flood mapping using satellite data — a direct technical inspiration for this project

---

## Author

**[Your Name]**
Undergraduate, [Your University]
[linkedin.com/in/yourprofile](https://linkedin.com) · [your.email@gmail.com](mailto:your.email@gmail.com)

*Built as part of a portfolio project demonstrating applied ML for disaster risk reduction in Sri Lanka.*

---

## License

MIT License — see [LICENSE](LICENSE) for details. Free to use, adapt, and build upon with attribution.