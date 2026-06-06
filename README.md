# Synergizing UAV, Satellite, and Weather Data for Paddy Yield Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** *Synergizing High-Resolution UAV Remote Sensing with Satellite Time-Series and Weather Covariates for Plot-Level Paddy Yield Prediction: A Benchmark of TabPFN and Machine Learning in Tropical Smallholder Systems*
>
> Submitted to **IEEE Access**

---

## Overview

This repository contains the complete reproducible pipeline for the paper. We fuse three sources of remote sensing and environmental data:

| Source | Data | Features |
|---|---|---|
| **UAV** | High-res multispectral imagery, 3 growth stages | 66 features (color moments, GLCM texture, NDVI stats) |
| **Sentinel-2** | L2A time-series, Aug–Sep 2025 via AWS STAC API | 6 features (NDVI, CIRE2, LSWI × 2 months) |
| **NASA POWER** | Daily weather parameters, Aug–Sep 2025 | 12 features (Tmax, Tmin, SRAD, RH, WS, GDD × 2 months) |

Models benchmarked: **SVR, Random Forest, CatBoost, TabPFN** across 3 feature configurations (Satellite-only / UAV-only / Fused) under standard 5-fold CV and Leave-One-Petak-Out spatial block CV.

---

## Key Results

| Configuration | Model | Standard R² | Spatial R² |
|---|---|:---:|:---:|
| UAV-only | Random Forest | **0.2126** | 0.0773 |
| UAV-only | TabPFN | 0.2086 | 0.0879 |
| Fused | **TabPFN** | 0.1968 | **0.1150** |
| Satellite-only | CatBoost | 0.1556 | -0.0048 |

**TabPFN (Fused)** achieves the best spatial generalizability with R² = 0.1150 under Leave-One-Petak-Out cross-validation, outperforming Random Forest (0.0658) and CatBoost (0.0350) by a wide margin.

---

## Repository Structure

```
paddy-yield-prediction/
│
├── run_pipeline.py              # Master orchestrator — runs all steps in sequence
│
├── extract_coordinates.py       # Step 1: Extract UTM centroids from UAV NDVI rasters
├── fetch_weather.py             # Step 2: Fetch NASA POWER weather data (Aug–Sep 2025)
├── query_satellite_weather.py   # Step 3: Query Sentinel-2 L2A via Element84 STAC API
├── build_combined_dataset.py    # Step 4: Fuse UAV + Sentinel-2 + Weather features
├── train_combined_models.py     # Step 5: Train & evaluate SVR, RF, CatBoost, TabPFN
├── generate_satellite_figure.py # Step 6: Generate Sentinel-2 NDVI study area map
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Data Requirements

The pipeline requires the following **user-provided** data (not included in repo due to size):

### 1. UAV NDVI Rasters (ESRI Grid format)
```
E:\ITS\NDVI\
├── fase 1\      # Phase 1: Vegetative (Aug 13–17, 2025)
│   ├── F1_P1_U1\w001001.adf
│   └── ...
├── fase 2\      # Phase 2: Reproductive (Aug 30 – Sep 14, 2025)
└── fase 3\      # Phase 3: Ripening (Sep 20–22, 2025)
```
- **514 plots** total, each in a separate subfolder
- Each subfolder contains an ESRI Grid raster (`.adf`)

### 2. UAV Feature Excel (ANN Dataset)
```
C:\...\ITS\Dataset ANN Padi.xlsx
```
Sheets: `FASE 1`, `FASE 2`, `FASE 3`  
Columns: `Nama`, `Jumlah Tanaman Per Ubin`, `Temperatur Tanah (°C)`, `Soil Moisture (%)`, `Curah Hujan (mm)`, `Target Output atau Hasil Panen Aktual (kg/m2)`

### 3. UAV Multitemporal Feature CSV (pre-extracted)
```
C:\...\ITS\paddy_multitemporal_features.csv
```
Run `extract_multitemporal_features.py` first to generate this file from the UAV images and NDVI rasters.

### 4. Public APIs (no authentication required)
- **Sentinel-2 L2A:** [Element84 STAC API](https://earth-search.aws.element84.com/v1)
- **Weather:** [NASA POWER API](https://power.larc.nasa.gov/api)

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip
- GDAL/rasterio (requires GDAL binary)

### Install Dependencies

```bash
# Conda (recommended)
conda create -n paddy-yield python=3.11
conda activate paddy-yield
conda install -c conda-forge rasterio gdal
pip install -r requirements.txt
```

### TabPFN Weight Fix (required for offline use)

TabPFN v0.1.9 auto-downloads model weights from GitHub. If the download fails (404), manually download the checkpoint:

```python
import urllib.request, shutil

url = "https://github.com/automl/TabPFN/raw/tabpfn_v1/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
save_path = r"C:\Users\<USER>\miniconda3\Lib\site-packages\tabpfn\models_diff\prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
urllib.request.urlretrieve(url, save_path)
# TabPFN loader also tries epoch_100, so copy:
shutil.copy(save_path.replace("42", "42"), save_path.replace("42", "100"))
```

---

## Running the Pipeline

### Configure Paths

Before running, update the path constants in each script to match your local data location. All paths are defined at the top of each script as constants:

```python
# In extract_coordinates.py:
NDVI_ROOT = r"E:\ITS\NDVI\fase 3"          # Update this
OUTPUT_CSV = r"C:\...\ITS\plot_coordinates.csv"  # Update this

# In run_pipeline.py:
WORKSPACE_DIR = r"C:\...\ITS"              # Update this
```

### Run Full Pipeline

```bash
python run_pipeline.py
```

Expected runtime on a typical laptop:
- Step 1 (Coordinate extraction): ~5–10 min (disk I/O intensive)
- Step 2 (Weather fetch): ~5 sec
- Step 3 (Sentinel-2 query): ~15–20 min (network I/O)
- Step 4 (Dataset fusion): ~5 sec
- Step 5 (Model training): ~3–5 min
- Step 6 (NDVI figure): ~1–2 min

### Run Individual Steps

```bash
python extract_coordinates.py   # Step 1
python fetch_weather.py         # Step 2
python query_satellite_weather.py  # Step 3
python build_combined_dataset.py   # Step 4
python train_combined_models.py    # Step 5
python generate_satellite_figure.py  # Step 6
```

---

## Output Files

| File | Description |
|---|---|
| `plot_coordinates.csv` | 514 plot centroids (UTM + WGS84) |
| `weather_data.csv` | Daily weather for Aug–Sep 2025 |
| `satellite_features.csv` | Sentinel-2 spectral indices per plot |
| `fused_paddy_dataset.csv` | Final fused dataset (482 matched plots, 84 features) |
| `results_combined/model_comparison_results.csv` | R², MAE, RMSE for all models |
| `results_combined/performance_comparison_plot.png` | Bar chart comparison figure |
| `results_combined/best_model_scatter.png` | Predicted vs. actual yield scatter |
| `results_combined/sentinel_ndvi_map.png` | Sentinel-2 NDVI study area map |

---

## Hyperparameter Settings

All models use `random_state=42` for reproducibility.

| Model | Hyperparameters |
|---|---|
| SVR | `kernel='rbf'`, `C=10`, `gamma='scale'` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `n_jobs=-1` |
| CatBoost | `iterations=300`, `depth=6`, `learning_rate=0.05` |
| TabPFN | `N_ensemble_configurations=4`, `n_bins=10` (regression discretization) |

### Cross-Validation
- **Standard:** `KFold(n_splits=5, shuffle=True, random_state=42)`
- **Spatial:** `GroupKFold(n_splits=len(unique_petaks))` — Leave-One-Petak-Out

---

## Compatibility Patches

Two compatibility patches are applied in `train_combined_models.py` to ensure the pipeline works with current library versions:

```python
# PyTorch 2.6+ compatibility
import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

# scikit-learn 1.6+ compatibility (removed force_all_finite)
import sklearn.utils.validation
# (see train_combined_models.py for full patch)
```

---

## Study Site

**Location:** Desa Kedungwaras, Modo, Lamongan, East Java, Indonesia (7°11'S, 112°8'E)  
**Area:** ~2.714 hectares of irrigated and rainfed paddy fields  
**Plots:** 514 experimental plots (ubin), each 1 m × 1 m  
**Spatial blocks:** 9 farm blocks (Petaks P1–P9)  
**Harvest:** September 20–22, 2025

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{putra2026paddy,
  title={Synergizing High-Resolution UAV Remote Sensing with Satellite Time-Series 
         and Weather Covariates for Plot-Level Paddy Yield Prediction},
  author={Putra, Zico Pratama and others},
  journal={IEEE Access},
  year={2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions about this research, please contact:  
**Zico Hasan** — GitHub: [@zicohasan](https://github.com/zicohasan)
