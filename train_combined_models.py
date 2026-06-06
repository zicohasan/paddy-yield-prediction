import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply PyTorch 2.6+ weights_only compatibility patch
import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

# Apply scikit-learn compatibility patch for newer sklearn versions
import sklearn.utils.validation
original_check_X_y = sklearn.utils.validation.check_X_y
original_check_array = sklearn.utils.validation.check_array

def patched_check_X_y(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return original_check_X_y(*args, **kwargs)

def patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return original_check_array(*args, **kwargs)

sklearn.utils.validation.check_X_y = patched_check_X_y
sklearn.utils.validation.check_array = patched_check_array

# Try to import CatBoost and TabPFN, fallback if failed
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

from sklearn.base import BaseEstimator, RegressorMixin

class TabPFNRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, device='cpu', n_bins=10, random_state=42):
        self.device = device
        self.n_bins = n_bins
        self.random_state = random_state
        self.classifier = None
        self.bin_centers = None
        self.bin_edges = None

    def fit(self, X, y):
        # We handle uniform binning across the target y range
        y_min, y_max = np.min(y), np.max(y)
        y_max += 1e-9
        
        self.bin_edges = np.linspace(y_min, y_max, self.n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        
        y_binned = np.digitize(y, self.bin_edges) - 1
        y_binned = np.clip(y_binned, 0, self.n_bins - 1)
        
        # TabPFN is very fast on N=514 samples. We set N_ensemble_configurations=4 for speed/accuracy balance.
        self.classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=4)
        self.classifier.fit(X, y_binned)
        return self

    def predict(self, X):
        probs = self.classifier.predict_proba(X)
        classes = self.classifier.classes_
        
        y_pred = np.zeros(X.shape[0])
        for idx, cls in enumerate(classes):
            y_pred += probs[:, idx] * self.bin_centers[int(cls)]
            
        return y_pred


FUSED_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\fused_paddy_dataset.csv"
RESULTS_DIR = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\results_combined"

def prepare_data(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['yield'].copy()
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X_imp)
    
    return X_scale, y.values

def evaluate_configuration(df, feature_cols, config_name, groups, unique_petaks):
    print(f"\nEvaluating configuration: {config_name} ({len(feature_cols)} features)...")
    X, y = prepare_data(df, feature_cols)
    
    # Setup cross-validation
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=len(unique_petaks)) # Leave-One-Petak-Out
    
    # Setup models
    models = {
        'SVR': SVR(C=10, kernel='rbf', gamma='scale'),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
        
    if HAS_TABPFN:
        # TabPFN is pretrained and doesn't require hyperparameter tuning
        # tabpfn can be slow on CPU if N is large, but for N=514 it is very fast
        # TabPFNRegressorWrapper is scikit-learn compatible
        models['TabPFN'] = TabPFNRegressorWrapper(device='cpu')
        
    results = []
    
    for name, model in models.items():
        print(f"  Training {name}...")
        t0 = time.time()
        
        # 1. Standard 5-Fold CV
        try:
            yp_5f = cross_val_predict(model, X, y, cv=kf5)
            r2_5f = r2_score(y, yp_5f)
            mae_5f = mean_absolute_error(y, yp_5f)
            rmse_5f = np.sqrt(mean_squared_error(y, yp_5f))
        except Exception as e:
            print(f"    Error in 5-fold CV for {name}: {e}")
            r2_5f, mae_5f, rmse_5f = np.nan, np.nan, np.nan
            yp_5f = np.zeros_like(y)
            
        # 2. Spatial Block CV (LOPOCV)
        try:
            yp_sp = cross_val_predict(model, X, y, cv=gkf, groups=groups)
            r2_sp = r2_score(y, yp_sp)
            mae_sp = mean_absolute_error(y, yp_sp)
            rmse_sp = np.sqrt(mean_squared_error(y, yp_sp))
        except Exception as e:
            print(f"    Error in Spatial CV for {name}: {e}")
            r2_sp, mae_sp, rmse_sp = np.nan, np.nan, np.nan
            yp_sp = np.zeros_like(y)
            
        print(f"    Standard 5-Fold | R²={r2_5f:.4f}, MAE={mae_5f:.4f}")
        print(f"    Spatial Block   | R²={r2_sp:.4f}, MAE={mae_sp:.4f} ({time.time() - t0:.1f}s)")
        
        results.append({
            'Configuration': config_name,
            'Model': name,
            'Standard_R2': r2_5f,
            'Standard_MAE': mae_5f,
            'Standard_RMSE': rmse_5f,
            'Spatial_R2': r2_sp,
            'Spatial_MAE': mae_sp,
            'Spatial_RMSE': rmse_sp,
            'predictions_5f': yp_5f,
            'predictions_sp': yp_sp
        })
        
    return results

def main():
    t_start = time.time()
    print("=" * 60)
    print("RUNNING MACHINE LEARNING BENCHMARK (INCLUDING TABPFN)")
    print("=" * 60)
    print("CatBoost available:", HAS_CATBOOST)
    print("TabPFN available:", HAS_TABPFN)
    
    if not os.path.exists(FUSED_CSV):
        print(f"Error: {FUSED_CSV} not found. Run fusion script first.")
        return
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(FUSED_CSV).dropna(subset=['yield'])
    print(f"Loaded {len(df)} samples for training.")
    
    # Define groups for spatial CV (Petaks)
    if 'plot_id' in df.columns:
        df['petak'] = df['plot_id'].astype(str).str.split('_').str[0]
    else:
        df['petak'] = df['code'].astype(str).str.split('_').str[1]
        
    unique_petaks = sorted(df['petak'].unique())
    print(f"Unique spatial blocks (petak) for CV: {unique_petaks}")
    
    # Encode petak as integer groups
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    groups = le.fit_transform(df['petak'])
    
    # Feature columns
    all_cols = df.columns.tolist()
    
    # 1. UAV features
    # Exclude non-features: 'code', 'plot_id', 'yield', 'petak' and S2/weather columns
    s2_weather_keywords = ['s2_', 'w_']
    uav_exclude = ['code', 'plot_id', 'yield', 'petak']
    uav_cols = [c for c in all_cols if not any(k in c for k in s2_weather_keywords) and c not in uav_exclude]
    
    # 2. Satellite-only features (S2 indices + weather)
    s2_cols = [c for c in all_cols if 's2_' in c]
    weather_cols = [c for c in all_cols if 'w_' in c]
    sat_cols = s2_cols + weather_cols
    
    # 3. Fused features
    fused_cols = uav_cols + sat_cols
    
    print(f"Feature sizes: UAV-only={len(uav_cols)}, Satellite-only={len(sat_cols)}, Fused={len(fused_cols)}")
    
    all_results = []
    
    # Run evaluations
    all_results.extend(evaluate_configuration(df, sat_cols, 'Satellite-only', groups, unique_petaks))
    all_results.extend(evaluate_configuration(df, uav_cols, 'UAV-only', groups, unique_petaks))
    all_results.extend(evaluate_configuration(df, fused_cols, 'Fused', groups, unique_petaks))
    
    # Create Summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Configuration': r['Configuration'],
            'Model': r['Model'],
            'Standard_R2': r['Standard_R2'],
            'Standard_MAE': r['Standard_MAE'],
            'Standard_RMSE': r['Standard_RMSE'],
            'Spatial_R2': r['Spatial_R2'],
            'Spatial_MAE': r['Spatial_MAE'],
            'Spatial_RMSE': r['Spatial_RMSE']
        })
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULTS_DIR, "model_comparison_results.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary results to: {summary_path}")
    print(df_summary)
    
    # --- VISUALIZATION ---
    print("\nGenerating performance comparison plots...")
    # Plot Standard CV vs Spatial CV R² comparison
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Reshape for plotting
    plot_df = []
    for r in all_results:
        plot_df.append({
            'Model-Config': f"{r['Model']}\n({r['Configuration']})",
            'Strategy': 'Standard 5-Fold CV',
            'R2': r['Standard_R2']
        })
        plot_df.append({
            'Model-Config': f"{r['Model']}\n({r['Configuration']})",
            'Strategy': 'Spatial Block CV',
            'R2': r['Spatial_R2']
        })
    df_plot = pd.DataFrame(plot_df)
    
    ax = sns.barplot(x='Model-Config', y='R2', hue='Strategy', data=df_plot, palette=['#2196F3', '#FF5722'], edgecolor='black', alpha=0.9)
    plt.title('Paddy Yield Prediction R² Comparison: UAV vs Satellite vs Fused Models', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model & Feature Configuration', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim(-0.2, 0.45)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.legend(fontsize=11)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')
            
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "performance_comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")
    
    # Find best model to plot scatter
    best_idx = df_summary['Standard_R2'].idxmax()
    best_r = all_results[best_idx]
    best_model_name = f"{best_r['Model']} ({best_r['Configuration']})"
    print(f"Best model based on Standard R²: {best_model_name} with R² = {best_r['Standard_R2']:.4f}")
    
    # Plot Scatter Predicted vs Actual for Best Model
    y_actual = df['yield'].values
    y_pred = best_r['predictions_5f']
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred, alpha=0.6, color='#2E7D32', edgecolors='k', linewidth=0.5)
    mn = min(y_actual.min(), y_pred.min())
    mx = max(y_actual.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='1:1 Reference Line')
    plt.xlabel('Actual Yield (kg/m²)', fontsize=12)
    plt.ylabel('Predicted Yield (kg/m²)', fontsize=12)
    plt.title(f'Predicted vs. Actual Yield: {best_model_name}\nStandard 5-Fold CV (R² = {best_r["Standard_R2"]:.3f}, MAE = {best_r["Standard_MAE"]:.3f})', fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    scatter_path = os.path.join(RESULTS_DIR, "best_model_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")
    
    print(f"\nSUCCESS: Training and validation completed in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
