import os
import pandas as pd
import numpy as np

UAV_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\paddy_multitemporal_features.csv"
S2_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\satellite_features.csv"
WEATHER_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\weather_data.csv"
OUTPUT_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\fused_paddy_dataset.csv"

def main():
    print("=" * 60)
    print("FUSING UAV, SATELLITE AND WEATHER FEATURES")
    print("=" * 60)
    
    # Check inputs
    for p in [UAV_CSV, S2_CSV, WEATHER_CSV]:
        if not os.path.exists(p):
            print(f"Error: {p} not found. Cannot fuse.")
            return
            
    df_uav = pd.read_csv(UAV_CSV)
    df_s2 = pd.read_csv(S2_CSV)
    df_weather = pd.read_csv(WEATHER_CSV)
    
    print(f"Loaded: {len(df_uav)} UAV rows, {len(df_s2)} S2 rows, {len(df_weather)} weather rows.")
    
    # Process weather data
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather.set_index('date', inplace=True)
    
    # August aggregates
    aug_w = df_weather.loc['2025-08-01':'2025-08-31']
    # September aggregates
    sep_w = df_weather.loc['2025-09-01':'2025-09-30']
    
    weather_feats = {
        'w_temp_max_mean_aug': aug_w['t2m_max'].mean(),
        'w_temp_min_mean_aug': aug_w['t2m_min'].mean(),
        'w_solar_rad_sum_aug': aug_w['solar_radiation'].sum(),
        'w_rh_mean_aug': aug_w['relative_humidity'].mean(),
        'w_wind_speed_mean_aug': aug_w['wind_speed'].mean(),
        'w_gdd_sum_aug': aug_w['gdd'].sum(),
        
        'w_temp_max_mean_sep': sep_w['t2m_max'].mean(),
        'w_temp_min_mean_sep': sep_w['t2m_min'].mean(),
        'w_solar_rad_sum_sep': sep_w['solar_radiation'].sum(),
        'w_rh_mean_sep': sep_w['relative_humidity'].mean(),
        'w_wind_speed_mean_sep': sep_w['wind_speed'].mean(),
        'w_gdd_sum_sep': sep_w['gdd'].sum()
    }
    
    # Convert weather features to a single row DataFrame
    df_w_row = pd.DataFrame([weather_feats])
    
    # We will replicate the weather row for all plots in the fusion
    print("Weather features calculated:")
    for k, v in weather_feats.items():
        print(f"  {k:25s}: {v:.4f}")
        
    # Standardize plot_id in UAV dataset
    # Wait, in paddy_multitemporal_features.csv, the column is 'plot_id'
    # Let's ensure plot_id is string and stripped
    df_uav['plot_id'] = df_uav['plot_id'].astype(str).str.strip()
    df_s2['plot_id'] = df_s2['plot_id'].astype(str).str.strip()
    
    # Merge UAV and S2
    df_fused = pd.merge(df_uav, df_s2, on='plot_id', how='inner')
    print(f"Merged UAV + S2. Rows: {len(df_fused)}")
    
    # Append weather features (since they are constant spatially, we assign them to all rows)
    for col, val in weather_feats.items():
        df_fused[col] = val
        
    # Save the fused dataset
    df_fused.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Saved fused dataset with {len(df_fused)} plots and {len(df_fused.columns)} features to {OUTPUT_CSV}")
    print("Fused columns sample:", df_fused.columns.tolist()[-15:])

if __name__ == "__main__":
    main()
