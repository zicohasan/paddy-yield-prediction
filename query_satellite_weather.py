import os
import requests
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds
import time

COORDS_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\plot_coordinates.csv"
OUTPUT_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\satellite_features.csv"

def search_sentinel2_scenes(bbox):
    print("Searching for Sentinel-2 L2A scenes in Element84 STAC API...")
    url = 'https://earth-search.aws.element84.com/v1/search'
    params = {
        'collections': ['sentinel-2-l2a'],
        'bbox': bbox,
        'datetime': '2025-08-01T00:00:00Z/2025-09-30T23:59:59Z',
        'limit': 100
    }
    try:
        response = requests.post(url, json=params, timeout=15)
        if response.status_code != 200:
            print(f"Error: STAC search returned status code {response.status_code}")
            return []
        data = response.json()
        features = data.get('features', [])
        print(f"Found {len(features)} Sentinel-2 scenes.")
        return features
    except Exception as e:
        print("Error searching Sentinel-2 scenes:", e)
        return []

def main():
    t_start = time.time()
    print("=" * 60)
    print("SENTINEL-2 TIME-SERIES EXTRACTION")
    print("=" * 60)
    
    if not os.path.exists(COORDS_CSV):
        print(f"Error: {COORDS_CSV} not found. Run extract_coordinates.py first.")
        return
        
    df_coords = pd.read_csv(COORDS_CSV)
    print(f"Loaded coordinates for {len(df_coords)} plots.")
    
    # Bounding box of the plots (WGS84)
    min_lon = df_coords['longitude'].min() - 0.002
    max_lon = df_coords['longitude'].max() + 0.002
    min_lat = df_coords['longitude'].min() # Wait, this is longitude! Let's check latitude bounds
    # Let's fix latitude bounds
    min_lat = df_coords['latitude'].min() - 0.002
    max_lat = df_coords['latitude'].max() + 0.002
    
    bbox = [min_lon, min_lat, max_lon, max_lat]
    print(f"Query bounding box: {bbox}")
    
    scenes = search_sentinel2_scenes(bbox)
    if not scenes:
        print("No scenes found. Exiting.")
        return
        
    # Sort scenes by date
    # Group scenes by date to avoid duplicate orbits on the same day
    scenes_by_date = {}
    for s in scenes:
        date_str = s['properties']['datetime'][:10] # YYYY-MM-DD
        cloud_pct = s['properties']['eo:cloud_cover']
        if date_str not in scenes_by_date or cloud_pct < scenes_by_date[date_str]['properties']['eo:cloud_cover']:
            scenes_by_date[date_str] = s
            
    sorted_dates = sorted(scenes_by_date.keys())
    print(f"Unique dates found ({len(sorted_dates)}): {sorted_dates}")
    
    # We will query:
    # B04 (red, 10m), B08 (nir, 10m), B06 (rededge2, 20m), B11 (swir16, 20m), SCL (scl, 20m)
    band_keys = {
        'red': 'red',
        'nir': 'nir',
        're2': 'rededge2',
        'swir16': 'swir16',
        'scl': 'scl'
    }
    
    # Initialize empty records for each plot and date
    # plot_data = {plot_id: {date: {ndvi, cire2, lswi, valid}}}
    plot_data = {row['plot_id']: {} for _, row in df_coords.iterrows()}
    
    for date_str in sorted_dates:
        print(f"\nProcessing date: {date_str}...")
        scene = scenes_by_date[date_str]
        assets = scene['assets']
        
        # Open bands using rasterio
        bands_src = {}
        for key, asset_name in band_keys.items():
            url = assets[asset_name]['href']
            try:
                bands_src[key] = rasterio.open(url)
            except Exception as e:
                print(f"  Warning: Failed to open {key} band: {e}")
                
        if len(bands_src) < len(band_keys):
            print(f"  Skipping date {date_str} due to missing bands.")
            for src in bands_src.values(): src.close()
            continue
            
        # Extract values for each plot
        # To avoid reopening / reading from URL multiple times, we read the bounding box window for all bands
        # We need to project bbox into the scene's CRS
        crs = bands_src['red'].crs
        # Transform WGS84 bbox to scene CRS
        xs, ys = transform('EPSG:4326', crs, [min_lon, max_lon], [min_lat, max_lat])
        scene_bbox = [min(xs), min(ys), max(xs), max(ys)]
        
        # Read windows for each band
        bands_data = {}
        bands_transform = {}
        for key, src in bands_src.items():
            win = from_bounds(*scene_bbox, src.transform)
            # Clip window to raster size to prevent invalid bounds
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            bands_data[key] = src.read(1, window=win)
            bands_transform[key] = rasterio.windows.transform(win, src.transform)
            
        # Process each plot
        for _, row in df_coords.iterrows():
            pid = row['plot_id']
            east = row['utm_east']
            north = row['utm_north']
            
            # Since plots are in UTM Zone 49S (EPSG:32749), we check if scene CRS is the same
            # If not, project plot coordinates to scene CRS
            if crs != 'EPSG:32749':
                px, py = transform('EPSG:32749', crs, [east], [north])
                px, py = px[0], py[0]
            else:
                px, py = east, north
                
            # Read pixel value from window
            values = {}
            for key in band_keys.keys():
                tr = bands_transform[key]
                arr = bands_data[key]
                # Inverse transform to get pixel coords in window array
                col_idx, row_idx = ~tr * (px, py)
                row_idx, col_idx = int(row_idx), int(col_idx)
                
                # Check boundaries
                if 0 <= row_idx < arr.shape[0] and 0 <= col_idx < arr.shape[1]:
                    values[key] = float(arr[row_idx, col_idx])
                else:
                    values[key] = np.nan
                    
            # Compute indices
            red = values.get('red', np.nan)
            nir = values.get('nir', np.nan)
            re2 = values.get('re2', np.nan)
            swir = values.get('swir16', np.nan)
            scl = values.get('scl', np.nan)
            
            ndvi = (nir - red) / (nir + red) if not np.isnan(red) and not np.isnan(nir) and (nir + red) != 0 else np.nan
            cire2 = (nir / re2) - 1.0 if not np.isnan(nir) and not np.isnan(re2) and re2 != 0 else np.nan
            lswi = (nir - swir) / (nir + swir) if not np.isnan(nir) and not np.isnan(swir) and (nir + swir) != 0 else np.nan
            
            # SCL cloud filter: 4=Vegetation, 5=Not Vegetated, 6=Water, 7=Unclassified
            # 3=Shadows, 8=Cloud Medium, 9=Cloud High, 10=Cirrus
            is_valid = scl in [4, 5, 6, 7]
            
            plot_data[pid][date_str] = {
                'ndvi': ndvi if is_valid else np.nan,
                'cire2': cire2 if is_valid else np.nan,
                'lswi': lswi if is_valid else np.nan,
                'valid': 1 if is_valid else 0
            }
            
        # Close band sources
        for src in bands_src.values():
            src.close()
            
    print("\nInterpolating missing values and compiling monthly aggregates...")
    # Compile a daily time series for each plot and interpolate
    all_dates = pd.date_range(start="2025-08-01", end="2025-09-30")
    
    records = []
    for pid in plot_data.keys():
        # Create df for this plot
        p_dates = sorted_dates
        p_ndvi = [plot_data[pid][d]['ndvi'] for d in p_dates if d in plot_data[pid]]
        p_cire2 = [plot_data[pid][d]['cire2'] for d in p_dates if d in plot_data[pid]]
        p_lswi = [plot_data[pid][d]['lswi'] for d in p_dates if d in plot_data[pid]]
        
        pdf = pd.DataFrame(index=pd.to_datetime(p_dates))
        pdf['ndvi'] = p_ndvi
        pdf['cire2'] = p_cire2
        pdf['lswi'] = p_lswi
        
        # Reindex to daily and interpolate
        pdf = pdf.reindex(all_dates)
        pdf = pdf.interpolate(method='linear', limit_direction='both')
        
        # Calculate monthly means
        # August
        aug_df = pdf.loc['2025-08-01':'2025-08-31']
        # September
        sep_df = pdf.loc['2025-09-01':'2025-09-30']
        
        records.append({
            'plot_id': pid,
            's2_ndvi_aug': aug_df['ndvi'].mean(),
            's2_cire2_aug': aug_df['cire2'].mean(),
            's2_lswi_aug': aug_df['lswi'].mean(),
            's2_ndvi_sep': sep_df['ndvi'].mean(),
            's2_cire2_sep': sep_df['cire2'].mean(),
            's2_lswi_sep': sep_df['lswi'].mean()
        })
        
    df_out = pd.DataFrame(records)
    # Check if there are any NaNs in aggregates and fill them
    df_out = df_out.fillna(df_out.mean(numeric_only=True))
    
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Extracted satellite features for {len(df_out)} plots in {time.time() - t_start:.1f}s.")
    print(df_out.head())

if __name__ == "__main__":
    main()
