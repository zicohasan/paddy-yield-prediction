import os
import requests
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import time
import shutil

COORDS_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\plot_coordinates.csv"
OUTPUT_DIR = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\results_combined"
BRAIN_DIR = r"C:\Users\DELL LATITUDE E5420\.gemini\antigravity\brain\dd992cad-8a81-417b-a9a1-257f06c3bcfe"

def main():
    print("=" * 60)
    print("GENERATING SENTINEL-2 NDVI MAP FIGURE")
    print("=" * 60)
    
    if not os.path.exists(COORDS_CSV):
        print(f"Error: {COORDS_CSV} not found.")
        return
        
    df_coords = pd.read_csv(COORDS_CSV)
    print(f"Loaded {len(df_coords)} plot coordinates.")
    
    # Define bounding box with a margin of ~1 km
    margin = 0.008
    min_lon = df_coords['longitude'].min() - margin
    max_lon = df_coords['longitude'].max() + margin
    min_lat = df_coords['latitude'].min() - margin
    max_lat = df_coords['latitude'].max() + margin
    bbox = [min_lon, min_lat, max_lon, max_lat]
    print(f"Query bbox: {bbox}")
    
    # Search STAC API
    url = 'https://earth-search.aws.element84.com/v1/search'
    params = {
        'collections': ['sentinel-2-l2a'],
        'bbox': bbox,
        'datetime': '2025-08-01T00:00:00Z/2025-09-30T23:59:59Z',
        'limit': 100
    }
    
    try:
        res = requests.post(url, json=params, timeout=15)
        if res.status_code != 200:
            print("STAC search failed.")
            return
        data = res.json()
        features = data.get('features', [])
    except Exception as e:
        print("Error searching scenes:", e)
        return
        
    if not features:
        print("No scenes found.")
        return
        
    # Find a scene in late August or early September with low cloud cover
    # Let's sort by cloud cover and date
    selected_scene = None
    for f in sorted(features, key=lambda x: (x['properties']['eo:cloud_cover'], -pd.to_datetime(x['properties']['datetime']).value)):
        date_str = f['properties']['datetime'][:10]
        # Prefer dates between August 20 and September 15 (peak growth / heading)
        if "2025-08-20" <= date_str <= "2025-09-15":
            selected_scene = f
            break
            
    if selected_scene is None:
        selected_scene = features[0]
        
    date_str = selected_scene['properties']['datetime'][:10]
    cloud_cover = selected_scene['properties']['eo:cloud_cover']
    print(f"Selected Scene Date: {date_str} (Cloud Cover: {cloud_cover:.1f}%)")
    
    # Open B04 (Red) and B08 (NIR)
    assets = selected_scene['assets']
    red_url = assets['red']['href']
    nir_url = assets['nir']['href']
    
    try:
        print("Opening Red band raster stream...")
        src_red = rasterio.open(red_url)
        print("Opening NIR band raster stream...")
        src_nir = rasterio.open(nir_url)
    except Exception as e:
        print("Error opening rasters:", e)
        return
        
    # Warp bounding box to scene CRS
    crs = src_red.crs
    xs, ys = transform('EPSG:4326', crs, [min_lon, max_lon], [min_lat, max_lat])
    scene_bbox = [min(xs), min(ys), max(xs), max(ys)]
    
    # Read windows
    win = from_bounds(*scene_bbox, src_red.transform)
    win = win.intersection(rasterio.windows.Window(0, 0, src_red.width, src_red.height))
    
    print("Reading raster data...")
    red_data = src_red.read(1, window=win).astype(np.float32)
    nir_data = src_nir.read(1, window=win).astype(np.float32)
    
    # Get spatial extent for plotting in scene CRS
    tr = rasterio.windows.transform(win, src_red.transform)
    # Extent is [xmin, xmax, ymin, ymax]
    extent = [tr.c, tr.c + tr.a * win.width, tr.f + tr.e * win.height, tr.f]
    # Adjust extent orientation
    extent = [min(extent[0], extent[1]), max(extent[0], extent[1]), min(extent[2], extent[3]), max(extent[2], extent[3])]
    
    src_red.close()
    src_nir.close()
    
    # Compute NDVI
    print("Computing NDVI...")
    denom = nir_data + red_data
    # Avoid divide by zero
    denom[denom == 0] = 1e-9
    ndvi = (nir_data - red_data) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # Transform plot UTM coordinates to scene CRS if necessary
    # Since UTM Zone 49S is EPSG:32749, if scene CRS is the same, we just use utm_east and utm_north
    plot_xs = []
    plot_ys = []
    
    if crs != 'EPSG:32749':
        xs_p, ys_p = transform('EPSG:32749', crs, df_coords['utm_east'].tolist(), df_coords['utm_north'].tolist())
    else:
        xs_p, ys_p = df_coords['utm_east'].tolist(), df_coords['utm_north'].tolist()
        
    # Filter plots that fall inside the extent
    for px, py in zip(xs_p, ys_p):
        if extent[0] <= px <= extent[1] and extent[2] <= py <= extent[3]:
            plot_xs.append(px)
            plot_ys.append(py)
            
    print(f"Overlaying {len(plot_xs)} plots falling inside map extent.")
    
    # Plotting
    print("Creating plot...")
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Use terrain or green colormap
    im = plt.imshow(ndvi, cmap='RdYlGn', extent=extent, vmin=0.1, vmax=0.9)
    
    # Overlay plots
    plt.scatter(plot_xs, plot_ys, c='blue', s=8, edgecolors='white', linewidths=0.3, label='Experimental Plots (1 m²)')
    
    # Add title and labels
    plt.title(f"Sentinel-2 NDVI Map of Modo, Lamongan (Date: {date_str})", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("UTM Easting (m)", fontsize=10)
    plt.ylabel("UTM Northing (m)", fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI Value", fontsize=10)
    
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.8, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_path = os.path.join(OUTPUT_DIR, "sentinel_ndvi_map.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to: {img_path}")
    
    # Copy to brain artifacts
    brain_path = os.path.join(BRAIN_DIR, "sentinel_ndvi_map.png")
    shutil.copy2(img_path, brain_path)
    print(f"Copied figure to brain directory: {brain_path}")
    print("SUCCESS!")

if __name__ == "__main__":
    main()
