import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform
import time

NDVI_ROOT = r"E:\ITS\NDVI\fase 3"
OUTPUT_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\plot_coordinates.csv"

def main():
    t0 = time.time()
    print("Starting optimized coordinate extraction for all plots...")
    
    if not os.path.exists(NDVI_ROOT):
        print(f"Error: Directory {NDVI_ROOT} does not exist.")
        return
        
    subdirs = [d for d in os.listdir(NDVI_ROOT) if os.path.isdir(os.path.join(NDVI_ROOT, d)) and d.lower() != "info"]
    print(f"Found {len(subdirs)} plot directories to process.")
    
    results = []
    processed = 0
    
    for d in subdirs:
        code = d.upper() # e.g. F3_P1_U1
        adf_path = os.path.join(NDVI_ROOT, d, "w001001.adf")
        if not os.path.exists(adf_path):
            continue
            
        try:
            with rasterio.open(adf_path) as src:
                h, w = src.height, src.width
                # Decimate by 10 to speed up reading by ~10x
                dec_h, dec_w = max(1, h // 10), max(1, w // 10)
                arr = src.read(1, out_shape=(dec_h, dec_w))
                nodata = src.nodata
                mask = (arr != nodata) & (~np.isnan(arr))
                rows, cols = np.where(mask)
                
                if len(rows) > 0:
                    sc_y = h / dec_h
                    sc_x = w / dec_w
                    dec_transform = src.transform * src.transform.scale(sc_x, sc_y)
                    xs, ys = rasterio.transform.xy(dec_transform, rows, cols)
                    mean_x = np.mean(xs)
                    mean_y = np.mean(ys)
                    
                    # Convert to Lat/Lon
                    lon, lat = transform('EPSG:32749', 'EPSG:4326', [mean_x], [mean_y])
                    
                    results.append({
                        'code': code,
                        'utm_east': mean_x,
                        'utm_north': mean_y,
                        'longitude': lon[0],
                        'latitude': lat[0],
                        'pixel_count': len(rows)
                    })
        except Exception as e:
            print(f"Error processing {code}: {e}")
            
        processed += 1
        if processed % 100 == 0 or processed == len(subdirs):
            print(f"  Processed {processed}/{len(subdirs)} plots ({time.time() - t0:.1f}s)...")
            
    df = pd.DataFrame(results)
    df['plot_id'] = df['code'].apply(lambda x: "_".join(x.split("_")[1:]))
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Completed! Saved {len(df)} plot coordinates to {OUTPUT_CSV} in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
