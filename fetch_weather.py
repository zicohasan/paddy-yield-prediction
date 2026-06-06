import os
import requests
import json
import pandas as pd

OUTPUT_CSV = r"C:\Users\DELL LATITUDE E5420\Documents\ITS\weather_data.csv"
site_lon, site_lat = 112.1333, -7.1833
start_date = "20250801"
end_date = "20250930"

def main():
    print(f"Fetching weather data for Lamongan ({site_lat}, {site_lon}) from {start_date} to {end_date}...")
    url = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    params = {
        'parameters': 'T2M_MIN,T2M_MAX,ALLSKY_SFC_SW_DWN,RH2M,WS2M',
        'community': 'ag',
        'longitude': site_lon,
        'latitude': site_lat,
        'start': start_date,
        'end': end_date,
        'format': 'JSON'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return
            
        data = response.json()
        parameter_data = data['properties']['parameter']
        
        dates = list(parameter_data['T2M_MIN'].keys())
        print(f"Received weather data for {len(dates)} days.")
        
        records = []
        for d in dates:
            # Date is in format YYYYMMDD, let's convert to standard YYYY-MM-DD
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            records.append({
                'date': date_str,
                't2m_min': parameter_data['T2M_MIN'][d],
                't2m_max': parameter_data['T2M_MAX'][d],
                'solar_radiation': parameter_data['ALLSKY_SFC_SW_DWN'][d],
                'relative_humidity': parameter_data['RH2M'][d],
                'wind_speed': parameter_data['WS2M'][d]
            })
            
        df = pd.DataFrame(records)
        
        # Calculate Growing Degree Days (GDD)
        # Base temperature for rice is typically 10 degrees C
        t_base = 10.0
        df['gdd'] = ((df['t2m_max'] + df['t2m_min']) / 2.0 - t_base).clip(lower=0)
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully saved weather data to {OUTPUT_CSV}")
        print(df.head())
        
    except Exception as e:
        print("Error fetching weather data:", e)

if __name__ == "__main__":
    main()
