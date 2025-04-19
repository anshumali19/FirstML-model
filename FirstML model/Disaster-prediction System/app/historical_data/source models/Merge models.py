import pandas as pd
import os

# Get the current directory and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), "data")

# Read the individual prediction files
flood_df = pd.read_csv(os.path.join(data_dir, "Flood.csv"))
cyclone_df = pd.read_csv(os.path.join(data_dir, "Cyclone.csv"))
earthquake_df = pd.read_csv(os.path.join(data_dir, "Earthquake.csv"))

# Ensure 'Latitude' and 'Longitude' columns are of the same type (float64)
flood_df['Latitude'] = pd.to_numeric(flood_df['Latitude'], errors='coerce')
flood_df['Longitude'] = pd.to_numeric(flood_df['Longitude'], errors='coerce')
cyclone_df['Latitude'] = pd.to_numeric(cyclone_df['Latitude'], errors='coerce')
cyclone_df['Longitude'] = pd.to_numeric(cyclone_df['Longitude'], errors='coerce')
earthquake_df['Latitude'] = pd.to_numeric(earthquake_df['Latitude'], errors='coerce')
earthquake_df['Longitude'] = pd.to_numeric(earthquake_df['Longitude'], errors='coerce')

# Create risk columns for each disaster type
flood_df['flood_risk'] = flood_df['flood_risk'].fillna(0)
cyclone_df['cyclone_risk'] = cyclone_df.apply(
    lambda row: 1 if row['Wind Speed (km/h)'] >= 100 or row['Pressure (mb)'] <= 980 else 0,
    axis=1
)
earthquake_df['earthquake_risk'] = earthquake_df.apply(
    lambda row: 1 if row['Magnitude'] >= 5.0 else 0,
    axis=1
)

# Merge the dataframes on Latitude and Longitude
merged_df = pd.merge(flood_df, cyclone_df, on=['Latitude', 'Longitude'], how='outer')
merged_df = pd.merge(merged_df, earthquake_df, on=['Latitude', 'Longitude'], how='outer')

# Create a combined risk column
merged_df['combined_risk'] = merged_df.apply(
    lambda row: 1 if (row.get('flood_risk', 0) == 1 or 
                      row.get('cyclone_risk', 0) == 1 or 
                      row.get('earthquake_risk', 0) == 1) else 0,
    axis=1
)

# Save the merged predictions to a new CSV file
output_file = os.path.join(data_dir, 'combined_disaster_predictions.csv')
merged_df.to_csv(output_file, index=False)

print(f"✅ Merged data saved as '{output_file}'")
print("\nSample of combined predictions:")
print(merged_df[['Latitude', 'Longitude', 'flood_risk', 'cyclone_risk', 'earthquake_risk', 'combined_risk']].head())
