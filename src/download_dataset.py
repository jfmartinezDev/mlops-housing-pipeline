import pandas as pd
import os

# Define the dataset URL
dataset_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

# Create the output directory if it doesn't exist
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Define the full path to save the dataset
output_path = os.path.join(output_dir, "BostonHousing.csv")

# Download the dataset and save it as a CSV file
df = pd.read_csv(dataset_url)

# Tiny null values check
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("Null values per column:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

# Convert df to .csv and put it in the output path
df.to_csv(output_path, index=False)

# Print confirmation
print(f"Dataset successfully downloaded and saved to: {output_path}")

