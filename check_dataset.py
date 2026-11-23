
import pandas as pd

# Load Amazon dataset
df = pd.read_csv("data/raw/amazon_product_sales.csv")

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
