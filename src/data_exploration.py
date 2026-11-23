
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/amazon_product_sales.csv")

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate rows:", df.duplicated().sum())

print("Duplicate ASINs:", df["asin"].duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe().round(2))

print("\n========== CATEGORICAL SUMMARY ==========")

for col in ["brand", "grade", "categories", "is_prime"]:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

print("\n========== DATA QUALITY CHECK ==========")

print("Zero prices:", (df["price"] == 0).sum())

print("Negative prices:", (df["price"] < 0).sum())

print("Zero monthly sales:", (df["monthly_sales"] == 0).sum())

print("Negative monthly sales:",
      (df["monthly_sales"] < 0).sum())

print("Invalid ratings:",
      ((df["rating"] < 0) | (df["rating"] > 5)).sum())

print("\n========== REVENUE VALIDATION ==========")

df["calculated_revenue"] = (
    df["price"] * df["monthly_sales"]
)

df["revenue_difference"] = (
    df["monthly_revenue"] - df["calculated_revenue"]
)

print("Average revenue difference:",
      df["revenue_difference"].abs().mean())

print("Maximum revenue difference:",
      df["revenue_difference"].abs().max())

print("\nEDA basic investigation completed!")
