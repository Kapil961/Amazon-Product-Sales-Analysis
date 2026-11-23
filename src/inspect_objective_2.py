
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/amazon_product_sales_latest.csv"

print("\n========== OBJECTIVE 2: DATA INSPECTION ==========")

df = pd.read_csv(DATA_PATH)

columns = [
    "rating",
    "reviews",
    "monthly_sales",
    "monthly_revenue"
]

data = df[columns].copy()

print("\n========== DATASET OVERVIEW ==========")
print("Total products:", len(data))
print("Columns analysed:", columns)

print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\n========== DUPLICATE CHECK ==========")
print("Duplicate rows:", data.duplicated().sum())

print("\n========== INVALID VALUE CHECK ==========")

print("Invalid ratings:", ((data["rating"] < 0) | (data["rating"] > 5)).sum())
print("Negative reviews:", (data["reviews"] < 0).sum())
print("Negative monthly sales:", (data["monthly_sales"] < 0).sum())
print("Negative monthly revenue:", (data["monthly_revenue"] < 0).sum())

print("\n========== UNIQUE RATINGS ==========")
print(sorted(data["rating"].dropna().unique()))

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(data.describe().round(2).to_string())

print("\n========== SKEWNESS ==========")
print(data.skew().round(3))

print("\n========== OUTLIER INSPECTION (IQR) ==========")

for column in columns:
    values = data[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((values < lower) | (values > upper)).sum()

    print(f"\n{column}")
    print("Lower bound:", round(lower, 2))
    print("Upper bound:", round(upper, 2))
    print("Potential outliers:", outliers)

print("\n========== OBJECTIVE 2 DATA INSPECTION COMPLETED ==========")
