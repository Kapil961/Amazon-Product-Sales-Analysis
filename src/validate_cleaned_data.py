
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. LOAD CLEANED DATASET
# --------------------------------------------------

file_path = Path(
    "data/processed/amazon_product_sales_latest.csv"
)

df = pd.read_csv(file_path)

print("\n========== CLEANED DATASET VALIDATION ==========")

# --------------------------------------------------
# 2. BASIC DATASET INFORMATION
# --------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Unique products:", df["asin"].nunique())

print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# 3. DATA TYPES
# --------------------------------------------------

print("\n========== DATA TYPES ==========")

print(df.dtypes)

# --------------------------------------------------
# 4. MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")

missing_values = df.isnull().sum()

missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": (
        missing_values / len(df) * 100
    ).round(2)
})

print(missing_summary)

# --------------------------------------------------
# 5. DUPLICATE CHECK
# --------------------------------------------------

print("\n========== DUPLICATE CHECK ==========")

print(
    "Exact duplicate rows:",
    df.duplicated().sum()
)

print(
    "Duplicate ASINs:",
    df["asin"].duplicated().sum()
)

# --------------------------------------------------
# 6. INVALID AND UNUSUAL VALUES
# --------------------------------------------------

print("\n========== VALUE VALIDATION ==========")

print("Zero prices:", (df["price"] == 0).sum())
print("Negative prices:", (df["price"] < 0).sum())

print("Negative monthly sales:", (df["monthly_sales"] < 0).sum())
print("Zero monthly sales:", (df["monthly_sales"] == 0).sum())

print("Negative BSR values:", (df["bsr"] < 0).sum())
print("Zero BSR values:", (df["bsr"] == 0).sum())

print(
    "Ratings outside 0-5:",
    ((df["rating"] < 0) | (df["rating"] > 5)).sum()
)

print(
    "Missing timestamps:",
    df["updated_at"].isnull().sum()
)

# --------------------------------------------------
# 7. TIMESTAMP VALIDATION
# --------------------------------------------------

print("\n========== TIMESTAMP ANALYSIS ==========")

df["updated_at"] = pd.to_datetime(
    df["updated_at"],
    errors="coerce",
    utc=True
)

print("Earliest observation:", df["updated_at"].min())
print("Latest observation:", df["updated_at"].max())

print(
    "Unique timestamps:",
    df["updated_at"].nunique()
)

# --------------------------------------------------
# 8. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n========== DESCRIPTIVE STATISTICS ==========")

numeric_columns = df.select_dtypes(
    include="number"
).columns

print(
    df[numeric_columns].describe().round(2)
)

# --------------------------------------------------
# 9. SALES AND REVENUE OVERVIEW
# --------------------------------------------------

print("\n========== SALES AND REVENUE ANALYSIS ==========")

print(
    "Average monthly sales:",
    round(df["monthly_sales"].mean(), 2)
)

print(
    "Median monthly sales:",
    round(df["monthly_sales"].median(), 2)
)

print(
    "Average monthly revenue:",
    round(df["monthly_revenue"].mean(), 2)
)

print(
    "Median monthly revenue:",
    round(df["monthly_revenue"].median(), 2)
)

# --------------------------------------------------
# 10. FINAL VALIDATION STATUS
# --------------------------------------------------

print("\n========== FINAL VALIDATION STATUS ==========")

checks = {
    "Unique ASINs": df["asin"].nunique() == len(df),
    "No exact duplicate rows": df.duplicated().sum() == 0,
    "No missing timestamps": df["updated_at"].isnull().sum() == 0,
    "No invalid ratings": (
        ((df["rating"] < 0) | (df["rating"] > 5)).sum() == 0
    )
}

for check, result in checks.items():
    status = "PASS" if result else "CHECK"
    print(f"{check}: {status}")

print("\nDataset validation completed!")
