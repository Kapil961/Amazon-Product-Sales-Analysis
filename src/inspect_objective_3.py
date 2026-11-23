
import os
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/amazon_product_sales_latest.csv"
RESULT_DIR = "outputs/results"

os.makedirs(RESULT_DIR, exist_ok=True)

print("\n========== OBJECTIVE 3: PREDICTOR INSPECTION ==========")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Total products:", len(df))
print("Total columns:", len(df.columns))

print("\nAll available columns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. DEFINE TARGET AND POTENTIAL FEATURES
# --------------------------------------------------

target = "monthly_sales"

potential_features = [
    "price",
    "bsr",
    "rating",
    "reviews",
    "listing_quality_score",
    "is_prime",
    "grade",
    "categories",
    "brand"
]

print("\n========== TARGET VARIABLE ==========")
print("Target:", target)
print("Target description: Monthly product sales")

print("\n========== POTENTIAL PREDICTORS ==========")
print(potential_features)

# --------------------------------------------------
# 3. DATA TYPES
# --------------------------------------------------

print("\n========== DATA TYPES ==========")
print(df[potential_features].dtypes.to_string())

# --------------------------------------------------
# 4. MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")

missing = pd.DataFrame({
    "missing_count": df[potential_features].isnull().sum(),
    "missing_percentage": (
        df[potential_features].isnull().mean() * 100
    )
})

missing = missing.sort_values(
    by="missing_percentage",
    ascending=False
)

print(missing.round(2).to_string())

missing.to_csv(
    f"{RESULT_DIR}/objective3_missing_values.csv"
)

# --------------------------------------------------
# 5. NUMERIC FEATURE STATISTICS
# --------------------------------------------------

numeric_features = [
    "price",
    "bsr",
    "rating",
    "reviews",
    "listing_quality_score"
]

print("\n========== NUMERIC FEATURE STATISTICS ==========")

print(
    df[numeric_features]
    .describe()
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 6. CATEGORICAL FEATURE INSPECTION
# --------------------------------------------------

categorical_features = [
    "is_prime",
    "grade",
    "brand",
    "categories"
]

print("\n========== CATEGORICAL FEATURE INSPECTION ==========")

for column in categorical_features:

    print(f"\nColumn: {column}")

    print("Unique values:", df[column].nunique(dropna=True))

    print("Missing values:", df[column].isnull().sum())

    print("\nMost frequent values:")

    print(
        df[column]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

# --------------------------------------------------
# 7. INVALID VALUE CHECKS
# --------------------------------------------------

print("\n========== INVALID VALUE CHECKS ==========")

print("Zero prices:", (df["price"] == 0).sum())
print("Negative prices:", (df["price"] < 0).sum())

print("Zero BSR:", (df["bsr"] == 0).sum())
print("Negative BSR:", (df["bsr"] < 0).sum())

print(
    "Invalid ratings:",
    ((df["rating"] < 0) | (df["rating"] > 5)).sum()
)

print(
    "Negative review counts:",
    (df["reviews"] < 0).sum()
)

print(
    "Invalid target sales:",
    (df[target] <= 0).sum()
)

# --------------------------------------------------
# 8. TARGET VARIABLE DISTRIBUTION
# --------------------------------------------------

print("\n========== TARGET VARIABLE DISTRIBUTION ==========")

print(df[target].describe().round(2).to_string())

print("Target skewness:", round(df[target].skew(), 3))

# --------------------------------------------------
# 9. NUMERIC CORRELATION WITH TARGET
# --------------------------------------------------

print("\n========== NUMERIC FEATURE CORRELATION ==========")

correlation_data = df[
    numeric_features + [target]
].corr(method="spearman")

target_correlation = (
    correlation_data[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(target_correlation.round(4).to_string())

target_correlation.to_csv(
    f"{RESULT_DIR}/objective3_feature_sales_correlation.csv"
)

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n========== OBJECTIVE 3 INSPECTION COMPLETED ==========")

print("Target variable:", target)
print("Potential predictors inspected:", len(potential_features))

print("\nGenerated result files:")
print("1. objective3_missing_values.csv")
print("2. objective3_feature_sales_correlation.csv")

print("\nNext step: Select suitable features and prepare the regression dataset.")
