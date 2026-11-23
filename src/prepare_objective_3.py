
import os
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/amazon_product_sales_latest.csv"
OUTPUT_PATH = "data/processed/objective3_modeling_data.csv"
RESULT_DIR = "outputs/results"

os.makedirs(RESULT_DIR, exist_ok=True)

print("\n========== OBJECTIVE 3: MODEL DATA PREPARATION ==========")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nOriginal products:", len(df))

# --------------------------------------------------
# 2. SELECT TARGET AND PREDICTORS
# --------------------------------------------------

features = [
    "price",
    "bsr",
    "rating",
    "reviews",
    "listing_quality_score",
    "is_prime",
    "grade"
]

target = "monthly_sales"

model_data = df[features + [target]].copy()

# --------------------------------------------------
# 3. HANDLE INVALID VALUES
# --------------------------------------------------

print("\n========== INVALID VALUE HANDLING ==========")

# Zero prices are treated as missing until verified.
zero_prices = (model_data["price"] == 0).sum()
model_data.loc[model_data["price"] == 0, "price"] = np.nan

print("Zero prices converted to missing:", zero_prices)

# Negative BSR values are invalid for this analysis.
invalid_bsr = (model_data["bsr"] < 0).sum()
model_data.loc[model_data["bsr"] < 0, "bsr"] = np.nan

print("Negative BSR values converted to missing:", invalid_bsr)

# --------------------------------------------------
# 4. HANDLE CATEGORICAL DATA TYPES
# --------------------------------------------------

print("\n========== CATEGORICAL PREPARATION ==========")

categorical_features = ["is_prime", "grade"]

for column in categorical_features:

    model_data[column] = (
        model_data[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    print(f"\n{column}")
    print("Unique categories:", model_data[column].nunique())
    print(model_data[column].value_counts().head(10).to_string())

# --------------------------------------------------
# 5. CHECK NUMERIC DATA
# --------------------------------------------------

print("\n========== NUMERIC FEATURE CHECK ==========")

numeric_features = [
    "price",
    "bsr",
    "rating",
    "reviews",
    "listing_quality_score"
]

print(
    model_data[numeric_features]
    .describe()
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 6. MISSING VALUE REPORT
# --------------------------------------------------

print("\n========== MISSING VALUE REPORT ==========")

missing_report = pd.DataFrame({
    "missing_count": model_data.isnull().sum(),
    "missing_percentage": model_data.isnull().mean() * 100
})

print(missing_report.round(2).to_string())

missing_report.to_csv(
    f"{RESULT_DIR}/objective3_modeling_missing_values.csv"
)

# --------------------------------------------------
# 7. TARGET VALIDATION
# --------------------------------------------------

print("\n========== TARGET VALIDATION ==========")

print("Missing target values:", model_data[target].isnull().sum())
print("Non-positive target values:", (model_data[target] <= 0).sum())

# --------------------------------------------------
# 8. SAVE MODELING DATASET
# --------------------------------------------------

model_data.to_csv(OUTPUT_PATH, index=False)

print("\n========== MODELING DATASET SAVED ==========")

print("Output file:", OUTPUT_PATH)
print("Total products:", len(model_data))
print("Total columns:", len(model_data.columns))

print("\nSelected features:")
print(features)

print("\nTarget variable:", target)

print("\nOBJECTIVE 3 DATA PREPARATION COMPLETED!")
