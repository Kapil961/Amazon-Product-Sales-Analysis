
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

input_file = Path("data/raw/amazon_product_sales.csv")
output_folder = Path("data/processed")

output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

original_records = len(df)

print("\n========== DATA CLEANING PROCESS ==========")
print(f"Original records: {original_records}")

# --------------------------------------------------
# 2. CONVERT TIMESTAMP
# --------------------------------------------------

df["updated_at"] = pd.to_datetime(
    df["updated_at"],
    errors="coerce",
    utc=True
)

# --------------------------------------------------
# 3. REMOVE EXACT DUPLICATES
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = before_duplicates - len(df)

print(f"Exact duplicate rows removed: {duplicates_removed}")
print(f"Records after duplicate removal: {len(df)}")

# --------------------------------------------------
# 4. CREATE SNAPSHOT DATASET
# --------------------------------------------------

# Keep observations with valid timestamps.
snapshots = df.dropna(subset=["updated_at"]).copy()

snapshots = snapshots.sort_values(
    ["asin", "updated_at"]
)

snapshot_file = output_folder / "amazon_product_sales_snapshots.csv"

snapshots.to_csv(snapshot_file, index=False)

# --------------------------------------------------
# 5. IDENTIFY PRODUCTS WITHOUT VALID TIMESTAMPS
# --------------------------------------------------

all_asins = set(df["asin"].dropna().unique())

valid_asins = set(snapshots["asin"].dropna().unique())

asins_without_valid_timestamp = all_asins - valid_asins

# --------------------------------------------------
# 6. FIND LATEST OBSERVATION FOR EACH PRODUCT
# --------------------------------------------------

latest_timestamps = (
    snapshots.groupby("asin")["updated_at"]
    .max()
    .rename("latest_timestamp")
)

latest_rows = snapshots.merge(
    latest_timestamps,
    on="asin",
    how="inner"
)

latest_rows = latest_rows[
    latest_rows["updated_at"] == latest_rows["latest_timestamp"]
].copy()

latest_rows = latest_rows.drop(columns=["latest_timestamp"])

# --------------------------------------------------
# 7. IDENTIFY PRODUCTS WITH MULTIPLE LATEST RECORDS
# --------------------------------------------------

latest_counts = latest_rows.groupby("asin").size()

tied_asins = latest_counts[
    latest_counts > 1
].index

conflict_data = latest_rows[
    latest_rows["asin"].isin(tied_asins)
].copy()

# Save conflicting latest records separately.
conflict_file = output_folder / "amazon_product_sales_conflicts.csv"

conflict_data.to_csv(conflict_file, index=False)

# --------------------------------------------------
# 8. CREATE FINAL PRODUCT-LEVEL DATASET
# --------------------------------------------------

# Remove products with ambiguous latest observations.
final_data = latest_rows[
    ~latest_rows["asin"].isin(tied_asins)
].copy()

final_data = final_data.sort_values("asin")

final_file = output_folder / "amazon_product_sales_latest.csv"

final_data.to_csv(final_file, index=False)

# --------------------------------------------------
# 9. FINAL VALIDATION
# --------------------------------------------------

print("\n========== CLEANING SUMMARY ==========")

print(f"Original records: {original_records}")
print(f"Exact duplicates removed: {duplicates_removed}")
print(f"Records after duplicate removal: {len(df)}")

print(f"\nUnique products: {len(all_asins)}")
print(f"Products with valid timestamps: {len(valid_asins)}")
print(
    "Products without valid timestamps:",
    len(asins_without_valid_timestamp)
)

print(f"\nProducts with multiple latest records: {len(tied_asins)}")
print(f"Records saved for conflict investigation: {len(conflict_data)}")

print("\n========== FINAL DATASET ==========")

print(f"Final product records: {len(final_data)}")
print(f"Unique ASINs in final dataset: {final_data['asin'].nunique()}")

print(
    "Duplicate ASINs in final dataset:",
    final_data["asin"].duplicated().sum()
)

print(
    "Missing timestamps in final dataset:",
    final_data["updated_at"].isna().sum()
)

print("\n========== GENERATED FILES ==========")

print(f"1. {snapshot_file}")
print(f"2. {conflict_file}")
print(f"3. {final_file}")

print("\nData cleaning completed successfully!")
