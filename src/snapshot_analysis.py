
import pandas as pd

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

df = pd.read_csv("data/raw/amazon_product_sales.csv")

df["updated_at"] = pd.to_datetime(
    df["updated_at"],
    errors="coerce",
    utc=True
)

# Remove exact duplicate rows
original_rows = len(df)
df = df.drop_duplicates().copy()
removed_duplicates = original_rows - len(df)

print("\n========== SNAPSHOT ANALYSIS ==========")

print("Original records:", original_rows)
print("Exact duplicate rows removed:", removed_duplicates)
print("Records after duplicate removal:", len(df))

# ==========================================
# 2. ASIN AND TIMESTAMP DUPLICATES
# ==========================================

print("\n========== TIMESTAMP DUPLICATE CHECK ==========")

duplicate_mask = df.duplicated(
    subset=["asin", "updated_at"],
    keep=False
)

duplicate_groups = (
    df.groupby(["asin", "updated_at"], dropna=False)
    .size()
)

multiple_record_groups = duplicate_groups[
    duplicate_groups > 1
]

print(
    "ASIN-timestamp groups with multiple records:",
    len(multiple_record_groups)
)

print(
    "Rows involved in these groups:",
    duplicate_mask.sum()
)

print(
    "Rows with missing timestamps:",
    df["updated_at"].isna().sum()
)

print(
    "ASINs with at least one missing timestamp:",
    df.loc[df["updated_at"].isna(), "asin"].nunique()
)

# ==========================================
# 3. LATEST OBSERVATION PER PRODUCT
# ==========================================

valid_dates = df.dropna(subset=["updated_at"]).copy()

latest_dates = (
    valid_dates.groupby("asin")["updated_at"]
    .max()
)

latest_rows = valid_dates.merge(
    latest_dates.rename("latest_timestamp"),
    left_on="asin",
    right_index=True
)

latest_rows = latest_rows[
    latest_rows["updated_at"] == latest_rows["latest_timestamp"]
].copy()

print("\n========== LATEST OBSERVATION ==========")

print("Unique products:", df["asin"].nunique())
print("Products with at least one valid timestamp:",
      valid_dates["asin"].nunique())

print("ASINs with no valid timestamp:",
      df["asin"].nunique() - valid_dates["asin"].nunique())

# ==========================================
# 4. PRODUCT VALUE VARIATION
# ==========================================

print("\n========== PRODUCT VARIATION ==========")

variation_columns = [
    "price",
    "monthly_sales",
    "rating",
    "reviews",
    "bsr",
    "monthly_revenue"
]

for col in variation_columns:
    variation = df.groupby("asin")[col].nunique(dropna=False)

    print(
        f"{col}:",
        (variation > 1).sum(),
        "products have changing values"
    )

# ==========================================
# 5. HISTORICAL TIMESTAMP CONFLICTS
# ==========================================

print("\n========== HISTORICAL CONFLICT INVESTIGATION ==========")

historical_groups = (
    df.groupby(["asin", "updated_at"], dropna=False)
    .filter(lambda group: len(group) > 1)
)

check_columns = [
    col for col in variation_columns
    if col in df.columns
]

historical_conflicts = (
    historical_groups
    .groupby(["asin", "updated_at"], dropna=False)[check_columns]
    .nunique(dropna=False)
)

historical_conflicts = historical_conflicts[
    (historical_conflicts > 1).any(axis=1)
]

print(
    "Historical groups with conflicting numeric values:",
    len(historical_conflicts)
)

print("\nHistorical conflicting groups:")
print(historical_conflicts)

# ==========================================
# 6. GLOBAL LATEST TIMESTAMP TIE CHECK
# ==========================================

print("\n========== GLOBAL LATEST TIMESTAMP TIE CHECK ==========")

latest_counts = latest_rows.groupby("asin").size()

tied_asins = latest_counts[
    latest_counts > 1
].index

print("Products with multiple latest records:", len(tied_asins))

if len(tied_asins) > 0:

    tied_records = latest_rows[
        latest_rows["asin"].isin(tied_asins)
    ].copy()

    columns_to_check = [
        col for col in df.columns
        if col not in ["asin", "updated_at"]
    ]

    latest_conflicts = (
        tied_records.groupby("asin")[columns_to_check]
        .nunique(dropna=False)
    )

    conflicting_asins = latest_conflicts[
        (latest_conflicts > 1).any(axis=1)
    ]

    print(
        "Products with conflicting latest values:",
        len(conflicting_asins)
    )

    print("\nConflicting latest ASINs:")
    print(conflicting_asins)

    # ======================================
    # 7. DETAILED INSPECTION OF LATEST TIES
    # ======================================

    print("\n========== DETAILED LATEST TIE INSPECTION ==========")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 100)

    for asin in tied_asins:

        print("\n" + "=" * 80)
        print("ASIN:", asin)

        product_records = tied_records[
            tied_records["asin"] == asin
        ]

        print(
            "Records at latest timestamp:",
            len(product_records)
        )

        print(
            product_records[df.columns].to_string(index=False)
        )

else:
    conflicting_asins = pd.Index([])
    print("No products have multiple latest records.")

# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n========== FINAL DATA QUALITY SUMMARY ==========")

print("Records after exact duplicate removal:", len(df))
print("Unique products:", df["asin"].nunique())
print("Products with valid timestamps:", valid_dates["asin"].nunique())
print("Products with no valid timestamps:",
      df["asin"].nunique() - valid_dates["asin"].nunique())

print("Products with multiple latest records:", len(tied_asins))

if len(tied_asins) > 0:
    print("Products with conflicting latest values:",
          len(conflicting_asins))
else:
    print("Products with conflicting latest values: 0")

print("\nSnapshot analysis completed!")
