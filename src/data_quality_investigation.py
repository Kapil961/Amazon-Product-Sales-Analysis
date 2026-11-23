
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/amazon_product_sales.csv")

print("\n========== 1. ASIN DUPLICATE INVESTIGATION ==========")

print("Total unique ASINs:", df["asin"].nunique())

print("Number of ASINs appearing multiple times:",
      df["asin"].value_counts().gt(1).sum())

print("\nMost repeated ASINs:")
print(df["asin"].value_counts().head(10))

print("\nExample of repeated ASIN records:")
repeated_asin = df["asin"].value_counts().loc[lambda x: x > 1].index[:3]

print(df[df["asin"].isin(repeated_asin)][
    ["asin", "price", "monthly_sales", "monthly_revenue", "updated_at"]
].to_string(index=False))


print("\n========== 2. BSR INVESTIGATION ==========")

print("Negative BSR:", (df["bsr"] < 0).sum())
print("Zero BSR:", (df["bsr"] == 0).sum())

print("\nBSR values below zero:")
print(df[df["bsr"] < 0][["asin", "bsr"]].head(10))


print("\n========== 3. PRICE INVESTIGATION ==========")

print("Zero price records:")
print(df[df["price"] == 0][
    ["asin", "title", "price", "monthly_sales"]
].head(10).to_string(index=False))


print("\n========== 4. REVENUE INVESTIGATION ==========")

df["calculated_revenue"] = (
    df["price"] * df["monthly_sales"]
)

df["revenue_difference"] = (
    df["monthly_revenue"] - df["calculated_revenue"]
)

print("Exact revenue matches:",
      np.isclose(df["monthly_revenue"],
                 df["calculated_revenue"]).sum())

print("Revenue mismatches:",
      (~np.isclose(df["monthly_revenue"],
                   df["calculated_revenue"])).sum())

print("\nLargest revenue differences:")

print(
    df.assign(
        absolute_difference=df["revenue_difference"].abs()
    ).nlargest(10, "absolute_difference")[
        ["asin", "price", "monthly_sales",
         "monthly_revenue", "calculated_revenue",
         "revenue_difference"]
    ].to_string(index=False)
)


print("\n========== 5. DATE INVESTIGATION ==========")

df["updated_at"] = pd.to_datetime(
    df["updated_at"], errors="coerce", utc=True
)

print("Unique update timestamps:",
      df["updated_at"].nunique())

print("Earliest update:", df["updated_at"].min())
print("Latest update:", df["updated_at"].max())

print("\nInvestigation completed!")
