
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

DATA_PATH = "data/processed/amazon_product_sales_latest.csv"
GRAPH_DIR = "outputs/graphs"
RESULT_DIR = "outputs/results"

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("\n========== OBJECTIVE 2: REVIEWS AND SALES ANALYSIS ==========")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

columns = [
    "rating",
    "reviews",
    "monthly_sales",
    "monthly_revenue"
]

data = df[columns].copy()

print("\nTotal products analysed:", len(data))

# --------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n========== DESCRIPTIVE STATISTICS ==========")

descriptive = data.describe().round(2)
print(descriptive.to_string())

descriptive.to_csv(
    f"{RESULT_DIR}/objective2_descriptive_statistics.csv"
)

# --------------------------------------------------
# 3. CORRELATION ANALYSIS
# --------------------------------------------------

relationships = [
    ("rating", "monthly_sales"),
    ("reviews", "monthly_sales"),
    ("rating", "monthly_revenue"),
    ("reviews", "monthly_revenue")
]

results = []

print("\n========== CORRELATION ANALYSIS ==========")

for x, y in relationships:

    subset = data[[x, y]].dropna()

    pearson_result = pearsonr(subset[x], subset[y])
    spearman_result = spearmanr(subset[x], subset[y])

    results.append({
        "variable_1": x,
        "variable_2": y,
        "sample_size": len(subset),
        "pearson_correlation": pearson_result.statistic,
        "pearson_p_value": pearson_result.pvalue,
        "spearman_correlation": spearman_result.statistic,
        "spearman_p_value": spearman_result.pvalue
    })

    print(f"\n{x} vs {y}")
    print("Sample size:", len(subset))
    print("Pearson correlation:", round(pearson_result.statistic, 4))
    print("Pearson p-value:", f"{pearson_result.pvalue:.6g}")
    print("Spearman correlation:", round(spearman_result.statistic, 4))
    print("Spearman p-value:", f"{spearman_result.pvalue:.6g}")

# --------------------------------------------------
# 4. STATISTICAL SIGNIFICANCE

results_df = pd.DataFrame(results)

results_df["pearson_significant"] = (
    results_df["pearson_p_value"] < 0.05
)

results_df["spearman_significant"] = (
    results_df["spearman_p_value"] < 0.05
)

results_df.to_csv(
    f"{RESULT_DIR}/objective2_correlation_results.csv",
    index=False
)

print("\n========== STATISTICAL SIGNIFICANCE ==========")

print("Significance level: 0.05")
print(
    results_df[
        [
            "variable_1",
            "variable_2",
            "pearson_p_value",
            "spearman_p_value",
            "pearson_significant",
            "spearman_significant"
        ]
    ].round(6).to_string(index=False)
)

# 5. VISUALIZATION
# --------------------------------------------------

print("\n========== GENERATING GRAPHS ==========")

plot_relationships = [
    ("rating", "monthly_sales"),
    ("reviews", "monthly_sales"),
    ("rating", "monthly_revenue"),
    ("reviews", "monthly_revenue")
]

for x, y in plot_relationships:

    plot_data = data[[x, y]].dropna().copy()

    if x == "reviews":
        plot_data[x] = np.log1p(plot_data[x])

    plot_data[y] = np.log1p(plot_data[y])

    plt.figure(figsize=(9, 6))

    plt.scatter(
        plot_data[x],
        plot_data[y],
        alpha=0.35,
        s=18
    )

    x_label = (
        "Log(Review Count + 1)"
        if x == "reviews"
        else "Customer Rating"
    )

    y_label = (
        "Log(Monthly Sales + 1)"
        if y == "monthly_sales"
        else "Log(Monthly Revenue + 1)"
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x} vs {y}")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    filename = f"objective2_{x}_vs_{y}.png"

    plt.savefig(
        f"{GRAPH_DIR}/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Generated:", filename)

# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

print("\n========== OBJECTIVE 2 COMPLETED ==========")

print("Total products analysed:", len(data))

print("\nGenerated result files:")
print("1. objective2_descriptive_statistics.csv")
print("2. objective2_correlation_results.csv")

print("\nGenerated graphs:")
print("1. objective2_rating_vs_monthly_sales.png")
print("2. objective2_reviews_vs_monthly_sales.png")
print("3. objective2_rating_vs_monthly_revenue.png")
print("4. objective2_reviews_vs_monthly_revenue.png")

print("\nObjective 2 analysis completed successfully!")
