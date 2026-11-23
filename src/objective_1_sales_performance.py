
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kruskal
from itertools import combinations

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

file_path = "data/processed/amazon_product_sales_latest.csv"

df = pd.read_csv(file_path)

graph_folder = Path("outputs/graphs")
result_folder = Path("outputs/results")

graph_folder.mkdir(parents=True, exist_ok=True)
result_folder.mkdir(parents=True, exist_ok=True)

print("\n========== OBJECTIVE 1: SALES PERFORMANCE ANALYSIS ==========")

print("Total products:", len(df))

MIN_CATEGORY_SIZE = 30


# --------------------------------------------------
# 2. PREPARE ANALYSIS DATA
# --------------------------------------------------

analysis_df = df.copy()

price_data = analysis_df[
    (analysis_df["price"] > 0) &
    analysis_df["price"].notna() &
    analysis_df["monthly_sales"].notna()
].copy()

bsr_data = analysis_df[
    (analysis_df["bsr"] > 0) &
    analysis_df["bsr"].notna() &
    analysis_df["monthly_sales"].notna()
].copy()

print("\n========== DATA PREPARATION ==========")

print("Products available:", len(analysis_df))
print("Valid positive prices:", len(price_data))
print("Valid BSR observations:", len(bsr_data))


# --------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n========== SALES DESCRIPTIVE STATISTICS ==========")

sales_summary = analysis_df[
    ["monthly_sales", "monthly_revenue"]
].describe().round(2)

print(sales_summary)

sales_summary.to_csv(
    result_folder / "objective1_sales_descriptive_statistics.csv"
)


# --------------------------------------------------
# 4. SALES DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=analysis_df,
    x="monthly_sales",
    bins=40,
    kde=True
)

plt.title("Distribution of Monthly Product Sales")
plt.xlabel("Monthly Sales")
plt.ylabel("Number of Products")
plt.tight_layout()

plt.savefig(
    graph_folder / "monthly_sales_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 5. REVENUE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=analysis_df,
    x="monthly_revenue",
    bins=40,
    kde=True
)

plt.title("Distribution of Monthly Product Revenue")
plt.xlabel("Monthly Revenue (Dataset Units)")
plt.ylabel("Number of Products")
plt.tight_layout()

plt.savefig(
    graph_folder / "monthly_revenue_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 6. CATEGORY DATA PREPARATION
# --------------------------------------------------

def parse_categories(value):

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(category).strip()
                for category in parsed
                if str(category).strip()
            ]

        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []

    except (ValueError, SyntaxError):
        return [str(value).strip()]

    return []


category_df = analysis_df.copy()

category_df["category_list"] = category_df[
    "categories"
].apply(parse_categories)

category_df = category_df.explode("category_list")

category_df = category_df.dropna(
    subset=["category_list"]
)

category_df = category_df[
    category_df["category_list"].str.strip() != ""
]

print("\n========== CATEGORY ANALYSIS ==========")

print(
    "Products represented in category records:",
    category_df["asin"].nunique()
)

print(
    "Number of category records:",
    len(category_df)
)


# --------------------------------------------------
# 7. CATEGORY SALES AND REVENUE COMPARISON
# --------------------------------------------------

category_summary = (
    category_df.groupby("category_list")
    .agg(
        product_count=("asin", "nunique"),
        average_sales=("monthly_sales", "mean"),
        median_sales=("monthly_sales", "median"),
        average_revenue=("monthly_revenue", "mean"),
        median_revenue=("monthly_revenue", "median")
    )
    .round(2)
    .sort_values("average_sales", ascending=False)
)

# Save complete category results.
category_summary.to_csv(
    result_folder / "objective1_category_performance.csv"
)

# Only use categories with at least 30 products
# for the reliable category ranking.
eligible_category_summary = category_summary[
    category_summary["product_count"] >= MIN_CATEGORY_SIZE
].copy()

eligible_category_summary = eligible_category_summary.sort_values(
    "average_sales",
    ascending=False
)

print("\nMinimum products required per category:", MIN_CATEGORY_SIZE)

print(
    "Categories meeting minimum sample size:",
    len(eligible_category_summary)
)

print("\nTop 10 Categories by Average Monthly Sales:")

print(eligible_category_summary.head(10))


# Save filtered category results separately.
eligible_category_summary.to_csv(
    result_folder / "objective1_eligible_category_performance.csv"
)


# --------------------------------------------------
# 8. TOP CATEGORY SALES GRAPH
# --------------------------------------------------

top_categories = (
    eligible_category_summary
    .head(10)
    .reset_index()
)

if not top_categories.empty:

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=top_categories,
        x="average_sales",
        y="category_list"
    )

    plt.title(
        "Top 10 Categories by Average Monthly Sales\n"
        "(Minimum 30 Products per Category)"
    )

    plt.xlabel("Average Monthly Sales")
    plt.ylabel("Category")
    plt.tight_layout()

    plt.savefig(
        graph_folder / "top_10_categories_sales.png",
        dpi=300
    )

    plt.close()

else:
    print("No categories meet the minimum sample-size requirement.")


# --------------------------------------------------
# 9. CORRELATION ANALYSIS FUNCTION
# --------------------------------------------------

def format_p_value(p_value):

    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.6f}"


def correlation_analysis(data, x_column, y_column, relationship):

    clean_data = data[
        [x_column, y_column]
    ].dropna()

    pearson_result = pearsonr(
        clean_data[x_column],
        clean_data[y_column]
    )

    spearman_result = spearmanr(
        clean_data[x_column],
        clean_data[y_column]
    )

    print(f"\n{relationship}")
    print("Sample size:", len(clean_data))

    print(
        "Pearson correlation:",
        round(pearson_result.statistic, 4)
    )

    print(
        "Pearson p-value:",
        format_p_value(pearson_result.pvalue)
    )

    print(
        "Spearman correlation:",
        round(spearman_result.statistic, 4)
    )

    print(
        "Spearman p-value:",
        format_p_value(spearman_result.pvalue)
    )

    return {
        "Relationship": relationship,
        "Sample Size": len(clean_data),
        "Pearson Correlation": pearson_result.statistic,
        "Pearson P-value": pearson_result.pvalue,
        "Pearson Significant": pearson_result.pvalue < 0.05,
        "Spearman Correlation": spearman_result.statistic,
        "Spearman P-value": spearman_result.pvalue,
        "Spearman Significant": spearman_result.pvalue < 0.05
    }


# --------------------------------------------------
# 10. PRICE AND MONTHLY SALES RELATIONSHIP
# --------------------------------------------------

print("\n========== PRICE AND SALES RELATIONSHIP ==========")

price_results = correlation_analysis(
    price_data,
    "price",
    "monthly_sales",
    "Price vs Monthly Sales"
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=price_data,
    x="price",
    y="monthly_sales",
    alpha=0.5
)

plt.title("Product Price vs Monthly Sales")
plt.xlabel("Price")
plt.ylabel("Monthly Sales")
plt.tight_layout()

plt.savefig(
    graph_folder / "price_vs_monthly_sales.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 11. BSR AND MONTHLY SALES RELATIONSHIP
# --------------------------------------------------

print("\n========== BSR AND SALES RELATIONSHIP ==========")

bsr_results = correlation_analysis(
    bsr_data,
    "bsr",
    "monthly_sales",
    "BSR vs Monthly Sales"
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=bsr_data,
    x="bsr",
    y="monthly_sales",
    alpha=0.5
)

plt.title("Best Sellers Rank vs Monthly Sales")
plt.xlabel("Best Sellers Rank")
plt.ylabel("Monthly Sales")
plt.tight_layout()

plt.savefig(
    graph_folder / "bsr_vs_monthly_sales.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 12. KRUSKAL-WALLIS CATEGORY TEST
# --------------------------------------------------

print("\n========== CATEGORY DIFFERENCE TEST ==========")

eligible_categories = eligible_category_summary.index.tolist()

category_groups = []

for category in eligible_categories:

    values = category_df.loc[
        category_df["category_list"] == category,
        "monthly_sales"
    ].dropna()

    category_groups.append(values)

test_result = None

if len(category_groups) >= 2:

    test_result = kruskal(*category_groups)

    print("Categories tested:", len(category_groups))

    print(
        "Kruskal-Wallis statistic:",
        round(test_result.statistic, 4)
    )

    print(
        "P-value:",
        format_p_value(test_result.pvalue)
    )

    print(
        "Statistically significant:",
        test_result.pvalue < 0.05
    )

else:

    print("Not enough eligible categories for this test.")


# --------------------------------------------------
# 13. POST-HOC CATEGORY COMPARISON
# --------------------------------------------------

print("\n========== POST-HOC CATEGORY COMPARISON ==========")

posthoc_results = []

if test_result is not None and test_result.pvalue < 0.05:

    try:
        import scikit_posthocs as sp

        posthoc_data = category_df[
            category_df["category_list"].isin(eligible_categories)
        ][
            ["asin", "category_list", "monthly_sales"]
        ].dropna()

        # The same product may appear in multiple categories.
        # Dunn's test is exploratory here because category groups
        # may not be independent.

        posthoc_matrix = sp.posthoc_dunn(
            posthoc_data,
            val_col="monthly_sales",
            group_col="category_list",
            p_adjust="bonferroni"
        )

        posthoc_matrix.to_csv(
            result_folder / "objective1_category_posthoc_matrix.csv"
        )

        print("Dunn's post-hoc test completed.")
        print("Bonferroni correction applied.")

        for category_a, category_b in combinations(
            posthoc_matrix.index, 2
        ):

            p_value = posthoc_matrix.loc[
                category_a, category_b
            ]

            posthoc_results.append({
                "Category 1": category_a,
                "Category 2": category_b,
                "Adjusted P-value": p_value,
                "Significant": p_value < 0.05
            })

        posthoc_df = pd.DataFrame(posthoc_results)

        posthoc_df.to_csv(
            result_folder / "objective1_category_posthoc_pairs.csv",
            index=False
        )

        print(
            "Significant category pairs:",
            int(posthoc_df["Significant"].sum())
        )

    except ImportError:

        print(
            "Post-hoc test not available. "
            "Install scikit-posthocs using:"
        )

        print("pip install scikit-posthocs")

else:

    print(
        "Post-hoc test skipped because the overall test "
        "was not statistically significant or had too few groups."
    )


# --------------------------------------------------
# 14. SAVE CORRELATION RESULTS
# --------------------------------------------------

correlation_results = pd.DataFrame([
    price_results,
    bsr_results
])

correlation_results.to_csv(
    result_folder / "objective1_correlation_results.csv",
    index=False
)


# --------------------------------------------------
# 15. FINAL SUMMARY
# --------------------------------------------------

print("\n========== OBJECTIVE 1 COMPLETED ==========")

print("Total products analysed:", len(analysis_df))

print(
    "Categories meeting minimum sample size:",
    len(eligible_category_summary)
)

print("\nGenerated graphs:")

print("1. monthly_sales_distribution.png")
print("2. monthly_revenue_distribution.png")
print("3. top_10_categories_sales.png")
print("4. price_vs_monthly_sales.png")
print("5. bsr_vs_monthly_sales.png")

print("\nGenerated result files:")

print("1. objective1_sales_descriptive_statistics.csv")
print("2. objective1_category_performance.csv")
print("3. objective1_eligible_category_performance.csv")
print("4. objective1_correlation_results.csv")

if test_result is not None and test_result.pvalue < 0.05:
    print("5. objective1_category_posthoc_matrix.csv")
    print("6. objective1_category_posthoc_pairs.csv")

print("\nObjective 1 analysis completed!")
