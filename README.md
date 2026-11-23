
# Statistical Analysis of Amazon Product Sales

A Python-based statistical analysis and machine learning project that studies Amazon product sales performance, explores the relationship between customer reviews and sales, and predicts monthly product sales using regression models.

## Project Overview

This project analyses Amazon product-level data to understand sales patterns, investigate relationships between product characteristics and sales, and develop a machine learning model for estimating monthly product sales.

The project uses Python, statistical techniques, data visualization, and machine learning.

**Final dataset:** 4,732 unique products.

## Research Objectives

### Objective 1: Amazon Product Sales Performance Analysis

- Analyse monthly product sales and revenue.
- Compare sales performance across product categories.
- Study the relationship between product price and monthly sales.
- Investigate the relationship between Best Sellers Rank (BSR) and monthly sales.
- Apply descriptive statistics, correlation analysis, and category comparison tests.

### Objective 2: Relationship Between Customer Reviews and Product Sales

- Examine the relationship between customer ratings and monthly sales.
- Analyse how review counts relate to monthly sales.
- Investigate relationships between ratings, reviews, and monthly revenue.
- Apply Pearson correlation, Spearman rank correlation, and hypothesis testing.

### Objective 3: Factors Affecting and Predicting Amazon Product Sales

- Identify product characteristics associated with monthly sales.
- Apply multiple linear regression.
- Compare original Linear Regression, improved Linear Regression, and Ridge Regression.
- Evaluate models using MAE, RMSE, and R-squared.

## Dataset Information

**Dataset:** Tikool/amazon-intel-sample

**Source:** https://huggingface.co/datasets/Tikool/amazon-intel-sample

The original dataset contained 10,000 records and 17 columns.

After data cleaning and timestamp-based snapshot investigation:

- Original records: 10,000
- Exact duplicate records removed: 92
- Unique products: 4,738
- Products without any valid timestamp: 2
- Ambiguous latest observations separated: 4 ASINs
- Final primary analysis dataset: 4,732 products

The project retains the original raw dataset and stores cleaned and processed datasets separately.

### Important Dataset Fields

| Column | Description |
|---|---|
| asin | Unique product identifier |
| title | Product title |
| brand | Product brand |
| price | Product price |
| bsr | Best Sellers Rank |
| reviews | Number of customer reviews |
| rating | Customer rating |
| monthly_sales | Estimated monthly product sales |
| monthly_revenue | Estimated monthly revenue |
| profit_estimate | Estimated profit |
| categories | Product categories |
| is_prime | Prime indicator |
| listing_quality_score | Listing quality score |
| grade | Product/listing grade |
| updated_at | Observation timestamp |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Scikit-learn
- Scikit-posthocs
- Joblib
- Jupyter Notebook

## Statistical Techniques

- Descriptive Statistics
- Pearson Correlation
- Spearman Rank Correlation
- Hypothesis Testing
- Bonferroni Correction
- Kruskal-Wallis Test
- Dunn Post-hoc Test
- Multiple Linear Regression
- Ridge Regression
- Variance Inflation Factor (VIF)
- Breusch-Pagan Test
- Residual Analysis
- Cook's Distance and Influence Analysis

## Project Results

### Objective 1: Sales Performance Analysis

- Products analysed: 4,732
- Eligible product categories: 56
- Price vs monthly sales:
  - Pearson correlation: -0.2331
  - Spearman correlation: -0.8439
- BSR vs monthly sales:
  - Pearson correlation: -0.1696
  - Spearman correlation: -0.7345
- Kruskal-Wallis test: p < 0.001

The results show negative associations between price and sales, and between numerical BSR and sales. The category test indicates differences in sales distributions across categories.

### Objective 2: Customer Reviews and Sales

| Relationship | Pearson | Spearman |
|---|---:|---:|
| Rating vs Monthly Sales | 0.2441 | 0.2657 |
| Reviews vs Monthly Sales | 0.4395 | 0.5689 |
| Rating vs Monthly Revenue | 0.0173 | 0.0529 |
| Reviews vs Monthly Revenue | 0.0159 | 0.1332 |

Review counts showed the strongest positive association with monthly sales among the customer-review variables examined.

### Objective 3: Sales Prediction

Three regression models were compared.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Original Linear Regression | 31,611.89 | 46,778.96 | 0.3326 |
| Improved Linear Regression | 31,161.20 | 46,453.13 | 0.3419 |
| Ridge Regression | 31,145.91 | 46,374.44 | 0.3441 |

**Selected Final Model: Ridge Regression**

- Selected alpha: 10
- Training records: 3,785
- Testing records: 947
- Test MAE: 31,145.91
- Test RMSE: 46,374.44
- Test R²: 0.3441

The final Ridge Regression model performed best among the three evaluated models.

The model explains approximately 34.41% of the variation in monthly sales in the test dataset. It provides a useful baseline but has difficulty predicting products with exceptionally high sales.

## Project Structure

```text
Amazon_Product_Sales_Analysis/
│
├── data/
│   ├── raw/
│   │   └── amazon_product_sales.csv
│   │
│   └── processed/
│       ├── amazon_product_sales_latest.csv
│       ├── amazon_product_sales_conflicts.csv
│       ├── amazon_product_sales_snapshots.csv
│       └── objective3_modeling_data.csv
│
├── models/
│   └── final_ridge_sales_model.joblib
│
├── notebooks/
│
├── outputs/
│   ├── graphs/
│   └── results/
│
├── src/
│   ├── clean_data.py
│   ├── data_exploration.py
│   ├── data_quality_investigation.py
│   ├── snapshot_analysis.py
│   ├── validate_cleaned_data.py
│   ├── inspect_objective_2.py
│   ├── objective_2_reviews_sales.py
│   ├── objective_1_sales_performance.py
│   ├── inspect_objective_3.py
│   ├── prepare_objective_3.py
│   ├── objective_3_regression.py
│   ├── objective_3_model_diagnostics.py
│   ├── objective_3_influence_analysis.py
│   ├── objective_3_model_improvement.py
│   └── final_objective_3_model.py
│
├── check_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd Amazon_Product_Sales_Analysis
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run the scripts from the project root directory.

### Data Cleaning

```bash
python src/clean_data.py
```

### Dataset Validation

```bash
python src/validate_cleaned_data.py
```

### Objective 1

```bash
python src/objective_1_sales_performance.py
```

### Objective 2

```bash
python src/objective_2_reviews_sales.py
```

### Objective 3

```bash
python src/prepare_objective_3.py
python src/objective_3_regression.py
python src/objective_3_model_diagnostics.py
python src/objective_3_influence_analysis.py
python src/objective_3_model_improvement.py
python src/final_objective_3_model.py
```

Generated graphs and result files are saved inside the `outputs/` directory.

## Final Model

The final trained Ridge Regression model is saved at:

```text
models/final_ridge_sales_model.joblib
```

The model can be loaded using Joblib:

```python
import joblib

model = joblib.load("models/final_ridge_sales_model.joblib")
```

The model expects the same input features and preprocessing structure used during training. Apply a non-negative prediction constraint when using its raw predictions:

```python
import numpy as np

predictions = np.maximum(model.predict(X), 0)
```

## Limitations

- The dataset contains product-level estimates rather than verified transaction-level sales.
- The analysis is observational and does not establish causation.
- Some product attributes contain missing values.
- Product category assignments can overlap.
- The final model has moderate explanatory power.
- The model tends to underestimate some exceptionally high-selling products.
- BSR is a contemporaneous measure and may reflect current sales performance; this model should not be described as a fully prospective sales forecast.

## Conclusion

This project combines statistical analysis and machine learning to study Amazon product sales.

The three objectives examine sales performance, customer review relationships, and factors associated with monthly sales.

Ridge Regression achieved the best test performance among the evaluated regression models, with an R² of 0.3441.

The project demonstrates the use of data cleaning, statistical testing, visualization, regression modelling, and model evaluation in a practical e-commerce analysis.

## Dataset Attribution and Licensing

Dataset: Tikool/amazon-intel-sample

Source: https://huggingface.co/datasets/Tikool/amazon-intel-sample

The dataset page lists the licence as **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

Please refer to the original dataset page and licence terms when redistributing or adapting the dataset.

The dataset's licence is separate from the licence applicable to original project code and documentation.

## Author

**[Kapil Kumawat]**

Project: Statistical Analysis of Amazon Product Sales
