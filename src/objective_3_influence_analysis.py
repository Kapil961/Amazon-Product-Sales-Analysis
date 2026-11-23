
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA_PATH = "data/processed/objective3_modeling_data.csv"
PRODUCT_PATH = "data/processed/amazon_product_sales_latest.csv"

RESULTS_DIR = "outputs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "monthly_sales"

NUMERIC_FEATURES = [
    "price",
    "bsr",
    "rating",
    "reviews",
    "listing_quality_score"
]

CATEGORICAL_FEATURES = [
    "is_prime",
    "grade"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\n========== OBJECTIVE 3: INFLUENCE ANALYSIS ==========")

df = pd.read_csv(DATA_PATH)
products = pd.read_csv(PRODUCT_PATH)

if len(df) != len(products):
    raise ValueError(
        "The modeling dataset and product dataset have different "
        "row counts. Cannot safely match product identifiers."
    )

if not np.allclose(
    df[TARGET].to_numpy(dtype=float),
    products[TARGET].to_numpy(dtype=float),
    equal_nan=True
):
    raise ValueError(
        "The product rows do not match the modeling dataset order."
    )

X = df[FEATURES].copy()
y = df[TARGET].copy()

valid_rows = y.notna() & (y >= 0)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

products = products.loc[valid_rows].copy()

print("Total products:", len(X))


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

products_train = products.loc[X_train.index].copy()
products_test = products.loc[X_test.index].copy()

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])


# --------------------------------------------------
# 5. FIT ORIGINAL LINEAR REGRESSION
# --------------------------------------------------

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

if hasattr(X_train_encoded, "toarray"):
    X_train_encoded = X_train_encoded.toarray()

if hasattr(X_test_encoded, "toarray"):
    X_test_encoded = X_test_encoded.toarray()

model = LinearRegression()
model.fit(X_train_encoded, y_train)

train_predictions = model.predict(X_train_encoded)
test_predictions = model.predict(X_test_encoded)

print("\nModel fitted successfully.")


# --------------------------------------------------
# 6. TRAINING SET INFLUENCE ANALYSIS
# --------------------------------------------------

print("\n========== COOK'S DISTANCE AND LEVERAGE ==========")

X_train_sm = sm.add_constant(
    X_train_encoded,
    has_constant="add"
)

ols_model = sm.OLS(
    y_train.to_numpy(),
    X_train_sm
).fit()

influence = ols_model.get_influence()

cooks_distance = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
studentized_residuals = influence.resid_studentized_external

n = len(X_train)
p = X_train_sm.shape[1]

cooks_threshold = 4 / n
leverage_threshold = 2 * p / n

print("Training observations:", n)
print("Model parameters including intercept:", p)
print("Cook's distance threshold:", round(cooks_threshold, 6))
print("Leverage threshold:", round(leverage_threshold, 6))

print(
    "Observations above Cook's distance threshold:",
    int(np.sum(cooks_distance > cooks_threshold))
)

print(
    "Observations above leverage threshold:",
    int(np.sum(leverage > leverage_threshold))
)

print(
    "Observations with absolute studentized residual > 3:",
    int(np.sum(np.abs(studentized_residuals) > 3))
)


# --------------------------------------------------
# 7. CREATE TRAINING INFLUENCE REPORT
# --------------------------------------------------

train_report = products_train[
    ["asin", "title", "monthly_sales"]
].copy()

train_report["Predicted_Sales"] = train_predictions

train_report["Residual"] = (
    y_train.to_numpy() - train_predictions
)

train_report["Absolute_Error"] = np.abs(
    train_report["Residual"]
)

train_report["Cooks_Distance"] = cooks_distance

train_report["Leverage"] = leverage

train_report["Studentized_Residual"] = studentized_residuals

train_report["High_Cooks_Distance"] = (
    train_report["Cooks_Distance"] > cooks_threshold
)

train_report["High_Leverage"] = (
    train_report["Leverage"] > leverage_threshold
)

train_report["Extreme_Residual"] = (
    np.abs(train_report["Studentized_Residual"]) > 3
)

train_report["Potentially_Influential"] = (
    train_report["High_Cooks_Distance"]
    | train_report["High_Leverage"]
    | train_report["Extreme_Residual"]
)

train_report = train_report.sort_values(
    "Cooks_Distance",
    ascending=False
)

train_report.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_training_influence_report.csv"
    ),
    index=False
)


# --------------------------------------------------
# 8. TEST SET EXTREME PREDICTION ERRORS
# --------------------------------------------------

print("\n========== TEST SET PREDICTION ERRORS ==========")

test_report = products_test[
    ["asin", "title", "monthly_sales"]
].copy()

test_report["Predicted_Sales"] = test_predictions

test_report["Residual"] = (
    y_test.to_numpy() - test_predictions
)

test_report["Absolute_Error"] = np.abs(
    test_report["Residual"]
)

test_report = test_report.sort_values(
    "Absolute_Error",
    ascending=False
)

test_report.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_test_extreme_errors.csv"
    ),
    index=False
)

print("\nTop 10 products with highest prediction errors:")

print(
    test_report[
        [
            "asin",
            "monthly_sales",
            "Predicted_Sales",
            "Absolute_Error"
        ]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# 9. INFLUENCE SUMMARY
# --------------------------------------------------

print("\n========== INFLUENCE SUMMARY ==========")

print(
    "Potentially influential training observations:",
    int(train_report["Potentially_Influential"].sum())
)

print(
    "Percentage of training observations:",
    round(
        train_report["Potentially_Influential"].mean() * 100,
        2
    ),
    "%"
)

print("\nTop 10 observations by Cook's distance:")

print(
    train_report[
        [
            "asin",
            "Cooks_Distance",
            "Leverage",
            "Studentized_Residual"
        ]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# 10. SAVE SUMMARY
# --------------------------------------------------

summary = pd.DataFrame([{
    "Training_Records": n,
    "Model_Parameters": p,
    "Cooks_Distance_Threshold": cooks_threshold,
    "High_Cooks_Distance_Count": int(
        np.sum(cooks_distance > cooks_threshold)
    ),
    "High_Leverage_Count": int(
        np.sum(leverage > leverage_threshold)
    ),
    "Extreme_Residual_Count": int(
        np.sum(np.abs(studentized_residuals) > 3)
    ),
    "Potentially_Influential_Count": int(
        train_report["Potentially_Influential"].sum()
    ),
    "Potentially_Influential_Percentage": (
        train_report["Potentially_Influential"].mean() * 100
    )
}])

summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_influence_summary.csv"
    ),
    index=False
)

print("\n========== ANALYSIS COMPLETED ==========")

print("Generated files:")
print("- objective3_training_influence_report.csv")
print("- objective3_test_extreme_errors.csv")
print("- objective3_influence_summary.csv")

print("\nInfluence analysis completed successfully.")
