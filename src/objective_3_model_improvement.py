
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_PATH = "data/processed/objective3_modeling_data.csv"

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

print("\n========== OBJECTIVE 3: MODEL IMPROVEMENT ==========")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = pd.to_numeric(df[TARGET], errors="coerce")

valid_rows = y.notna() & (y >= 0)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("Total products:", len(X))
print("Predictor variables:", len(FEATURES))


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 4. ORIGINAL PREPROCESSING
# --------------------------------------------------

original_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

original_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

original_preprocessor = ColumnTransformer([
    ("numeric", original_numeric, NUMERIC_FEATURES),
    ("categorical", original_categorical, CATEGORICAL_FEATURES)
])


# --------------------------------------------------
# 5. IMPROVED PREPROCESSING
# --------------------------------------------------

improved_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

improved_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=10,
        drop="first"
    ))
])

improved_preprocessor = ColumnTransformer([
    ("numeric", improved_numeric, NUMERIC_FEATURES),
    ("categorical", improved_categorical, CATEGORICAL_FEATURES)
])


# --------------------------------------------------
# 6. MODEL 1: ORIGINAL LINEAR REGRESSION
# --------------------------------------------------

print("\n========== MODEL 1: ORIGINAL LINEAR REGRESSION ==========")

original_model = Pipeline([
    ("preprocessor", original_preprocessor),
    ("regressor", LinearRegression())
])

original_model.fit(X_train, y_train)

original_predictions = original_model.predict(X_test)

original_predictions = np.maximum(original_predictions, 0)

print("Original model fitted successfully.")


# --------------------------------------------------
# 7. MODEL 2: IMPROVED LINEAR REGRESSION
# --------------------------------------------------

print("\n========== MODEL 2: IMPROVED LINEAR REGRESSION ==========")

improved_linear_model = Pipeline([
    ("preprocessor", improved_preprocessor),
    ("regressor", LinearRegression())
])

improved_linear_model.fit(X_train, y_train)

improved_linear_predictions = improved_linear_model.predict(X_test)

improved_linear_predictions = np.maximum(
    improved_linear_predictions, 0
)

print("Improved linear model fitted successfully.")


# --------------------------------------------------
# 8. MODEL 3: RIDGE REGRESSION
# --------------------------------------------------

print("\n========== MODEL 3: RIDGE REGRESSION ==========")

ridge_pipeline = Pipeline([
    ("preprocessor", improved_preprocessor),
    ("regressor", Ridge())
])

parameter_grid = {
    "regressor__alpha": [0.1, 1, 10, 100, 500]
}

ridge_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=parameter_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1
)

ridge_search.fit(X_train, y_train)

best_ridge_model = ridge_search.best_estimator_

ridge_predictions = best_ridge_model.predict(X_test)

ridge_predictions = np.maximum(ridge_predictions, 0)

print("Best Ridge model fitted successfully.")
print("Best alpha:", ridge_search.best_params_["regressor__alpha"])
print(
    "Best cross-validation MAE:",
    round(-ridge_search.best_score_, 2)
)


# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

models = {
    "Original Linear Regression": original_predictions,
    "Improved Linear Regression": improved_linear_predictions,
    "Ridge Regression": ridge_predictions
}

comparison_results = []

for model_name, predictions in models.items():

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    comparison_results.append({
        "Model": model_name,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2
    })

comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    "Test_MAE",
    ascending=True
)

comparison_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_model_improvement_comparison.csv"
    ),
    index=False
)

print(comparison_df.to_string(index=False))


# --------------------------------------------------
# 10. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = comparison_df.iloc[0]["Model"]

print("\nBest model based on test MAE:", best_model_name)


# --------------------------------------------------
# 11. SAVE TEST PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual_Monthly_Sales": y_test.to_numpy(),
    "Original_Linear_Predictions": original_predictions,
    "Improved_Linear_Predictions": improved_linear_predictions,
    "Ridge_Predictions": ridge_predictions
})

prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_improved_model_predictions.csv"
    ),
    index=False
)


# --------------------------------------------------
# 12. SAVE RIDGE TUNING RESULTS
# --------------------------------------------------

ridge_cv_results = pd.DataFrame(ridge_search.cv_results_)

ridge_cv_results = ridge_cv_results[
    [
        "param_regressor__alpha",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].copy()

ridge_cv_results["Mean_CV_MAE"] = (
    -ridge_cv_results["mean_test_score"]
)

ridge_cv_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_ridge_tuning_results.csv"
    ),
    index=False
)


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n========== ANALYSIS COMPLETED ==========")

print("Total products retained:", len(X))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nGenerated files:")
print("- objective3_model_improvement_comparison.csv")
print("- objective3_improved_model_predictions.csv")
print("- objective3_ridge_tuning_results.csv")

print("\nModel improvement analysis completed successfully.")
