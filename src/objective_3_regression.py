
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA_PATH = "data/processed/objective3_modeling_data.csv"
RESULT_DIR = "outputs/results"

os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

numeric_features = [
    "price",
    "bsr",
    "rating",
    "reviews",
    "listing_quality_score"
]

categorical_features = [
    "is_prime",
    "grade"
]

target = "monthly_sales"

print("\n========== OBJECTIVE 3: MULTIPLE LINEAR REGRESSION ==========")

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

print("\nTotal products:", len(df))
print("Predictor variables:", len(numeric_features) + len(categorical_features))
print("Target variable:", target)

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# --------------------------------------------------
# 5. FUNCTION TO BUILD AND EVALUATE MODELS
# --------------------------------------------------

def evaluate_model(model_name, log_transform=False):

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    if log_transform:
        y_train_model = np.log1p(y_train)
    else:
        y_train_model = y_train

    model.fit(X_train, y_train_model)

    # Predictions
    predictions = model.predict(X_test)

    if log_transform:
        predictions_original = np.expm1(predictions)
        predictions_original = np.maximum(predictions_original, 0)
    else:
        predictions_original = predictions

    # Metrics on original sales scale
    mae = mean_absolute_error(y_test, predictions_original)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions_original)
    )

    original_scale_r2 = r2_score(
        y_test,
        predictions_original
    )

    # R-squared on the model's target scale
    if log_transform:
        test_target = np.log1p(y_test)
    else:
        test_target = y_test

    model_scale_r2 = r2_score(
        test_target,
        predictions
    )

    # Training adjusted R-squared
    transformed_train = model.named_steps[
        "preprocessor"
    ].transform(X_train)

    feature_count = transformed_train.shape[1]

    train_predictions = model.predict(X_train)

    train_r2 = r2_score(
        y_train_model,
        train_predictions
    )

    n = len(X_train)
    p = feature_count

    if n > p + 1:
        adjusted_r2 = 1 - (
            (1 - train_r2) * (n - 1) / (n - p - 1)
        )
    else:
        adjusted_r2 = np.nan

    print(f"\n========== {model_name} ==========")

    print("Encoded predictor columns:", feature_count)
    print("Training R-squared:", round(train_r2, 4))
    print("Training Adjusted R-squared:", round(adjusted_r2, 4))
    print("Test R-squared (model scale):", round(model_scale_r2, 4))
    print("Test R-squared (original sales scale):", round(original_scale_r2, 4))
    print("Test MAE:", round(mae, 2))
    print("Test RMSE:", round(rmse, 2))

    return {
        "model": model_name,
        "encoded_features": feature_count,
        "training_r2": train_r2,
        "training_adjusted_r2": adjusted_r2,
        "test_r2_model_scale": model_scale_r2,
        "test_r2_original_sales_scale": original_scale_r2,
        "test_mae": mae,
        "test_rmse": rmse
    }, model, predictions_original


# --------------------------------------------------
# 6. MODEL 1: ORIGINAL SALES
# --------------------------------------------------

results = []

raw_result, raw_model, raw_predictions = evaluate_model(
    "Original Sales Model",
    log_transform=False
)

results.append(raw_result)

# --------------------------------------------------
# 7. MODEL 2: LOG-TRANSFORMED SALES
# --------------------------------------------------

log_result, log_model, log_predictions = evaluate_model(
    "Log Transformed Sales Model",
    log_transform=True
)

results.append(log_result)

# --------------------------------------------------
# 8. MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    f"{RESULT_DIR}/objective3_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")

print(
    comparison.round(4).to_string(index=False)
)

# --------------------------------------------------
# 9. SAVE TEST PREDICTIONS
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "actual_monthly_sales": y_test.values,
    "original_model_prediction": raw_predictions,
    "log_model_prediction": log_predictions
})

prediction_results.to_csv(
    f"{RESULT_DIR}/objective3_test_predictions.csv",
    index=False
)

print("\n========== OUTPUT FILES ==========")

print("1. objective3_model_comparison.csv")
print("2. objective3_test_predictions.csv")

print("\n========== OBJECTIVE 3 REGRESSION COMPLETED ==========")
