
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_PATH = "data/processed/objective3_modeling_data.csv"

RESULTS_DIR = "outputs/results"
GRAPHS_DIR = "outputs/graphs"
MODEL_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "monthly_sales"

ALPHA = 10

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

print("\n========== FINAL OBJECTIVE 3 MODEL ==========")

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
# 4. NUMERICAL PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# --------------------------------------------------
# 5. CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=10,
        drop="first"
    ))
])


# --------------------------------------------------
# 6. COMBINE PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])


# --------------------------------------------------
# 7. FINAL RIDGE REGRESSION MODEL
# --------------------------------------------------

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=ALPHA))
])


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

print("\n========== MODEL TRAINING ==========")

final_model.fit(X_train, y_train)

print("Ridge Regression trained successfully.")
print("Selected alpha:", ALPHA)


# --------------------------------------------------
# 9. MAKE PREDICTIONS
# --------------------------------------------------

train_predictions = final_model.predict(X_train)
test_predictions = final_model.predict(X_test)

# Monthly sales cannot be negative.
train_predictions = np.maximum(train_predictions, 0)
test_predictions = np.maximum(test_predictions, 0)


# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

print("\n========== FINAL MODEL EVALUATION ==========")

train_mae = mean_absolute_error(
    y_train, train_predictions
)

test_mae = mean_absolute_error(
    y_test, test_predictions
)

train_rmse = np.sqrt(
    mean_squared_error(y_train, train_predictions)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

train_r2 = r2_score(
    y_train, train_predictions
)

test_r2 = r2_score(
    y_test, test_predictions
)

print("\nTraining Performance")
print("--------------------")
print("Training MAE:", round(train_mae, 2))
print("Training RMSE:", round(train_rmse, 2))
print("Training R2:", round(train_r2, 4))

print("\nTesting Performance")
print("-------------------")
print("Testing MAE:", round(test_mae, 2))
print("Testing RMSE:", round(test_rmse, 2))
print("Testing R2:", round(test_r2, 4))


# --------------------------------------------------
# 11. SAVE EVALUATION METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "Dataset": "Training",
        "MAE": train_mae,
        "RMSE": train_rmse,
        "R2": train_r2
    },
    {
        "Dataset": "Testing",
        "MAE": test_mae,
        "RMSE": test_rmse,
        "R2": test_r2
    }
])

metrics_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_final_model_metrics.csv"
    ),
    index=False
)


# --------------------------------------------------
# 12. FEATURE COEFFICIENTS
# --------------------------------------------------

print("\n========== FEATURE COEFFICIENTS ==========")

fitted_preprocessor = final_model.named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

coefficients = final_model.named_steps[
    "regressor"
].coef_

coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coefficient_df["Absolute_Coefficient"] = (
    coefficient_df["Coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

coefficient_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_final_feature_coefficients.csv"
    ),
    index=False
)

print(
    coefficient_df.head(15).to_string(index=False)
)


# --------------------------------------------------
# 13. SAVE TEST PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual_Monthly_Sales": y_test.to_numpy(),
    "Predicted_Monthly_Sales": test_predictions,
    "Prediction_Error": (
        y_test.to_numpy() - test_predictions
    )
})

prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "objective3_final_model_predictions.csv"
    ),
    index=False
)


# --------------------------------------------------
# 14. ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.5
)

max_value = max(
    float(y_test.max()),
    float(test_predictions.max())
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Actual Monthly Sales")
plt.ylabel("Predicted Monthly Sales")

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPHS_DIR,
        "objective3_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 15. SAVE FINAL TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_ridge_sales_model.joblib"
)

joblib.dump(final_model, MODEL_PATH)


# --------------------------------------------------
# 16. FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL MODEL SAVED ==========")

print("Model file:")
print(MODEL_PATH)

print("\nGenerated result files:")
print("- objective3_final_model_metrics.csv")
print("- objective3_final_feature_coefficients.csv")
print("- objective3_final_model_predictions.csv")

print("\nGenerated graph:")
print("- objective3_actual_vs_predicted.png")

print("\nObjective 3 final model completed successfully.")
