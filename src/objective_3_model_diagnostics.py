
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA_PATH = "data/processed/objective3_modeling_data.csv"

RESULTS_DIR = "outputs/results"
GRAPHS_DIR = "outputs/graphs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

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

print("\n========== OBJECTIVE 3: MODEL DIAGNOSTICS ==========")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = pd.to_numeric(df[TARGET], errors="coerce")

valid_rows = y.notna() & (y >= 0)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("\nTotal products:", len(df))
print("Products used:", len(X))
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
# 5. ORIGINAL LINEAR REGRESSION MODEL
# --------------------------------------------------

original_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

original_model.fit(X_train, y_train)

original_predictions = original_model.predict(X_test)

original_predictions = np.maximum(original_predictions, 0)


# --------------------------------------------------
# 6. LOG TRANSFORMED REGRESSION MODEL
# --------------------------------------------------

log_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

y_train_log = np.log1p(y_train)

log_model.fit(X_train, y_train_log)

log_predictions = log_model.predict(X_test)

# Duan's Smearing Estimator
train_log_predictions = log_model.predict(X_train)

log_residuals = y_train_log - train_log_predictions

smearing_factor = np.mean(np.exp(log_residuals))

corrected_log_predictions = (
    np.exp(log_predictions) * smearing_factor
) - 1

corrected_log_predictions = np.maximum(
    corrected_log_predictions, 0
)

print("\n========== LOG MODEL CORRECTION ==========")
print("Smearing factor:", round(smearing_factor, 4))


# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

models = {
    "Original Linear Regression": original_predictions,
    "Log Regression (Smearing Corrected)": corrected_log_predictions
}

comparison = []

for model_name, predictions in models.items():

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    comparison.append({
        "Model": model_name,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2_Original_Scale": r2
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    os.path.join(RESULTS_DIR, "objective3_corrected_model_comparison.csv"),
    index=False
)

print("\n========== CORRECTED MODEL COMPARISON ==========")
print(comparison_df.to_string(index=False))


# --------------------------------------------------
# 8. SELECT MODEL FOR DIAGNOSTICS
# --------------------------------------------------

# Use the original linear regression model for
# residual diagnostics, consistent with the baseline.

selected_model = original_model

train_predictions = selected_model.predict(X_train)
test_predictions = selected_model.predict(X_test)

test_residuals = y_test.to_numpy() - test_predictions


# --------------------------------------------------
# 9. VIF: MULTICOLLINEARITY CHECK
# --------------------------------------------------

print("\n========== VIF: MULTICOLLINEARITY ==========")

# VIF is calculated for numeric predictors only.
# Median imputation is fitted using training data.

vif_imputer = SimpleImputer(strategy="median")

X_numeric_train = pd.DataFrame(
    vif_imputer.fit_transform(X_train[NUMERIC_FEATURES]),
    columns=NUMERIC_FEATURES,
    index=X_train.index
)

vif_data = sm.add_constant(X_numeric_train)

vif_results = []

for i, column in enumerate(vif_data.columns):

    if column == "const":
        continue

    try:
        vif_value = variance_inflation_factor(
            vif_data.values, i
        )
    except Exception:
        vif_value = np.nan

    vif_results.append({
        "Feature": column,
        "VIF": vif_value
    })

vif_df = pd.DataFrame(vif_results)

vif_df.to_csv(
    os.path.join(RESULTS_DIR, "objective3_vif_results.csv"),
    index=False
)

print(vif_df.to_string(index=False))

print("\nVIF Interpretation:")
print("VIF close to 1: Low multicollinearity")
print("VIF between 1 and 5: Usually acceptable")
print("VIF above 5: Possible multicollinearity")
print("VIF above 10: High multicollinearity")


# --------------------------------------------------
# 10. BREUSCH-PAGAN TEST
# --------------------------------------------------

print("\n========== BREUSCH-PAGAN TEST ==========")

# Use the encoded test predictors and test residuals.

X_test_encoded = selected_model.named_steps[
    "preprocessor"
].transform(X_test)

if hasattr(X_test_encoded, "toarray"):
    X_test_encoded = X_test_encoded.toarray()

X_test_encoded = sm.add_constant(
    X_test_encoded,
    has_constant="add"
)

bp_result = het_breuschpagan(
    test_residuals,
    X_test_encoded
)

bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = bp_result

print("LM Statistic:", round(bp_lm_stat, 4))
print("LM p-value:", bp_lm_pvalue)
print("F Statistic:", round(bp_f_stat, 4))
print("F p-value:", bp_f_pvalue)

if bp_lm_pvalue < 0.05:
    print("Result: Evidence of heteroscedasticity.")
else:
    print("Result: No statistically significant evidence of heteroscedasticity.")

bp_df = pd.DataFrame([{
    "LM_Statistic": bp_lm_stat,
    "LM_P_Value": bp_lm_pvalue,
    "F_Statistic": bp_f_stat,
    "F_P_Value": bp_f_pvalue
}])

bp_df.to_csv(
    os.path.join(RESULTS_DIR, "objective3_breusch_pagan_test.csv"),
    index=False
)


# --------------------------------------------------
# 11. RESIDUAL VS FITTED PLOT
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=test_predictions,
    y=test_residuals,
    alpha=0.5
)

plt.axhline(0, linestyle="--", color="red")

plt.title("Residuals vs Fitted Values")
plt.xlabel("Predicted Monthly Sales")
plt.ylabel("Residuals")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS_DIR, "objective3_residual_vs_fitted.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 12. RESIDUAL DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    test_residuals,
    kde=True
)

plt.title("Distribution of Regression Residuals")
plt.xlabel("Residuals")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS_DIR, "objective3_residual_distribution.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 13. Q-Q PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 6))

stats.probplot(
    test_residuals,
    dist="norm",
    plot=plt
)

plt.title("Q-Q Plot of Regression Residuals")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS_DIR, "objective3_qq_plot.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. SAVE TEST PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual_Monthly_Sales": y_test.to_numpy(),
    "Original_Model_Predictions": original_predictions,
    "Smearing_Corrected_Log_Predictions": corrected_log_predictions,
    "Original_Model_Residuals": test_residuals
})

prediction_df.to_csv(
    os.path.join(RESULTS_DIR, "objective3_diagnostic_predictions.csv"),
    index=False
)


# --------------------------------------------------
# 15. FINAL SUMMARY
# --------------------------------------------------

print("\n========== DIAGNOSTICS COMPLETED ==========")

print("\nGenerated result files:")
print("- objective3_corrected_model_comparison.csv")
print("- objective3_vif_results.csv")
print("- objective3_breusch_pagan_test.csv")
print("- objective3_diagnostic_predictions.csv")

print("\nGenerated graphs:")
print("- objective3_residual_vs_fitted.png")
print("- objective3_residual_distribution.png")
print("- objective3_qq_plot.png")

print("\nObjective 3 diagnostics completed successfully.")
