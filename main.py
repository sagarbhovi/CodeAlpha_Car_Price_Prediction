
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RandomizedSearchCV
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)


# ==========================================================
# 1. PROJECT SETUP
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "car data.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
VISUAL_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("             CAR PRICE PREDICTION")
print("=" * 65)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nORIGINAL DATASET")
print("-" * 65)
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================================
# 3. DATA CLEANING
# ==========================================================

df.columns = df.columns.str.strip()

# Remove duplicate records
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()

# Remove rows with missing values
df = df.dropna().copy()

# Clean categorical columns
categorical_columns = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

# Convert numerical columns
numerical_columns = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Selling_Price"
]

for column in numerical_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna().copy()

print("\nDATA CLEANING")
print("-" * 65)
print("Duplicate rows removed:", duplicate_count)
print("Cleaned dataset shape:", df.shape)
print("Missing values after cleaning:", df.isnull().sum().sum())

# Save cleaned dataset
df.to_csv(OUTPUT_DIR / "cleaned_car_data.csv", index=False)


# ==========================================================
# 4. EXPLORATORY DATA ANALYSIS
# ==========================================================

print("\nEXPLORATORY DATA ANALYSIS")
print("-" * 65)

print("\nStatistical summary:")
print(df.describe())

print("\nFuel type distribution:")
print(df["Fuel_Type"].value_counts())

print("\nTransmission distribution:")
print(df["Transmission"].value_counts())

print("\nSelling type distribution:")
print(df["Selling_type"].value_counts())

print("\nAverage selling price:", round(df["Selling_Price"].mean(), 2))
print("Minimum selling price:", round(df["Selling_Price"].min(), 2))
print("Maximum selling price:", round(df["Selling_Price"].max(), 2))


# ==========================================================
# 5. DATA VISUALIZATION
# ==========================================================

# Chart 1: Selling price distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "selling_price_distribution.png")
plt.close()


# Chart 2: Present price vs selling price
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Present_Price",
    y="Selling_Price",
    hue="Fuel_Type"
)
plt.title("Present Price vs Selling Price")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "present_vs_selling_price.png")
plt.close()


# Chart 3: Selling price by fuel type
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price")
plt.title("Selling Price by Fuel Type")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "selling_price_by_fuel.png")
plt.close()


# Chart 4: Selling price by transmission
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Transmission", y="Selling_Price")
plt.title("Selling Price by Transmission")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "selling_price_by_transmission.png")
plt.close()


# Chart 5: Selling price by vehicle year
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df,
    x="Year",
    y="Selling_Price",
    hue="Transmission"
)
plt.title("Vehicle Year vs Selling Price")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "year_vs_selling_price.png")
plt.close()


# Chart 6: Numerical correlation heatmap
plt.figure(figsize=(9, 6))
correlation = df[numerical_columns].corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "correlation_heatmap.png")
plt.close()


# ==========================================================
# 6. FEATURE SELECTION
# ==========================================================

# Car_Name is excluded from this baseline model.
features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner"
]

target = "Selling_Price"

X = df[features]
y = df[target]

numeric_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

print("\nFEATURES:", features)
print("TARGET:", target)


# ==========================================================
# 7. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("-" * 65)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================================
# 8. PREPROCESSING PIPELINE
# ==========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)



# ==========================================================
# 9. MODEL TRAINING AND HYPERPARAMETER TUNING
# ==========================================================

models = {
    "Linear Regression": {
        "estimator": LinearRegression(),
        "params": {
            "model__fit_intercept": [True, False]
        },
        "iterations": 2
    },

    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=42),
        "params": {
            "model__max_depth": [None, 3, 5, 7, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "iterations": 12
    },

    "Random Forest": {
        "estimator": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 8, 12],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"]
        },
        "iterations": 12
    },

    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "params": {
            "model__n_estimators": [100, 150, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.8, 1.0]
        },
        "iterations": 12
    }
}


# Repeated 5-fold cross-validation (3 repetitions)
repeated_cv = RepeatedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

results = []
trained_models = {}

print("\nMODEL TRAINING AND HYPERPARAMETER TUNING")
print("-" * 65)

for name, configuration in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", configuration["estimator"])
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=configuration["params"],
        n_iter=configuration["iterations"],
        scoring="neg_root_mean_squared_error",
        cv=repeated_cv,
        random_state=42,
        n_jobs=-1
    )

    # Tune using training data only
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_

    # Evaluate on untouched test data
    predictions = best_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    cv_rmse = -search.best_score_

    results.append({
        "Model": name,
        "CV_RMSE": cv_rmse,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2
    })

    trained_models[name] = best_pipeline

    print(f"\n{name}")
    print(f"Repeated CV RMSE: {cv_rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    print("Best Parameters:", search.best_params_)


# ==========================================================
# 10. MODEL COMPARISON AND SELECTION
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_RMSE",
    ascending=True
).reset_index(drop=True)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nMODEL COMPARISON SUMMARY")
print("-" * 65)
print(results_df.to_string(index=False))

print("\nBEST MODEL:", best_model_name)
print("Selected using the lowest repeated CV RMSE.")


# ==========================================================
# 11. FINAL MODEL EVALUATION
# ==========================================================

best_predictions = best_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    best_predictions
)

final_rmse = root_mean_squared_error(
    y_test,
    best_predictions
)

final_r2 = r2_score(
    y_test,
    best_predictions
)

print("\nFINAL MODEL PERFORMANCE")
print("-" * 65)
print("Best Model:", best_model_name)
print(f"MAE: {final_mae:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"R2 Score: {final_r2:.4f}")


# ==========================================================
# 12. ACTUAL VS PREDICTED VISUALIZATION
# ==========================================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_predictions, alpha=0.7)

min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "actual_vs_predicted.png")
plt.close()


# ==========================================================
# 13. RESIDUAL ANALYSIS
# ==========================================================

residuals = y_test - best_predictions

plt.figure(figsize=(9, 5))
sns.scatterplot(
    x=best_predictions,
    y=residuals
)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "residual_analysis.png")
plt.close()


# ==========================================================
# 14. MODEL FEATURE IMPORTANCE
# ==========================================================

best_estimator = best_model.named_steps["model"]

if hasattr(best_estimator, "feature_importances_"):

    feature_names = (
        best_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_estimator.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance.head(12),
        x="Importance",
        y="Feature"
    )
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "feature_importance.png")
    plt.close()


# ==========================================================
# 15. CUSTOM CAR PRICE PREDICTION
# ==========================================================

custom_car = pd.DataFrame([{
    "Year": 2015,
    "Present_Price": 5.5,
    "Driven_kms": 30000,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}])

custom_prediction = best_model.predict(custom_car)[0]

print("\nCUSTOM CAR PRICE PREDICTION")
print("-" * 65)
print(custom_car.to_string(index=False))
print(f"\nPredicted Selling Price: {custom_prediction:.2f} lakh")


# ==========================================================
# 16. SAVE TRAINED MODEL
# ==========================================================

model_path = OUTPUT_DIR / "best_car_price_model.pkl"

joblib.dump(best_model, model_path)

print("\nModel saved at:", model_path)


# ==========================================================
# 17. SAVE KEY INSIGHTS REPORT
# ==========================================================

insights = f"""
CAR PRICE PREDICTION - KEY INSIGHTS

Dataset:
- Total records after cleaning: {len(df)}
- Total features used: {len(features)}
- Target variable: Selling_Price

Dataset Statistics:
- Average selling price: {df['Selling_Price'].mean():.2f} lakh
- Minimum selling price: {df['Selling_Price'].min():.2f} lakh
- Maximum selling price: {df['Selling_Price'].max():.2f} lakh

Model Comparison:
{results_df.to_string(index=False)}

Best Model:
{best_model_name}

Final Model Evaluation:
- MAE: {final_mae:.4f}
- RMSE: {final_rmse:.4f}
- R2 Score: {final_r2:.4f}

Custom Prediction:
- Predicted selling price: {custom_prediction:.2f} lakh

Note:
Predictions are estimates based on the dataset and model.
They are not guaranteed market prices.
"""

with open(
    REPORT_DIR / "key_insights.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(insights)


# ==========================================================
# 18. PROJECT COMPLETION
# ==========================================================

print("\n" + "=" * 65)
print("          PROJECT EXECUTION COMPLETED")
print("=" * 65)

print("Cleaned records:", len(df))
print("Models compared:", len(models))
print("Best model:", best_model_name)
print("Visualizations saved:", len(list(VISUAL_DIR.glob("*.png"))))
print("Reports saved:", len(list(REPORT_DIR.glob("*"))))
print("Trained model saved:", model_path)
print("\nCheck the outputs folder for all generated files.")
