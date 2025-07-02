import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ---------------------------
# Load Cleaned Data
# ---------------------------
df = pd.read_csv("cleaned/cleanedcardata_imputed.csv")

# Drop raw torque/bhp text columns (already split earlier)
df.drop(columns=['Power', 'Torque'], inplace=True)

# ---------------------------
# Features & Target
# ---------------------------
target = 'Price'

# Keep Model as a feature now
num_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target]).columns.tolist()
cat_features = df.select_dtypes(include='object').columns.tolist()

# Confirm 'Model' is included
assert 'Model' in cat_features, "❌ 'Model' column is missing from categorical features!"

X = df[num_features + cat_features]
y = df[target]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42)

# ---------------------------
# Preprocessing Pipeline
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ],
    remainder='passthrough'
)

# ---------------------------
# XGBoost Model + Pipeline
# ---------------------------
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb)
])

# ---------------------------
# Model Training
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)

print("🔍 Model Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ---------------------------
# Cross-Validation
# ---------------------------
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\n🔁 Cross-Validation R² Scores:", cv_scores)
print("Mean R²:", cv_scores.mean())

# ---------------------------
# Feature Importance Plot
# ---------------------------
reg = model.named_steps['regressor']
encoder = model.named_steps['preprocessor'].named_transformers_['cat']
encoded_cols = encoder.get_feature_names_out(cat_features)
all_feature_names = list(encoded_cols) + list(num_features)

importances = reg.feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ---------------------------
# Save the Model
# ---------------------------
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/car_price_model_xgb.pkl")
print("\n✅ Model saved as: saved_models/car_price_model_xgb.pkl")
