import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv("dataset.csv")

df = df.dropna(subset=["k_L (m/s)"]).reset_index(drop=True)

features = [
    'Column diameter(m)',
    'Liquid height (m)',
    'Superficial gas velocity (m/s)',
    'Density of Liquid (kg/m3)',
    'Viscosity of Liquid (Pa.s)',
    'Density of Gas (kg/m3)',
    'Viscosity of Gas (Pa.s)',
    'Surface Tension of Liquid (N/m)',
    'Temperature (K)',
    'Pressure (kPa)',
    '%FA',
    'Interfacial Area (m2/m3)',
]
X = df[features]
y = df['k_L (m/s)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

xgb_model = XGBRegressor(
    n_estimators=400,       # number of boosting rounds
    learning_rate=0.05,     # step size shrinkage
    max_depth=6,            # complexity of trees
    subsample=0.8,          # row sampling
    colsample_bytree=0.8,   # feature sampling
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nXGBoost Performance")
print("-------------------")
print("R² Score:", round(r2, 4))
print("Mean Squared Error:", "{:.2e}".format(mse))

import matplotlib.pyplot as plt

importance = xgb_model.feature_importances_
plt.figure(figsize=(6,4))
plt.barh(X.columns, importance, color="teal")
plt.xlabel("Feature Importance")
plt.title("Which variables influence kL the most?")
plt.grid(alpha=0.3)
plt.show()

comparison = pd.DataFrame({
    'Actual kL': y_test.values[:10],
    'Predicted kL': y_pred[:10]
})
print("\nComparison (first 10 values):\n")
print(comparison)

