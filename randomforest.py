import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv("kl_lab_synthetic.csv")

if "Packing_Type" in df.columns:
    df["Packing_Type"] = df["Packing_Type"].astype("category").cat.codes

X = df[['Re', 'Sc', 'D_AB_m2_s', 'D_m', 
        'Temperature_K', 'Pressure_Pa', 'Packing_Type']]
y = df['k_L_m_s']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=300,   # Number of trees
    max_depth=None,    # Let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Random Forest Performance")
print("-------------------------")
print("R² Score:", round(r2, 4))
print("Mean Squared Error:", "{:.2e}".format(mse))

import matplotlib.pyplot as plt

importance = rf.feature_importances_
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
