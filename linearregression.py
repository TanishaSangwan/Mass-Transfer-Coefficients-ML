import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLinear Regression Performance")
print("-----------------------------")
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("MSE:", "{:.2e}".format(mean_squared_error(y_test, y_pred)))

comparison = pd.DataFrame({
    'Actual k_L (m/s)': y_test.values[:10],
    'Predicted k_L (m/s)': y_pred[:10]
})

print("\nFirst 10 Predictions vs Actual:")
print(comparison)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolors='k')

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel("Actual kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Linear Regression: Actual vs Predicted kL")
plt.grid(True, alpha=0.3)
plt.show()
