import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Polynomial Regression (Degree 2)")
print("R² Score:", round(r2, 4))
print("Mean Squared Error:", "{:.2e}".format(mse))

comp = pd.DataFrame({
    'Actual kL': y_test.values[:10],
    'Predicted kL': y_pred[:10]
})
print("\nComparison (first 10 values):\n")
print(comp)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolors='k')

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel("Actual kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Polynomial Regression (Degree 2): Actual vs Predicted kL")
plt.grid(True, alpha=0.3)
plt.show()
