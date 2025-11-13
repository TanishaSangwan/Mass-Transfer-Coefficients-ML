import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.001)
svr.fit(X_train, y_train)

y_pred_svr = svr.predict(X_test)

r2_svr = r2_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)

print("SVR Model Performance")
print("----------------------")
print("R² Score:", round(r2_svr, 4))
print("MSE:", "{:.2e}".format(mse_svr))

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_svr, color='purple', alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("SVR Model: Actual vs Predicted kL")
plt.grid(True, alpha=0.3)
plt.show()

comparison = pd.DataFrame({
    'Actual kL': y_test.values[:10],
    'Predicted kL (SVR)': y_pred_svr[:10]
})
print("\nComparison (first 10 values):\n")
print(comparison)
