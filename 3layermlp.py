import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

df = df.dropna(subset=["k_L (m/s)"]).reset_index(drop=True)

X = df[[
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
    'Interfacial Area (m2/m3)'
]]

y = df['k_L (m/s)'].values.reshape(-1,1)   # reshape for scaling

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),  # multiple layers
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=8000,
    random_state=42
)

mlp.fit(X_train, y_train.ravel())

y_pred_scaled = mlp.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
y_test_real = y_scaler.inverse_transform(y_test).ravel()

print("Multi-Layer Neural Network Performance")
print("--------------------------------------")
print("R² Score:", round(r2_score(y_test_real, y_pred), 4))
print("MSE:", "{:.2e}".format(mean_squared_error(y_test_real, y_pred)))

plt.figure(figsize=(6,6))
plt.scatter(y_test_real, y_pred, color='teal', alpha=0.7, edgecolors='k')
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()], 'r--', linewidth=2)
plt.xlabel("Actual kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Multi-Layer Neural Network: Actual vs Predicted kL")
plt.grid(True, alpha=0.3)
plt.show()
