import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

df = pd.read_csv("bubble_column_merged_kL_dataset.csv")
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

y = df['k_L (m/s)'].values.reshape(-1,1)

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(16, activation='relu'),

    layers.Dense(1)   
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse')

history = model.fit(X_train, y_train, 
                    validation_split=0.2,
                    epochs=400,
                    batch_size=32,
                    verbose=0)   

y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
y_test_real = y_scaler.inverse_transform(y_test).ravel()

print("\nAdvanced MLP Performance")
print("------------------------")
print("R² Score:", round(r2_score(y_test_real, y_pred), 4))
print("MSE:", "{:.2e}".format(mean_squared_error(y_test_real, y_pred)))

plt.figure(figsize=(6,6))
plt.scatter(y_test_real, y_pred, color='darkblue', alpha=0.7, edgecolors='k')
plt.plot([min(y_test_real), max(y_test_real)],
         [min(y_test_real), max(y_test_real)], 'r--', linewidth=2)
plt.xlabel("Actual kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Advanced MLP Neural Network: Actual vs Predicted kL")
plt.grid(True, alpha=0.3)
plt.show()
