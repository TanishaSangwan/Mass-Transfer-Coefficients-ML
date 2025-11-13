import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("dataset.csv")

df = df.dropna(subset=[
    "k_L (m/s)",
    "Density of Liquid (kg/m3)",
    "Surface Tension of Liquid (N/m)",
    "Interfacial Area (m2/m3)"
]).reset_index(drop=True)

g = 9.81              
D_L = 2.0e-9          
epsilon_G = 0.25      

df["d_vs (m)"] = 6 * epsilon_G / df["Interfacial Area (m2/m3)"]

df["d_vs (m)"] = df["d_vs (m)"].clip(lower=1e-4, upper=0.02)

df["kL_AkitaYoshida (m/s)"] = (
    0.5
    * (g ** 0.625)
    * (df["Density of Liquid (kg/m3)"] ** 0.375)
    * (df["Surface Tension of Liquid (N/m)"] ** -0.375)
    * (D_L ** 0.5)
    * (df["d_vs (m)"] ** 0.5)
)

actual = df["k_L (m/s)"]
pred = df["kL_AkitaYoshida (m/s)"]

r2 = r2_score(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

print("\n📘 Akita & Yoshida (1974) Correlation Performance")
print("--------------------------------------------------")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.3e} m/s")

plt.figure(figsize=(6,6))
plt.scatter(actual, pred, color='green', alpha=0.7, edgecolors='k', label="Akita & Yoshida (1974)")
plt.plot([actual.min(), actual.max()],
         [actual.min(), actual.max()], 'r--', lw=2, label="Ideal Fit (y=x)")
plt.xlabel("Experimental kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Akita & Yoshida (1974) vs Experimental kL")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
