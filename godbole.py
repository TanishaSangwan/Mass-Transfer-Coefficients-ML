import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("dataset.csv")

df = df.dropna(subset=[
    "k_L (m/s)",
    "Viscosity of Liquid (Pa.s)",
    "Superficial gas velocity (m/s)"
]).reset_index(drop=True)

U_G = df["Superficial gas velocity (m/s)"]
mu_L = df["Viscosity of Liquid (Pa.s)"]

df["kL_Godbole (m/s)"] = 1.12e-4 * (U_G ** -0.03) * (mu_L ** -0.5)
df["a_Godbole (m2/m3)"] = 19.2 * (U_G ** 0.47) * (mu_L ** -0.76)

df["kLa_Godbole (1/s)"] = df["kL_Godbole (m/s)"] * df["a_Godbole (m2/m3)"]

actual = df["k_L (m/s)"]
pred = df["kL_Godbole (m/s)"]

r2 = r2_score(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

print("\n📘 Godbole et al. (1984) Correlation Performance")
print("------------------------------------------------")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.3e} m/s")

plt.figure(figsize=(6,6))
plt.scatter(actual, pred, color='orange', alpha=0.7, edgecolors='k', label="Godbole et al. (1984)")
plt.plot([actual.min(), actual.max()],
         [actual.min(), actual.max()], 'r--', lw=2, label="Ideal Fit (y=x)")
plt.xlabel("Experimental kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Godbole et al. (1984) vs Experimental kL")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

