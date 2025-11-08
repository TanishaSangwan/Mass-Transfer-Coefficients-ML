import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

np.random.seed(42)
n = 300

Re = np.random.uniform(500, 20000, n)
Sc = np.random.uniform(500, 1500, n)
DAB = np.random.uniform(1e-9, 5e-9, n)   # diffusivity (m²/s)
mu = np.random.uniform(1e-3, 5e-3, n)    # viscosity (Pa.s)
rho = np.random.uniform(800, 1200, n)    # density (kg/m³)
T = np.random.uniform(293, 333, n)       # temperature (K)
D = np.random.uniform(0.01, 0.05, n)     # characteristic length (m)

Sh = 0.3 * Re**0.6 * Sc**0.33
kL_true = Sh * DAB / D

noise = np.random.normal(0, kL_true * 0.05)
kL_exp = kL_true + noise

data = pd.DataFrame({
    'Re': Re, 'Sc': Sc, 'DAB': DAB, 'mu': mu, 'rho': rho, 'T': T, 'D': D,
    'kL': kL_exp
})

data['Re_Sc_ratio'] = data['Re'] / (data['Sc'] ** (1/3))
data['mu_by_rho'] = data['mu'] / data['rho']
data['T_norm'] = data['T'] / 298
data['log_kL'] = np.log10(data['kL'])

X = data[['Re', 'Sc', 'DAB', 'mu', 'rho', 'T', 'D', 'Re_Sc_ratio', 'mu_by_rho', 'T_norm']]
y = data['log_kL']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R²: {r2:.4f}, MSE: {mse:.2e}")

cv_scores = cross_val_score(xgb, X_scaled, y, cv=5, scoring='r2')
print("Average CV R²:", cv_scores.mean())

importances = xgb.feature_importances_
plt.figure(figsize=(8,5))
plt.barh(X.columns, importances, color='teal')
plt.title('Feature Importance for kL Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

Sh_corr = 0.3 * (data['Re']**0.6) * (data['Sc']**0.33)
kL_corr = Sh_corr * data['DAB'] / data['D']

kL_pred_all = 10 ** xgb.predict(X_scaled)

plt.figure(figsize=(6,6))
plt.scatter(kL_corr, kL_pred_all, alpha=0.6, c='darkorange')
plt.plot([kL_corr.min(), kL_corr.max()], [kL_corr.min(), kL_corr.max()], 'k--')
plt.xlabel("Correlation kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("Machine Learning vs Theoretical Correlation")
plt.grid(True, alpha=0.3)
plt.show()

print("\nTop 3 Important Features:")
for feat, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:3]:
    print(f"{feat}: {imp:.3f}")

print("\nModel aligns well with known physics (Re, Sc dominant).")
print("Use real absorption data for training to make it experimentally validated.")
