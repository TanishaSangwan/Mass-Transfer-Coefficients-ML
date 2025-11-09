import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("kl_lab_synthetic.csv")

if "Packing_Type" in df.columns:
    df["Packing_Type"] = df["Packing_Type"].astype("category").cat.codes

X = df[['Re', 'Sc', 'D_AB_m2_s', 'D_m', 
        'Temperature_K', 'Pressure_Pa', 'Packing_Type']]
y = df['k_L_m_s']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(32,),   
                   activation='relu',
                   solver='adam',
                   learning_rate='adaptive',
                   max_iter=2000,
                   random_state=42)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MLP Neural Network Performance")
print("------------------------------")
print("R² Score:", round(r2, 4))
print("MSE:", "{:.2e}".format(mse))

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='navy', alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual kL (m/s)")
plt.ylabel("Predicted kL (m/s)")
plt.title("MLP Neural Network: Actual vs Predicted kL")
plt.grid(True, alpha=0.3)
plt.show()
