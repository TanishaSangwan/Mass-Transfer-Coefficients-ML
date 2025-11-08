import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("kl_lab_synthetic.csv")

if "Packing_Type" in df.columns:
    df["Packing_Type"] = df["Packing_Type"].astype("category").cat.codes

X = df[['Re', 'Sc', 'D_AB_m2_s', 'D_m', 'Temperature_K', 'Pressure_Pa']]
y = df['k_L_m_s']

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
