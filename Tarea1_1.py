import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

# 1. Nuestros 7 pacientes equilibrados (7 en X, 7 en y)
X_imc = np.array([[22], [24], [26], [28], [30], [32], [35]]) 
y_presion = np.array([110, 115, 130, 128, 135, 145, 140])

# 2. Entrenamos los 3 Modelos
# Modelo Normal (Sin penalización)
modelo_normal = LinearRegression()
modelo_normal.fit(X_imc, y_presion)

# Modelo Ridge (Suaviza la pendiente)
modelo_ridge = Ridge(alpha=100) # Probemos un alpha de 100
modelo_ridge.fit(X_imc, y_presion)

# Modelo Lasso (Puede hacer la pendiente exactamente 0)
modelo_lasso = Lasso(alpha=50) # Probemos un alpha de 50
modelo_lasso.fit(X_imc, y_presion)

# 3. Hacemos las predicciones
prediccion_normal = modelo_normal.predict(X_imc)
prediccion_ridge = modelo_ridge.predict(X_imc)
prediccion_lasso = modelo_lasso.predict(X_imc)

# 4. Imprimimos los valores exactos de las pendientes (pesos) en la consola
print(f"Pendiente Normal: {modelo_normal.coef_[0]:.2f}")
print(f"Pendiente Ridge:  {modelo_ridge.coef_[0]:.2f}")
print(f"Pendiente Lasso:  {modelo_lasso.coef_[0]:.2f}")

# 5. Visualizamos
plt.figure(figsize=(8, 6))
plt.scatter(X_imc, y_presion, color='black', label='Pacientes Reales', zorder=5)

plt.plot(X_imc, prediccion_normal, color='blue', label='Normal (OLS)')
plt.plot(X_imc, prediccion_ridge, color='green', linestyle='dashed', label='Ridge (L2)')
plt.plot(X_imc, prediccion_lasso, color='red', linestyle='dotted', linewidth=3, label='Lasso (L1)')

plt.title("Comparación: Normal vs Ridge vs Lasso (1 Variable)")
plt.xlabel("IMC")
plt.ylabel("Presión Arterial")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()