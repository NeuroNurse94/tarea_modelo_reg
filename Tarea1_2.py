import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures

# 1. Datos inventados (Edad vs Nivel de Hormona que hace una curva)
X_edad = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
y_hormona = np.array([15, 45, 60, 65, 60, 40, 20, 10]) # Sube y luego baja

# 2. EL TRUCO: Transformamos nuestra X en un polinomio de GRADO 8
# Esto creará columnas para Edad, Edad^2, Edad^3... hasta Edad^8
transformador = PolynomialFeatures(degree=8, include_bias=False)
X_polinomio = transformador.fit_transform(X_edad)

# 3. Entrenamos los modelos con los datos transformados
# OLS Normal (Se va a volver loco intentando tocar todos los puntos)
modelo_normal = LinearRegression()
modelo_normal.fit(X_polinomio, y_hormona)

# Lasso (Va a apagar los grados altos y dejará una curva suave)
# Usamos un alpha altísimo y aumentamos las iteraciones porque es un problema más duro
modelo_lasso = Lasso(alpha=1000.0, max_iter=100000) 
modelo_lasso.fit(X_polinomio, y_hormona)

# 4. Generamos puntos suaves para dibujar la línea continua
X_suave = np.linspace(10, 80, 100).reshape(-1, 1)
X_suave_polinomio = transformador.transform(X_suave)

pred_normal = modelo_normal.predict(X_suave_polinomio)
pred_lasso = modelo_lasso.predict(X_suave_polinomio)

# 5. Visualizar
plt.figure(figsize=(10, 6))
plt.scatter(X_edad, y_hormona, color='black', label='Datos Reales', zorder=5, s=60)
plt.plot(X_suave, pred_normal, color='blue', label='OLS (Polinomio Grado 8 - Sobreajustado)')
plt.plot(X_suave, pred_lasso, color='red', linewidth=3, label='Lasso (Polinomio Grado 8 - Suavizado)')

# Acercamos el gráfico para verlo bien (OLS se dispara a valores locos)
plt.ylim(0, 100) 
plt.title("Regresión Polinomial: OLS vs Lasso")
plt.xlabel("Edad")
plt.ylabel("Nivel de Hormona")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Veamos la magia de Lasso imprimiendo los coeficientes
print("Pesos del modelo Lasso (observa cuántos son exactamente CERO):")
print(np.round(modelo_lasso.coef_, 2))