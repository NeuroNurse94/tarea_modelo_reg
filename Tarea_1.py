# 1. Importamos nuestras "cajas de herramientas"
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt

# 2. Creamos los datos de nuestros 5 pacientes
# X = IMC de cada paciente. (Ponemos doble corchete porque Python opera tabla/matriz)
X_imc = np.array([[22], [24], [26], [28], [32], [80]]) 

# y = Presión Arterial Sistólica real de esos mismos pacientes
y_presion = np.array([110, 115, 130, 128, 145, 300]) # El último paciente es un "outlier" con un IMC muy alto y presión muy alta

# 3. Creamos el modelo de Regresión Lineal Normal y lo "entrenamos"
modelo_normal = LinearRegression()
modelo_normal.fit(X_imc, y_presion) # Aquí el modelo usa matemáticas para hallar la línea

# Creamos un modelo con Regularización Ridge (le ponemos una penalización alta: alpha=100)
# para ver cómo "suaviza" la línea.
modelo_ridge = Ridge(alpha=100)
modelo_ridge.fit(X_imc, y_presion)

# 4. Hacemos predicciones
# ¿Qué presión tendrían según nuestros modelos?
prediccion_normal = modelo_normal.predict(X_imc)
prediccion_ridge = modelo_ridge.predict(X_imc)

# 5. Visualizamos los resultados (dibujamos el gráfico)
plt.scatter(X_imc, y_presion, color='red', label='Pacientes Reales')
plt.plot(X_imc, prediccion_normal, color='blue', label='Línea Normal')
plt.plot(X_imc, prediccion_ridge, color='green', linestyle='dashed', label='Línea Ridge (Penalizada)')

plt.title("Predicción de Presión Arterial según IMC")
plt.xlabel("IMC")
plt.ylabel("Presión Arterial")
plt.legend()
plt.show()