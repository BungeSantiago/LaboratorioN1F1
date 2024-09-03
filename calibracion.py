import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Preguntar lo de las incertezas truchas y lo de la ordenada al origen

# Función lineal 
funcion_lineal = lambda x, a, b: a*x + b

# Mediciones
sensor = np.array([253, 309, 365, 719, 896, 1024, 1247, 1487, 1705]) # Eje x
mediciones = np.array([4, 5, 6, 10, 14, 18, 22, 26, 28]) # Eje y
incertezas = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60]) # Incertezas del sensor truchas para consultar
# incertezas = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) # Incertezas del sensor

# Ajuste lineal
params, covariance = curve_fit(funcion_lineal, sensor, mediciones, sigma=incertezas, absolute_sigma=True)
pendiente, ordenada = params
incerteza_pendiente, incerteza_ordenada = np.sqrt(np.diag(covariance))
print(f'Pendiente: {pendiente} ± {incerteza_pendiente}')
print(f'Ordenada al origen: {ordenada} ± {incerteza_ordenada}')

# Grafico los datos con sus incertezas
plt.errorbar(sensor, mediciones, xerr=incertezas, fmt='ro', label='Datos con incertezas')

# Grafico la recta de ajuste
values = np.linspace(min(sensor), max(sensor), 1000)
y = funcion_lineal(values, pendiente, ordenada)
plt.plot(values, y, label = 'Recta')

# Etiquetas y leyenda
plt.xlabel('Unidades de distancia del sensor')
plt.ylabel('Distancia real (cm)')
plt.title('Gráfico de Calibración')
plt.legend()

# Mostrar el gráfico
plt.show()

# Ejemplo de conversión
aproximacion = funcion_lineal(sensor, pendiente, ordenada)
conversion = [(x, y) for x, y in zip(aproximacion, mediciones)]
print(conversion)
