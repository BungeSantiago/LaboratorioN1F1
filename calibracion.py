import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Función lineal 
funcion_lineal = lambda x, a, b: a*x + b

# Mediciones
sensor = np.array([253, 309, 365, 719, 896, 1024, 1247, 1487, 1705]) # Eje x
mediciones = np.array([4, 5, 6, 10, 14, 18, 22, 26, 28]) # Eje y
incertezas_sensor = 1 # Incertezas del sensor
incertezas_mediciones = 0.1 # Incertezas de la regla

# Ajuste lineal
params, covariance = curve_fit(f = funcion_lineal, xdata = sensor, ydata = mediciones, sigma = incertezas_sensor, absolute_sigma = True)
pendiente, ordenada = params
incerteza_pendiente, incerteza_ordenada = np.sqrt(np.diag(covariance))
print(f'Pendiente: {pendiente} ± {incerteza_pendiente}')
print(f'Ordenada al origen: {ordenada} ± {incerteza_ordenada}')

# Grafico los datos con sus incertezas
plt.errorbar(sensor, mediciones, yerr=incertezas_mediciones, xerr=incertezas_sensor, fmt='ro', label='Datos con incertezas')

# Grafico la recta de ajuste
values = np.linspace(min(sensor), max(sensor), 1000)
y = funcion_lineal(values, pendiente, ordenada)
plt.plot(values, y, label = 'Recta')

# Etiquetas y leyenda
plt.xlabel('Unidades de distancia del sensor')
plt.ylabel('Distancia real (cm)')
plt.legend()

# Mostrar el gráfico
plt.show()

# Ejemplo de conversión
aproximacion = funcion_lineal(sensor, pendiente, ordenada)
errores = sensor**2 * incerteza_pendiente**2 + incerteza_ordenada**2
errores = errores**(1/2)
# Sacar esto
for i, error in enumerate(errores):
    errores[i] = round(error, 3)
conversion = [(round(x, 3), y) for x, y in zip(aproximacion, mediciones)]
print(f'(Aproximación, Mediciones): {conversion}')
print(f'Incertezas: {errores}')
