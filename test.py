import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Función lineal 
funcion_lineal = lambda x, a, b: a*x + b

# Mediciones
sensor = np.array([0.07, 0.14, 0.15, 0.33, 0.48, 0.52, 0.68, 0.78, 0.89, 0.99]) # Eje x
mediciones = np.array([0.06, 0.30, 0.40, 0.82, 0.92, 1.16, 1.18, 1.58, 1.84, 2.00]) # Eje y
incertezas_sensor = 0.05 # Incertezas del sensor
incertezas_mediciones = 0.15 # Incertezas de la regla

# Ajuste lineal
params, covariance = curve_fit(f = funcion_lineal, xdata = sensor, ydata = mediciones, sigma = incertezas_sensor, absolute_sigma = True)
pendiente, ordenada = params
incerteza_pendiente, incerteza_ordenada = np.sqrt(np.diag(covariance))
print(f'Pendiente: {pendiente} ± {incerteza_pendiente}')
print(f'Ordenada al origen: {ordenada} ± {incerteza_ordenada}')

# Ejemplo de conversión
aproximacion = funcion_lineal(sensor, pendiente, ordenada)
errores = lambda x : (pendiente**2 * incertezas_sensor**2) + (x**2 * incerteza_pendiente**2) + (incerteza_ordenada**2) + (2*x*covariance[0, 1]) # -> Esto es el error al cuadrado
errores = np.sqrt(errores(sensor))
lista_errores = [(round(error, 3)) for error in errores]

conversion = [(round(x, 3), y) for x, y in zip(aproximacion, mediciones)]
print(f'(Aproximación, Mediciones): {conversion}')
print(f'Incertezas: {lista_errores}')
# PREGUNTAR SI DEBERIAMOS GRAFICAR ESTOS ERRORES

# Grafico los datos con sus incertezas
plt.errorbar(sensor, mediciones, yerr=incertezas_mediciones, xerr=incertezas_sensor, fmt='ro', label='Datos con incertezas')

# Grafico la recta de ajuste
values = np.linspace(min(sensor), max(sensor), 1000)
y = funcion_lineal(values, pendiente, ordenada)
plt.plot(values, y, label = 'Recta')

# Etiquetas y leyenda
plt.xlabel('Unidades de distancia del sensor [U.A.]')
plt.ylabel('Distancia real [cm]')
plt.legend()

# Mostrar el gráfico
plt.show()