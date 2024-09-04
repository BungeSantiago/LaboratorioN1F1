import numpy as np
import matplotlib.pyplot as plt
import distanciavstiempo as dvt
import pandas as pd
from scipy.optimize import curve_fit


# Aceleracion con M = Masa dorada y distintas m para superficie de madera.
trineo = 110
masa_dorada = 73
masa_plateada = 22
masa_madera = 6
agua = 148
m1 = trineo + agua
m2 = agua + masa_plateada + trineo
m3 = masa_dorada +masa_plateada + masa_madera +trineo

def pasar_a_array(prueba):
    df = pd.read_csv(prueba)
    df_filtrado_t = df[df['test'] == 1]
    tiempo = df_filtrado_t['milisegundos'].to_numpy()
    posicion = df_filtrado_t['mediciones'].to_numpy()
    return tiempo, posicion

def aceleracion(prueba):
    tiempo, posicion = pasar_a_array(prueba)
    tiempo, posicion = dvt.correct_units(tiempo, posicion)
    errores_y = np.full(len(posicion), 0.1)

    # Definir la función cuadrática
    def modelo_cuadratico(t, a, v_0, x_0):      
        return 0.5 * a * t**2 + v_0 * t + x_0

    # Ajustar la curva
    popt, pcov = curve_fit(modelo_cuadratico, tiempo, posicion, sigma=errores_y, absolute_sigma=True)

    # Obtener los coeficientes ajustados y sus errores
    a_opt = popt[0]  # La aceleración óptima es el primer parámetro
    error_a = np.sqrt(pcov[0, 0])  # Error asociado a la aceleración

    return a_opt, error_a

aceleraciones = np.array([aceleracion(f'dataset/prueba{i}.csv')[0] for i in range(5, 8)])
errores_aceleracion = np.array([aceleracion(f'dataset/prueba{i}.csv')[1] for i in range(5, 8)])
m = np.array([m1, m2, m3])

plt.figure()
plt.errorbar(m, aceleraciones, yerr=errores_aceleracion, fmt='o', color='b', capsize=5)
plt.title('Aceleración vs m')
plt.xlabel('Masa m (g)')
plt.ylabel('Aceleración (m/s^2)')
plt.grid(True)
plt.show()
