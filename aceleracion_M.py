import numpy as np
import matplotlib.pyplot as plt
import distanciavstiempo as dvt
import pandas as pd
from scipy.optimize import curve_fit


trineo = 110
masa_dorada = 73
masa_plateada = 22
masa_madera = 6
agua = 148
m2 = trineo
m3 = masa_madera + trineo
m4 = masa_madera + masa_plateada + trineo
m5 = trineo + agua
m6 = agua + masa_plateada + trineo
m7 = masa_dorada +masa_plateada + masa_madera + trineo


def pasar_a_array(prueba):
    df = pd.read_csv(prueba)
    df_filtrado_t = df[df['test'] == 3]
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

# Aceleracion con M = Masa dorada y distintas m para superficie de madera.

aceleraciones = np.array([aceleracion(f'dataset/prueba{i}.csv')[0] for i in range(5, 8)])
errores_aceleracion = np.array([aceleracion(f'dataset/prueba{i}.csv')[1] for i in range(5, 8)])
ma1 = np.array([m5, m6, m7])

plt.figure()
plt.errorbar(ma1, aceleraciones, yerr=errores_aceleracion, fmt='o', color='b', capsize=5)
plt.title('Aceleración vs m con M = Masa dorada')
plt.xlabel('Masa m (g)')
plt.ylabel('Aceleración (m/s^2)')
plt.grid(True)
plt.show()

# Aceleracion con masas M = 2 masas de plata y distntas m para superficie de papel.
aceleraciones2 = np.array([aceleracion(f'dataset/prueba{i}.csv')[0] for i in range(2, 5)])
errores_aceleraciones2 = np.array([aceleracion(f'dataset/prueba{i}.csv')[1] for i in range(2, 5)])
ma2 = np.array([m2, m3, m4])

plt.figure()
plt.errorbar(ma2, aceleraciones2, yerr=errores_aceleraciones2, fmt='o', color='b', capsize=5)
plt.title('Aceleración vs m con M = 2 masas de plata')
plt.xlabel('Masa m (g)')
plt.ylabel('Aceleración (m/s^2)')
plt.grid(True)
plt.show()

# Coeficiente de Rozamiento Dinamico para distintas superficies
g = 9.81
mu_d1 = (g * ((masa_dorada/1000) / (m5/1000))) - (aceleraciones[0] / g)
mu_d2 = (g * ((masa_plateada/1000) / (m2/1000))) - (aceleraciones2[0] / g)
superficies = ['madera', 'papel']

plt.figure()
plt.bar(superficies, [mu_d1, mu_d2], color='b')
plt.title('Coeficiente de Rozamiento Dinamico para distintas superficies')
plt.ylabel('Coeficiente de Rozamiento Dinamico')
plt.xlabel('Superficie')
plt.grid(True)
plt.show()