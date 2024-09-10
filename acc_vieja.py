import numpy as np
import matplotlib.pyplot as plt
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

def masas_dict(file_path:str) -> dict:
    '''
    Pasa los datos del csv a un diccionario con la siguiente estructura:
    dict = {'trineo': 110, 'dorada': 73, ...}
    '''
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            masa, peso = line.strip().split(',')
            data[masa] = float(peso)
    return data

def csv_to_dict(file_path:str) -> dict:
    '''
    Pasa los datos del csv a un diccionario con la siguiente estructura:
    dict = {'test 1': {'milisegundos': [], 'mediciones': []}, 'test 2': {'milisegundos': [], 'mediciones': []} ...
    '''
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')
        for i in range(1, 4):
            data[f'test {i}'] = {header[1]: [], header[2]: []}
        for line in lines[1:]:
            test, time, measurement = line.strip().split(',')
            data[f'test {test}'][header[1]].append(float(time))
            data[f'test {test}'][header[2]].append(float(measurement))
        for key in data:
            for sub_key in data[key]:
                data[key][sub_key] = np.array(data[key][sub_key])
    return data

def pasar_a_array(prueba):
    data = csv_to_dict(prueba)
    tiempo, posicion = data['test 3']['milisegundos'], data['test 3']['mediciones']
    return tiempo, posicion

def correct_units(time:np.array, distance:np.array) -> tuple:
    '''
    Convierte las unidades de tiempo a segundos y la distancia a centímetros.
    '''
    a = 0.01736595797123086
    b = -0.6682770640993427
    
    time = time / 1000
    distance = a * distance + b
    return time, distance

def aceleracion(prueba):
    tiempo, posicion = pasar_a_array(prueba)
    tiempo, posicion = correct_units(tiempo, posicion)
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

pesos = masas_dict('dataset/datos.txt')

m2 = pesos['trineo']
m3 = pesos['madera'] + pesos['trineo']
m4 = pesos['madera'] + pesos['plateada'] + pesos['trineo']
m5 = pesos['trineo'] + pesos['agua']
m6 = pesos['agua'] + pesos['plateada'] + pesos['trineo']
m7 = pesos['dorada'] + pesos['plateada'] + pesos['madera'] + pesos['trineo']

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
mu_d1 = aceleraciones[0] / g
mu_d2 = aceleraciones2[0] / g
superficies = ['madera', 'papel']
plt.figure()
plt.bar(superficies, [mu_d1, mu_d2], color='b')
plt.title('Coeficiente de Rozamiento Dinamico para distintas superficies')
plt.ylabel('Coeficiente de Rozamiento Dinamico')
plt.xlabel('Superficie')
plt.grid(True)
plt.show()