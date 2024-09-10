import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Armar un diccionario con todos los pesos de todo

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

def correct_units(time:np.array, distance:np.array) -> tuple:
    '''
    Convierte las unidades de tiempo a segundos y la distancia a centímetros.
    '''
    a = 0.01736595797123086
    b = -0.6682770640993427
    
    time = time / 1000
    distance = a * distance + b

    return time, distance

def promedio_aceleracion(prueba):
    '''
    Calcula la aceleración de una prueba promediando los valores de los 3 tests.
    '''
    data = csv_to_dict(prueba)
    modelo_cuadratico = lambda t, a, v_0, x_0: a * t**2 + v_0 * t +  x_0
    aceleraciones = []
    errores_aceleracion = []

    for test in data:
        tiempo, posicion = data[test]['milisegundos'], data[test]['mediciones']
        tiempo, posicion = correct_units(tiempo, posicion)
        errores_y = np.full(len(posicion), 0.1)
        popt, pcov = curve_fit(modelo_cuadratico, tiempo, posicion, sigma=errores_y, absolute_sigma=True)
        a_opt = 2*popt[0]
        error_a = np.sqrt(pcov[0, 0])
        aceleraciones.append(a_opt)
        errores_aceleracion.append(error_a)

    # Convertir la aceleración de cm/s² a m/s² ---> Ver si esta bien esto
    aceleraciones = np.array(aceleraciones) / 100
    errores_aceleracion = np.array(errores_aceleracion) / 100
        
    return np.mean(aceleraciones), np.mean(errores_aceleracion)

def mu_dinamico(m, M, a):
    g = 9.81
    return ((m + M) * a + M*g) / (m * g)

pesos = masas_dict('dataset/datos.txt')

m2 = pesos['trineo']
m3 = pesos['madera'] + pesos['trineo']
m4 = pesos['madera'] + pesos['plateada'] + pesos['trineo']
m5 = pesos['trineo'] + pesos['agua']
m6 = pesos['agua'] + pesos['plateada'] + pesos['trineo']
m7 = pesos['dorada'] + pesos['plateada'] + pesos['madera'] + pesos['trineo']

# Aceleracion con M = DORADA y distintas m para superficie de madera.

aceleraciones_madera = {}
pesos_madera = [m5, m6, m7]

for i, j in enumerate(range(5, 8)):
    data = csv_to_dict(f'dataset/prueba{j}.csv')
    aceleracion, error_aceleracion = promedio_aceleracion(f'dataset/prueba{j}.csv')
    aceleraciones_madera[pesos_madera[i]] = [aceleracion, error_aceleracion]

# Grafico de aceleracion vs m con M = DORADA
plt.errorbar(aceleraciones_madera.keys(), 
             [aceleraciones_madera[key][0] for key in aceleraciones_madera], 
             yerr=[aceleraciones_madera[key][1] for key in aceleraciones_madera], 
             fmt='o', color='b', capsize=5)
plt.title('Aceleración vs m con M = Masa dorada')
plt.xlabel('Masa m (g)')
plt.ylabel('Aceleración (m/s^2)')
plt.grid(True)
plt.show()

# Aceleracion con masas M = PLATA x2 y distntas m para superficie de papel.
aceleraciones_papel = {}
pesos_papel = [m2, m3, m4]

for i, j in enumerate(range(2, 5)):
    data = csv_to_dict(f'dataset/prueba{j}.csv')
    aceleracion, error_aceleracion = promedio_aceleracion(f'dataset/prueba{j}.csv')
    aceleraciones_papel[pesos_papel[i]] = [aceleracion, error_aceleracion]

# Grafico de aceleracion vs m con M = PLATA x2
plt.errorbar(aceleraciones_papel.keys(),
                [aceleraciones_papel[key][0] for key in aceleraciones_papel],
                yerr=[aceleraciones_papel[key][1] for key in aceleraciones_papel],
                fmt='o', color='b', capsize=5)
plt.title('Aceleración vs m con M = 2 masas de plata')
plt.xlabel('Masa m (g)')
plt.ylabel('Aceleración (m/s^2)')
plt.grid(True)
plt.show()

# Coeficiente de Rozamiento Dinamico para distintas superficies
g = 9.81
mu_pruebas_madera = []
mu_pruebas_papel = []
superficies = ['Madera', 'Papel']

for i in range(3):
    mu = mu_dinamico(pesos_madera[i], pesos['dorada'], aceleraciones_madera[pesos_madera[i]][0])
    mu_pruebas_madera.append(mu)

for i in range(3):
    mu = mu_dinamico(pesos_papel[i], 2 * pesos['plateada'], aceleraciones_papel[pesos_papel[i]][0])
    mu_pruebas_papel.append(mu)

mu_d1 = np.mean(mu_pruebas_madera)
mu_d2 = np.mean(mu_pruebas_papel)

# Grafico de coeficiente de rozamiento dinamico para distintas superficies

plt.figure()
plt.bar(superficies, [mu_d1, mu_d2], color='b')
plt.title('Coeficiente de Rozamiento Dinamico para distintas superficies')
plt.ylabel('Coeficiente de Rozamiento Dinamico')
plt.xlabel('Superficie')
plt.grid(True)
plt.show()