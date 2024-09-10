import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

def aceleracion(tiempo, posicion):
    errores_y = np.full(len(posicion), 0.1) 

    modelo_cuadratico = lambda t, a, v_0, x_0: a * t**2 + v_0 * t +  x_0 

    popt, pcov = curve_fit(modelo_cuadratico, tiempo, posicion, sigma=errores_y, absolute_sigma=True)

    a_opt, v_0_opt, x_0_opt = popt

    errores = np.sqrt(np.diag(pcov))

    return a_opt, v_0_opt, x_0_opt, errores

data = csv_to_dict('dataset/prueba3.csv')
a = []

for i in range(1, 4):
    tiempo, posicion = data[f'test {i}']['milisegundos'], data[f'test {i}']['mediciones']
    tiempo, posicion = correct_units(tiempo, posicion)
    a_opt, v_0_opt, x_0_opt, errores = aceleracion(tiempo, posicion)
    print(f'acc test {i}: {2*a_opt}')
    a.append([a_opt, errores[0]])

a = sum([2*acc for acc, err in a])/len(a)
print(f'promedio acc = {a}')


# {110.0: [12.648996411425353, 0.16560645107233699], 116.0: [19.475777742395902, 0.15926021000032972], 138.0: [6.075825878446779, 0.033566061656943584]}
# Pruebas 2, 3, 4