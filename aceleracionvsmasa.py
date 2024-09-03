import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# PRUEBA 1:
# superficie ----> papel
# friccion -> trineo con papel
# m = 2 plateadas 1 dorada y 1 madera y el carrito
# M = masa dorada

# cada 300 milisegundos

# Preguntar si hay que pasar las unidades a cm

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
    a = 0.017365957941399224
    b = -0.668277026706441
    
    time = time / 1000
    distance = a * distance + b

    return time, distance

data = csv_to_dict('dataset/prueba1.csv')
tiempo, posicion = data['test 1']['milisegundos'], data['test 1']['mediciones']
tiempo, posicion = correct_units(tiempo, posicion)

