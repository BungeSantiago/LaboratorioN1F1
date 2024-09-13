import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# PRUEBA 1:
# superficie ----> papel
# friccion -> trineo con papel
# m = 2 plateadas 1 dorada y 1 madera y el carrito
# M = masa dorada

# cada 300 milisegundos

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

def correct_units(time:np.array, distance:np.array, pendiente:float, incerteza_pendiente:float, ordenada:float, incerteza_ordenada:float) -> tuple:
    '''
    Convierte las unidades de tiempo (milisegundos) a segundos y la distancia a centímetros.
    '''
    time = time / 1000
    distance_cm = pendiente * distance + ordenada
    incerteza_distance_cm = np.sqrt((distance * incerteza_pendiente)**2 + incerteza_ordenada**2)
    return time, distance_cm, incerteza_distance_cm

# Datos del ajuste lineal anterior
pendiente = 0.01736595797123086
incerteza_pendiente = 0.0006720241499391473
ordenada = -0.6682770640993427
incerteza_ordenada = 0.6843903118668289

pruebas = ['dataset/prueba4.csv', 'dataset/prueba5.csv']
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Definir la función cuadrática con v_0 = 0
modelo_cuadratico = lambda t, a, v_0, x_0: 0.5 * a * t**2 + v_0 * t +  x_0 


for i, prueba in enumerate(pruebas):
    data = csv_to_dict(prueba)
    tiempo, posicion = data['test 1']['milisegundos'], data['test 1']['mediciones']
    tiempo, posicion, incerteza_posicion = correct_units(tiempo, posicion, pendiente, incerteza_pendiente, ordenada, incerteza_ordenada)

    # Datos de ejemplo (tiempo, posición y errores en y)
    errores_y = np.full(len(posicion), 0.1)

    # Ajustar la curva
    popt, pcov = curve_fit(modelo_cuadratico, tiempo, posicion, sigma=errores_y, absolute_sigma=True)

    # Obtener los coeficientes ajustados y sus errores
    a_opt, v_0_opt, x_0_opt = popt
    errores = np.sqrt(np.diag(pcov))

    print(f'PRUEBA {prueba[14]}')
    print(f'Coeficientes ajustados: {popt}')
    print(f'Incertezas: {errores}')

    print(f"Aceleración a: {a_opt:.1f} ± {errores[0]:.1f} cm /s²")
    print(f"Velocidad inicial v_0: {v_0_opt:.0f} ± {errores[1]:.0f} cm/s")
    print(f"Posición inicial x_0: {x_0_opt:.0f} ± {errores[2]:.0f} cm")

    # Graficar los datos y el ajuste
    t_ajuste = np.linspace(min(tiempo), max(tiempo), 100)
    ax[i].errorbar(tiempo, posicion, yerr=incerteza_posicion, fmt='o', label='Datos con incertezas')
    ax[i].plot(t_ajuste, modelo_cuadratico(t_ajuste, *popt), 'r', label=f'Ajuste cuadrático')
    ax[i].set_xlabel('Tiempo [s]')
    ax[i].set_ylabel('Posición [cm]')
    ax[i].legend()

plt.show()
