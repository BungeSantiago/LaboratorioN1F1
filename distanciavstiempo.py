import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# PRUEBA 1:
# superficie ----> papel
# friccion -> trineo con papel
# m = 2 plateadas 1 dorada y 1 madera y el carrito
# M = masa dorada

# cada 300 milisegundos

# Saco el promedio de las aceleraciones de cada test

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
tiempo, posicion = data['test 3']['milisegundos'], data['test 3']['mediciones']
tiempo, posicion = correct_units(tiempo, posicion)

# Datos de ejemplo (tiempo, posición y errores en y)
errores_y = np.full(len(posicion), 0.1)

# Definir la función cuadrática con v_0 = 0
modelo_cuadratico = lambda t, a, v_0, x_0: a * t**2 + v_0 * t +  x_0 

# Ajustar la curva
popt, pcov = curve_fit(modelo_cuadratico, tiempo, posicion, sigma=errores_y, absolute_sigma=True)

# Obtener los coeficientes ajustados y sus errores
a_opt, v_0_opt, x_0_opt = popt
errores = np.sqrt(np.diag(pcov))

print(f"Aceleración a: {a_opt:.1f} ± {errores[0]:.1f} m /s^2")
print(f"Velocidad inicial v_0: {v_0_opt:.0f} ± {errores[1]:.0f} m /s")
print(f"Posición inicial x_0: {x_0_opt:.0f} ± {errores[2]:.0f}m")

# Graficar los datos y el ajuste
t_ajuste = np.linspace(min(tiempo), max(tiempo), 100)
plt.errorbar(tiempo, posicion, yerr=errores_y, fmt='o', label='Datos')

plt.plot(t_ajuste, modelo_cuadratico(t_ajuste, *popt), 'r', label=f'Ajuste cuadrático')
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición [cm]')
plt.legend()
plt.show()
