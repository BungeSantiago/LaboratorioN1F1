import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def masas_dict(file_path:str) -> dict:
    '''
    Pasa los datos del csv de las masas con sus pesos a un diccionario con la siguiente estructura:
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
    Pasa los datos del csv de una prueba a un diccionario con la siguiente estructura:
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

def promedio_aceleracion(prueba:str) -> tuple:
    '''
    Calcula la aceleración de una prueba promediando los valores de los 3 tests.
    '''
    data = csv_to_dict(prueba)
    modelo_cuadratico = lambda t, a, v_0, x_0: 0.5 * a * t**2 + v_0 * t +  x_0
    aceleraciones = []
    errores_aceleracion = []

    for test in data:
        tiempo, posicion = data[test]['milisegundos'], data[test]['mediciones']
        tiempo, posicion = correct_units(tiempo, posicion)
        errores_y = np.full(len(posicion), 0.1)
        popt, pcov = curve_fit(modelo_cuadratico, tiempo, posicion, sigma=errores_y, absolute_sigma=True)
        a_opt = popt[0]
        error_a = np.sqrt(pcov[0, 0])
        aceleraciones.append(a_opt)
        errores_aceleracion.append(error_a)
        
    aceleraciones = np.array(aceleraciones)
    errores_aceleracion = np.array(errores_aceleracion)

    return np.mean(aceleraciones), np.mean(errores_aceleracion)

def mu_dinamico(m:float, M:float, a:float) -> float:
    '''
    Calcula el coeficiente de rozamiento dinámico.
    '''
    g = 9.81
    # Pasamos la aceleracion de cm/s² a m/s²
    a = a / 100
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
# SACAR EL TITULO
plt.xlabel('Masa m [g]')
plt.ylabel('Aceleración [cm/s²]')
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
# SACAR EL TITULO
plt.xlabel('Masa m [g]')
plt.ylabel('Aceleración [cm/s²]')
plt.grid(True)
plt.show()

# Coeficiente de Rozamiento Dinamico para distintas superficies
mu_pruebas_madera = []
mu_pruebas_papel = []
superficies = ['Madera', 'Papel']

for i in range(3):
    # Mu dinamico para madera
    mu1 = mu_dinamico(pesos_madera[i], pesos['dorada'], aceleraciones_madera[pesos_madera[i]][0])
    mu_pruebas_madera.append(mu1)

    # Mu dinamico para papel
    mu = mu_dinamico(pesos_papel[i], 2 * pesos['plateada'], aceleraciones_papel[pesos_papel[i]][0])
    mu_pruebas_papel.append(mu)

mu_d1 = np.mean(mu_pruebas_madera)
mu_d2 = np.mean(mu_pruebas_papel)

print(f'Coeficiente de Rozamiento Dinamico para madera: {mu_d1}')
print(f'Coeficiente de Rozamiento Dinamico para papel: {mu_d2}')

# FALTA EL ERROR DE MU DINAMICO
def error_mu_dinamico(m: float, M: float, a: float, error_a: float) -> float:
    '''
    Calcula el error en el coeficiente de rozamiento dinámico.
    '''
    g = 9.81
    
    # Derivada parcial de mu_d respecto a a:
    dmu_da = (m + M) / (m * g)
    
    # Propagación de errores:
    error_mu_d = dmu_da * error_a
    
    return error_mu_d

# Calculo de errores para mu dinámico
errores_mu_madera = []
errores_mu_papel = []

for i in range(3):
    # Error de mu dinamico para madera
    error_mu1 = error_mu_dinamico(pesos_madera[i], pesos['dorada'], 
                                  aceleraciones_madera[pesos_madera[i]][0], 
                                  aceleraciones_madera[pesos_madera[i]][1])
    errores_mu_madera.append(error_mu1)
    
    # Error de mu dinamico para papel
    error_mu2 = error_mu_dinamico(pesos_papel[i], 2 * pesos['plateada'], 
                                  aceleraciones_papel[pesos_papel[i]][0], 
                                  aceleraciones_papel[pesos_papel[i]][1])
    errores_mu_papel.append(error_mu2)

error_mu_d_madera_mean = np.mean(errores_mu_madera)
error_mu_d_papel_mean = np.mean(errores_mu_papel)

# Grafico de coeficiente de rozamiento dinamico para distintas superficies
plt.figure()
plt.bar(superficies, [mu_d1, mu_d2], color='b')
plt.errorbar(superficies, 
             [mu_d1, mu_d2], 
             yerr=[error_mu_d_madera_mean, error_mu_d_papel_mean], 
             fmt='o', color='r', capsize=5)
plt.ylabel('Coeficiente de Rozamiento Dinamico')
plt.xlabel('Superficie')
plt.grid(True)
plt.show()