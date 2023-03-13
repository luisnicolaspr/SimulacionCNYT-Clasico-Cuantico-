''' ----------EJERCICIO DE PROGRAMACIÓN DRILL 3.1.1---------- '''

import numpy as np
import matplotlib.pyplot as plt

'''Canicas 3.1.1'''
def experimento_canicas_booleanos(matriz, estado_inicial, pasos):
    """
    Simula el experimento de canicas con coeficientes booleanos dado una matriz de transición,
    un estado inicial y un número de pasos.
    Devuelve el estado final después de realizar los pasos.
    """
    estado_actual = estado_inicial
    for i in range(pasos):
        estado_siguiente = [0] * len(estado_inicial)
        for j in range(len(matriz)):
            for k in range(len(matriz[j])):
                estado_siguiente[k] += estado_actual[j] * matriz[j][k]
        estado_actual = estado_siguiente
    return estado_actual

# Caso de prueba 1
matriz = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
estado_inicial = [1, 0, 0]
pasos = 2
resultado_esperado = [0, 1, 0]
resultado_obtenido = experimento_canicas_booleanos(matriz, estado_inicial, pasos)
assert resultado_obtenido == resultado_esperado, f"Error: resultado obtenido {resultado_obtenido}, resultado esperado {resultado_esperado}"

# Caso de prueba 2
matriz = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
estado_inicial = [1, 0, 0]
pasos = 3
resultado_esperado = [1, 0, 1]
resultado_obtenido = experimento_canicas_booleanos(matriz, estado_inicial, pasos)
assert resultado_obtenido == resultado_esperado, f"Error: resultado obtenido {resultado_obtenido}, resultado esperado {resultado_esperado}"

# Caso de prueba 3
matriz = [[0, 1, 1], [1, 0, 0], [0, 1, 0]]
estado_inicial = [1, 0, 0]
pasos = 2
resultado_esperado = [1, 1, 0]
resultado_obtenido = experimento_canicas_booleanos(matriz, estado_inicial, pasos)
assert resultado_obtenido == resultado_esperado, f"Error: resultado obtenido {resultado_obtenido}, resultado esperado {resultado_esperado}"

'''Rendijas clásico 3.1.1'''

def experimento_rendijas_clasico(matriz, estado_inicial, pasos):
    """
    Simula el experimento de múltiples rendijas clásico probabilístico dado una matriz de transición,
    un estado inicial y un número de pasos.
    Devuelve el estado final después de realizar los pasos.
    """
    estado_actual = estado_inicial
    for i in range(pasos):
        estado_siguiente = [0] * len(estado_inicial)
        for j in range(len(matriz)):
            for k in range(len(matriz[j])):
                estado_siguiente[k] += estado_actual[j] * matriz[j][k]
        estado_actual = estado_siguiente
    return estado_actual


# Caso de prueba 
matriz = [[0, 0, 0, 0, 0, 0, 0, 0],
          [1/3, 0, 0, 0, 0, 0, 0, 0],
          [1/3, 0, 0, 0, 0, 0, 0, 0],
          [1/3, 1/2, 0, 0, 0, 0, 0, 0],
          [0, 1/2, 1/3, 0, 0, 0, 0, 0],
          [0, 0, 1/3, 1/2, 0, 0, 0, 0],
          [0, 0, 0, 0, 1/2, 1/2, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1]]
estado_inicial = [1, 0, 0, 0, 0, 0, 0, 0]
pasos = 2
resultado_esperado = [0, 0, 0, 0, 0, 0, 1/6, 1/6]
resultado_obtenido = experimento_rendijas_clasico(matriz, estado_inicial, pasos)
assert resultado_obtenido == resultado_esperado, f"Error: resultado obtenido {resultado_obtenido}, resultado esperado {resultado_esperado}"

''' Rendijas Cuántico 3.1.1'''

import numpy as np

def experimento_rendijas_cuantico(matriz, estado_inicial, pasos):
    """
    Simula el experimento de múltiples rendijas cuántico dado una matriz de transición,
    un estado inicial y un número de pasos.
    Devuelve el estado final después de realizar los pasos.
    """
    estado_actual = np.array(estado_inicial)
    matriz_compleja = np.array(matriz, dtype=complex)
    for i in range(pasos):
        estado_siguiente = np.dot(matriz_compleja, estado_actual)
        estado_actual = estado_siguiente
    return estado_actual.real.tolist()

# Caso de prueba 
matriz = [[0, 0, 0, 0, 0, 0, 0, 0],
          [1/np.sqrt(3), 0, 0, 0, 0, 0, 0, 0],
          [1/np.sqrt(3), 0, 0, 0, 0, 0, 0, 0],
          [1/np.sqrt(3), 1/np.sqrt(2), 0, 0, 0, 0, 0, 0],
          [0, 1/np.sqrt(2), 1/np.sqrt(3), 0, 0, 0, 0, 0],
          [0, 0, 1/np.sqrt(3), 1/np.sqrt(2), 0, 0, 0, 0],
          [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1]]
estado_inicial = [1, 0, 0, 0, 0, 0, 0, 0]
pasos = 2
resultado_esperado = [0, 0, 0, 0, 0, 0, 1/6, 1/6]
resultado_obtenido = experimento_rendijas_cuantico(matriz, estado_inicial, pasos)
assert np.allclose(resultado_obtenido, resultado_esperado, f"Error: resultado obtenido {resultado_obtenido}, resultado esperado")

'''Gráfica e imágen vector de estados 3.1.1'''

import matplotlib.pyplot as plt

def graficar_probabilidades(vector, nombres_estados=None, titulo=None, guardar_imagen=False, nombre_archivo=None):
    """
    Grafica un vector de estados mostrando las probabilidades de cada estado en un diagrama de barras.
    Los nombres de los estados se pueden especificar con una lista de nombres.
    El título del gráfico se puede especificar con el parámetro titulo.
    Si guardar_imagen es True, guarda la imagen con el nombre de archivo especificado.
    """
    probabilidades = np.abs(np.array(vector))**2
    if nombres_estados is not None:
        if len(nombres_estados) != len(vector):
            raise ValueError('El número de nombres de estados debe coincidir con el número de elementos del vector.')
        estados = nombres_estados
    else:
        estados = range(len(vector))
    plt.bar(estados, probabilidades)
    plt.xticks(estados, estados)
    plt.ylabel('Probabilidad')
    if titulo is not None:
        plt.title(titulo)
    plt.show()
    if guardar_imagen:
        if nombre_archivo is None:
            raise ValueError('Debe especificar un nombre de archivo para guardar la imagen.')
        plt.savefig(nombre_archivo)

# Caso de prueba 
vector = [1/np.sqrt(2), 1/np.sqrt(2)]
nombres_estados = ['0', '1']
titulo = 'Probabilidades de los estados'
graficar_probabilidades(vector, nombres_estados, titulo, True, 'probabilidades.png')



''' ----------EJERCICIO DE PROGRAMACIÓN DRILL 3.2.1---------- ''' 

import numpy as np
from fractions import Fraction

def evolucionar_sistema(matriz_movimiento, vector_estado_inicial, num_ciclos):
    """
    Calcula la evolución de un sistema de canicas a lo largo de un número determinado de ciclos.
    La matriz_movimiento es una matriz cuadrada que describe el movimiento de las canicas.
    El vector_estado_inicial describe la cantidad de canicas en cada vértice al inicio del experimento.
    El número de ciclos determina la cantidad de veces que se aplicará la matriz de movimiento al vector de estado.
    """
    if not isinstance(matriz_movimiento[0][0], Fraction):
        matriz_movimiento = np.array(matriz_movimiento, dtype=float)
    vector_estado_actual = np.array(vector_estado_inicial, dtype=Fraction)
    for i in range(num_ciclos):
        vector_estado_actual = np.dot(matriz_movimiento, vector_estado_actual)
    return vector_estado_actual

# Caso de prueba 1
matriz_movimiento = [[0, Fraction(1, 2), Fraction(1, 2)],
                     [Fraction(1, 2), 0, Fraction(1, 2)],
                     [Fraction(1, 2), Fraction(1, 2), 0]]
vector_estado_inicial = [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)]
num_ciclos = 2
vector_estado_final = evolucionar_sistema(matriz_movimiento, vector_estado_inicial, num_ciclos)
print(vector_estado_final)  # [1/4, 1/2, 1/4]

# Caso de prueba 2
matriz_movimiento = [[Fraction(2, 3), Fraction(1, 3)],
                     [Fraction(1, 4), Fraction(3, 4)]]
vector_estado_inicial = [Fraction(1, 2), Fraction(1, 2)]
num_ciclos = 3
vector_estado_final = evolucionar_sistema(matriz_movimiento, vector_estado_inicial, num_ciclos)
print(vector_estado_final)  # [5/12, 7/12]


''' ----------EJERCICIO DE PROGRAMACIÓN DRILL 3.2.2---------- ''' 

import numpy as np

def experimento_multi_rendija(probabilidades, num_rendijas, num_objetivos, num_ciclos):
    """
    Simula un experimento de múltiples rendijas con un número determinado de rendijas, objetivos y ciclos.
    Las probabilidades indican la probabilidad de que una bala pase de una rendija a un objetivo en particular.
    """
    matriz_movimiento = np.array(probabilidades)
    vector_estado_inicial = np.zeros(num_rendijas)
    vector_estado_inicial[0] = 1
    for i in range(num_ciclos):
        vector_estado_final = evolucionar_sistema(matriz_movimiento, vector_estado_inicial, 1)
        vector_estado_inicial = vector_estado_final
    return vector_estado_final

# Caso de prueba 1
probabilidades = [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]]
num_rendijas = 4
num_objetivos = 2
num_ciclos = 2
vector_estado_final = experimento_multi_rendija(probabilidades, num_rendijas, num_objetivos, num_ciclos)
print(vector_estado_final)  # [0.25 0.25 0.25 0.25]

# Caso de prueba 2
probabilidades = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
num_rendijas = 3
num_objetivos = 2
num_ciclos = 3
vector_estado_final = experimento_multi_rendija(probabilidades, num_rendijas, num_objetivos, num_ciclos)
print(vector_estado_final)  # [0.3125 0.375  0.3125]

''' ----------EJERCICIO DE PROGRAMACIÓN DRILL 3.3.1---------- ''' 

import numpy as np

def multi_rendija_cuántico(prob_matrix, state_vector):
    # Normalizamos el vector de estados
    state_vector = state_vector / np.linalg.norm(state_vector)
    # Obtenemos la matriz unitaria correspondiente a la prob_matrix
    unitary_matrix = np.zeros_like(prob_matrix)
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            unitary_matrix[i][j] = np.sqrt(prob_matrix[i][j])
    # Obtenemos el vector de estados después de pasar por la matriz unitaria
    new_state_vector = np.dot(unitary_matrix, state_vector)
    # Normalizamos el vector resultante
    new_state_vector = new_state_vector / np.linalg.norm(new_state_vector)
    return new_state_vector

# Ejemplo de uso de la función multi_rendija_cuántico
prob_matrix = np.array([[0.3+0.4j, 0.2+0.1j, 0.2+0.3j],
                        [0.3+0.2j, 0.2+0.2j, 0.1+0.1j],
                        [0.4+0.4j, 0.6+0.1j, 0.7+0.2j]])
state_vector = np.array([0.4+0.3j, 0.1+0.2j, 0.2+0.5j])
new_state_vector = multi_rendija_cuántico(prob_matrix, state_vector)
print(new_state_vector)


from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

''' ----------EJERCICIO DE PROGRAMACIÓN DRILL 3.3.2---------- ''' 

def multi_slit_experiment(n_slits: int, n_targets: int, prob_matrix: np.ndarray, time: int) -> np.ndarray:
    """
   Realice un experimento de múltiples rendijas con probabilidades de transición complejas y devuelva la distribución de probabilidad final.

     Argumentos:
     - n_slits: el número de rendijas.
     - n_targets: el número de objetivos.
     - prob_matrix: una matriz numpy compleja de forma (n_slits, n_targets) que representa las probabilidades de transición.
     - tiempo: el número de iteraciones a realizar.

     Salidas:
     - Una matriz numérica de forma (n_objetivos) que representa la distribución de probabilidad final.
    """
    prob_dist = np.zeros(n_targets, dtype=np.complex128)
    prob_dist[0] = 1  # establecer la distribución de probabilidad inicial

    for i in range(time):
        prob_dist = np.matmul(prob_matrix, prob_dist)

    return prob_dist


def is_interference_present(prob_dist: np.ndarray) -> bool:
    """
    Compruebe si hay interferencia en la distribución de probabilidad.

     Argumentos:
     - prob_dist: una matriz numpy que representa la distribución de probabilidad.

     Salidas:
     - Verdadero si hay interferencia, falso en caso contrario.
    """
    return not np.allclose(prob_dist.real, prob_dist.imag)


def plot_prob_distribution(prob_dist: np.ndarray, title: str, save_path: str):
    """
    Trace la distribución de probabilidad como un gráfico de barras y guárdelo en la ruta de archivo especificada.

     Argumentos:
     - prob_dist: una matriz numpy que representa la distribución de probabilidad.
     - título: una cadena que representa el título de la trama.
     - save_path: una cadena que representa la ruta del archivo para guardar el gráfico.
    """
    n_targets = len(prob_dist)
    x_ticks = np.arange(1, n_targets+1)
    plt.bar(x_ticks, prob_dist.real)
    plt.xticks(x_ticks)
    plt.title(title)
    plt.xlabel("Objetivo")
    plt.ylabel("Probabilidad")
    plt.savefig(save_path)


# Ejemplo de uso
n_slits = 3
n_targets = 4
prob_matrix = np.array([[0.3+0.2j, 0.1-0.1j, 0.6-0.1j, 0.0+0.0j],
                        [0.2+0.3j, 0.2-0.2j, 0.5+0.1j, 0.1-0.2j],
                        [0.4-0.1j, 0.1+0.2j, 0.4+0.3j, 0.1-0.2j]])
time = 2

final_prob_dist = multi_slit_experiment(n_slits, n_targets, prob_matrix, time)
plot_prob_distribution(final_prob_dist, "Distribución de probabilidad final", "prob_dist.png")

if is_interference_present(final_prob_dist):
    print("La interferencia está presente.")
else:
    print("La interferencia no está presente.")
