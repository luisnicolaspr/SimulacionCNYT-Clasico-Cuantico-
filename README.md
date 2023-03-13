# SimulacionCNYT-Clasico-Cuantico-
Retos de programación del capítulo 3 del libro Quantum Computing for Computer Scientists

#### 👨🏻 NOMBRE: Luis Nicolás Pinilla Rodríguez CNYT 2023-1

### ✅ Reto de programación 3.1.1

La función experimento_canicas_booleanos recibe como parámetros una matriz de transición, un estado inicial y un número de pasos. La matriz de transición es una matriz booleana que indica la forma en que las canicas se mueven de un vértice a otro. El estado inicial es una lista que contiene la cantidad de canicas en cada vértice. El número de pasos indica la cantidad de veces que se aplicará la matriz de transición al estado inicial.

La función simula el experimento de canicas con coeficientes booleanos multiplicando la matriz de transición por el estado inicial en cada paso y actualizando el estado actual con el resultado. Finalmente, devuelve el estado final después de realizar los pasos.


La función experimento_rendijas_clasico recibe como parámetros una matriz de transición, un estado inicial y un número de pasos. La matriz de transición es una matriz de probabilidades que indica la probabilidad de que una partícula en una ranura salte a otra ranura. El estado inicial es una lista que contiene la probabilidad de que la partícula esté en cada ranura. El número de pasos indica la cantidad de veces que se aplicará la matriz de transición al estado inicial.

La función simula el experimento de las múltiples rendijas clásico probabilístico multiplicando la matriz de transición por el estado inicial en cada paso y actualizando el estado actual con el resultado. Finalmente, devuelve el estado final después de realizar los pasos.

La función experimento_rendijas_cuantico recibe como parámetros una matriz de transición, un estado inicial y un número de pasos. La matriz de transición es una matriz de amplitud de probabilidad que indica la probabilidad de que una partícula en una ranura salte a otra ranura. El estado inicial es una lista que contiene la amplitud de probabilidad de que la partícula esté en cada ranura. El número de pasos indica la cantidad de veces que se aplicará la matriz de transición al estado inicial.

La función simula el experimento de las múltiples rendijas cuántico multiplicando la matriz de transición por el estado inicial en cada paso y actualizando el estado actual con el resultado. Finalmente, devuelve el estado final después de realizar los pasos.

Es importante notar que en esta función se utiliza el tipo de datos complex de numpy para trabajar con números complejos.

La función graficar_probabilidades recibe como parámetros un vector de estados, una lista de nombres de estados (opcional), un título para el gráfico (opcional), un indicador para guardar la imagen y un nombre de archivo (opcional). La función calcula las probabilidades de cada estado a partir de su amplitud de probabilidad y las grafica en un diagrama de barras utilizando la librería matplotlib.

En caso de que se quiera guardar la imagen generada, se puede especificar un nombre de archivo con la extensión del formato deseado (por ejemplo, "imagen.png"). La imagen se guardará en la carpeta actual de trabajo.

### ✅ Reto de programación 3.2.1

Para realizar el ejercicio 3.2.1, se puede modificar la función evolucionar_sistema para que acepte matrices con fracciones en lugar de valores booleanos. Para ello, se puede utilizar la clase Fraction de la librería fractions de Python.

La función evolucionar_sistema recibe como parámetros una matriz de movimiento con valores fraccionales, un vector de estado inicial también con valores fraccionales y un número de ciclos. En caso de que la matriz de movimiento no contenga valores fraccionales, se convierte a un array de numpy con tipo de datos float para evitar errores de precisión.

### ✅ Reto de programación 3.2.2

Para resolver el ejercicio 3.2.2, podemos utilizar la misma función evolucionar_sistema que creamos para el ejercicio 3.1.1 y el ejercicio 3.2.1, pero en lugar de simular el experimento de una sola rendija, tendremos que pedir al usuario que ingrese la cantidad de rendijas y objetivos que desea simular, así como las probabilidades de que una bala pase de una rendija a un objetivo en particular.

La función experimento_multi_rendija recibe como parámetros una lista de listas de probabilidades, que representa la matriz de transición, así como el número de rendijas, objetivos y ciclos que se quieren simular. La función también utiliza la función evolucionar_sistema que definimos anteriormente para realizar la evolución del sistema.

### ✅ Reto de programación 3.3.1

Para el Programming Drill 3.3.1 necesitamos hacer algunas modificaciones al código del Programming Drill 3.2.1 para permitir el uso de números complejos en la matriz de probabilidades. Vamos a definir una nueva función multi_rendija_cuántico que tendrá como entrada una matriz compleja de probabilidades y un vector de estados complejos, y devolverá el vector de estados después de haber pasado por la rendija cuántica. Ahora podemos probar la función con un ejemplo de 3 rendijas y 2 objetivos, con una matriz de probabilidades complejas aleatorias y un vector de estados inicial también aleatorio.Debería imprimir el vector de estados después de pasar por la rendija cuántica correspondiente a la matriz de probabilidades y el vector de estados iniciales.

### ✅ Reto de programación 3.3.2

Para resolver este ejercicio, podemos modificar la función multi_slit_experiment creada anteriormente en el ejercicio 3.2.2, para que ahora acepte matrices de números complejos en lugar de matrices de fracciones. Además, podemos agregar una función auxiliar is_interference_present que verifica si hay interferencia en la distribución de probabilidad resultante.

