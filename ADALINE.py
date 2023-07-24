from FUNCTIONS import *
import random
import csv
import numpy as np

class ADALINE(FUNCTIONS):
    """ CLASE QUE IMPLEMENTA LA ESTRUCTURA Y FUNCIONALIDADES DEL ADALINE """
    def __init__(self, n_inputs: int, bias: bool, activation_function: str, n_outputs: int, manual_weights: list = None):
        """ Constructor del objeto ADALINE que recibe como argumentos de entrada:
            -> El número de entradas del ADALINE (dimensión de los datos de entrada)
            -> Un booleano que indica si el ADALINE utiliza o no un umbral
            -> La función de activación que se desee usar
            -> El número de salidas del ADALINE (dimensión de los datos de salida)
            -> (OPCIONAL) Los pesos iniciales del ADALINE (si no se introduce nada, los pesos serán aleatorios) """
        self.n_inputs = n_inputs # Número de inputs
        self.bias = bias # Opción umbral
        self.activation_function = self.select_activation_function(activation_function) # Función de activación
        self.n_outputs = n_outputs # Número de outputs

        # Si los pesos se introducen de forma manual no se randomizan
        if manual_weights:
            self.weights = np.array(manual_weights)
        # Si los pesos no se introducen de forma manual, se randomizan
        else:
            self.weights = self.initialize_weights()

    def initialize_weights(self) -> np.array:
        """ Función inicializar pesos de forma aleatoria """
        weights = []
        # Por cada salida del ADALINE hay tantos pesos como entradas tenga (n x m )
        for i in range(self.n_outputs):
            weights.append([])
            for j in range(self.n_inputs + 1): # El bucle realiza 1 iteración más porque se debe añadir el umbral randomizado
                weights[i].append(round(random.random(), 4))
        return np.array(weights)

    def select_activation_function(self, activation_function: str) -> FUNCTIONS:
        """ Función que permite seleccionar la función de activación del ADALINE """
        # Diccionario con las distintas funciones disponibles heredadas de la clase FUNCTIONS
        functions = {"unit_step": (FUNCTIONS.unit_step, None), 
                    "sign": (FUNCTIONS.sign, None),
                    "linear": (FUNCTIONS.linear, FUNCTIONS.d_linear),
                    "piece_wise_linear": (FUNCTIONS.piece_wise_linear, FUNCTIONS.d_piece_wise_linear),
                    "sigmoid": (FUNCTIONS.sigmoid, FUNCTIONS.d_sigmoid),
                    "hyperbolic_tangent": (FUNCTIONS.hyperbolic_tangent, FUNCTIONS.d_hyperbolic_tangent),
                    "relu": (FUNCTIONS.relu, FUNCTIONS.d_relu),
                    "soft_plus": (FUNCTIONS.soft_plus, FUNCTIONS.d_soft_plus)}
        # Se selecciona la función deseada (String de entrada) y se devuelve un objeto llamada a la función
        return functions[activation_function]

    def preprocess_data(self, x_matrix):
        """ Función que pre-procesa la matriz de datos """
        # Recorre cada patrón / ejemplo (x_vector) insertando un 0 (para neutralizar el umbral si bias == False) o un 1 en caso contrario
        for x_vector in x_matrix:
            if self.bias:
                x_vector.insert(0, 1)
            else:
                x_vector.insert(0, 0)

    def output(self, x_vector: list) -> np.array:
        """ Función que calcula el resultado (propagación hacia adelante) de un ADALINE """
        # El ejemplo de entrada de la red se convierte a un array de numpy (por cuestiones de optimizar las operaciones de matrices)
        x_vector = np.array(x_vector)
        output = []
        # Por cada salida del ADALINE se realiza el cálculo de la función de activación compuesta por el modelo de regresión lineal
        for i in range(self.n_outputs):
            output.append(self.activation_function[0](np.dot(self.weights[i], x_vector)))
        # El vector de salidas del ADALINE se convierte también en un array de numpy
        return np.array(output)

    def learning(self, x_matrix: list, d_matrix: list, learning_rate: float) -> None:
        """ Función que realiza el aprendizaje (1 ciclo) con una matriz de datos, una matriz de salidas y una tasa de aprendizaje """
        # Por cada patrón en la matriz de datos se calcula primero la salida correspondiente del ADALINE
        for pattern in range(len(x_matrix)):
            output = self.output(x_matrix[pattern])
            # Por cada "pin" de salida ajustamos los pesos correspondientes
            for output_pin in range(self.n_outputs):
                # Por cada peso utilizamos el método del Descenso de Gradiente para actualizar
                for weight_index in range(len(self.weights[output_pin])):
                    # wi <- wi - (tasa_aprendizaje) * (yj - dj) * derivada_función_activación * xi
                    # Donde i es el subíndice de pesos y j es el subíndice de patrones
                    self.weights[output_pin][weight_index] -= \
                    learning_rate * (output[output_pin] - d_matrix[pattern][output_pin]) * \
                    self.activation_function[1](np.dot(self.weights[output_pin], x_matrix[pattern])) * x_matrix[pattern][weight_index]

    def learning_cycles(self, x_matrix: list, d_matrix: list, learning_rate: float, n_cycles: int) -> None:
        """ Función que preprocesa los datos inicialmente y que luego llama durante n-ciclos a la función learning """
        # Se preprocesan los datos
        self.preprocess_data(x_matrix)
        # Se realizan tantos ciclos de aprendizaje como especifique la variable entera n_cycles
        for cycle in range(n_cycles):
            self.learning(x_matrix, d_matrix, learning_rate)

    def evaluate_MSE(self, d_input_test_matrix: list, d_output_test_matrix: list) -> float:
        """ Función que calcula el error cuadrático medio obtenido con el conjunto de test """
        # Inicializamos el error cuadrático medio a cero
        MSE = 0
        # Recorremos todos los patrones del conjunto de test
        for pattern_index in range(len(d_input_test_matrix)):
            # Calculamos las salidas obtenidas del ADALINE con los datos de entrada del conjunto test
            output = self.output(d_input_test_matrix[pattern_index])
            # Recorremos cada uno de los atributos de los datos de salida (número de outputs)
            for attribute_index in range(self.n_outputs):
                # Añadimos al MSE la diferencia entre (di - yi) ^ 2
                MSE += pow(output[attribute_index] - d_output_test_matrix[pattern_index][attribute_index], 2)
        return MSE / len(d_input_test_matrix)

    def stop_condition(self, MSE: float, previous_MSE: float) -> bool:
        """ Función que detecta si se cumplen las condiciones de parada del proceso de aprendizaje
            -> NOTA: El MSE se calcula utilizando los datos de TEST, no los de entrenamiento """
        # Si el MSE actual es menor que el obtenido en el ciclo de entrenamiento anterior, sigue iterando (NO para)
        if MSE < previous_MSE:
            # Si el MSE actual se ha mejorado por debajo del decimal asociado a 10^(-5) se detiene la ejecución (ajuste ínfimo)
            if abs(MSE - previous_MSE) <= 0.00001:
                return True
            else:
                print("Learning: " + str(round(0.001 / abs(MSE - previous_MSE), 2)) + '%')
                return False
        # Si el MSE actual es menor o igual que el obtenido en el ciclo anterior, seguir iterando provocaría sobreentrenamiento (SÍ para)
        else:
            return True

    def learning_until_overtraining(self, x_matrix: list, d_matrix: list, learning_rate: float, d_input_validate_matrix: list, d_output_validate_matrix: list) -> tuple:
        """ Función que entrena el ADALINE por ciclos hasta que el modelo comienza a sobre-ajustarse (sobreentrenamiento) """
        # Creamos dos listas que contengan los MSE obtenidos en aprendizaje y validación por cada ciclo
        plot_MSE_training = [[], []] # Primera sub-lista para los ciclos y segunda sub-lista para el MSE
        plot_MSE_validation = [[], []]
        # Preprocesamos las matrices de datos de entrada (los del conjunto de entrenamiento y de test)
        self.preprocess_data(x_matrix)
        self.preprocess_data(d_input_validate_matrix)
        # Creamos una varibale que almacene una copia de los pesos del ciclo de aprendizaje anterior
        previous_weights = None
        # Creamos una variable que almacene el MSE del ciclo de aprendizaje anterior. Se inicializa a infinito
        previous_MSE_validation = float('INF')
        # Creamos variables MSE donde cargamos el error cuadrático medio inicial
        MSE_training = self.evaluate_MSE(x_matrix, d_matrix)
        MSE_validation = self.evaluate_MSE(d_input_validate_matrix, d_output_validate_matrix)
        # Añadimos los primeros MSE a la lista de errores para la gráfica
        plot_MSE_training[0].append(0)
        plot_MSE_training[1].append(MSE_training)
        plot_MSE_validation[0].append(0)
        plot_MSE_validation[1].append(MSE_validation)
        cycles = 0 # Se crea una variable para contar el número de ciclos de aprendizaje (se necesita para la gráfica)
        # Realizamos ciclos de aprendizaje hasta que se cumpla la condición de parada
        while not self.stop_condition(MSE_validation, previous_MSE_validation):
            cycles += 1
            # Guardamos los pesos actuales para recuperarlos en caso de que el ciclo de aprendizaje incremente el MSE respecto al conjunto test
            previous_weights = np.copy(self.weights)
            # Actualizo el MSE previo como el MSE actual
            previous_MSE_validation = MSE_validation
            # Realizamos un ciclo de aprendizaje
            self.learning(x_matrix, d_matrix, learning_rate)
            # Actualizamos el MSE con los pesos resultantes del ciclo de aprendizaje
            MSE_training = self.evaluate_MSE(x_matrix, d_matrix)
            MSE_validation = self.evaluate_MSE(d_input_validate_matrix, d_output_validate_matrix)
            # Añadimos los MSE recién calculados después de completar el ciclo de aprendizaje
            plot_MSE_training[0].append(cycles)
            plot_MSE_training[1].append(MSE_training)
            plot_MSE_validation[0].append(cycles)
            plot_MSE_validation[1].append(MSE_validation)
        # Cuando se cumple la condición de parada (los pesos actuales no son óptimos) recuperamos la configuración de pesos previa (óptima)
        self.weights = previous_weights
        # Devolvemos el MSE previo para evaluar la calidad y fiabilidad del modelo
        return previous_MSE_validation, plot_MSE_training, plot_MSE_validation

    @staticmethod
    def socket(csv_path: str):
        """ Función socket que prepara los datos de los CSV a listas de python (el formato de entrada que requiere el ADALINE) """
        # Creamos la matriz de datos que se completará con la información del CSV
        processed_csv = []
        with open(csv_path, 'r') as file:
            reader = csv.reader(file, delimiter='t')
            # Leo el CSV ignorando la primera fila que contiene los nombres de los atributos
            for i, line in enumerate(reader):
                if i > 0:
                    newLine = line[0].split(',')
                    processed_csv.append(newLine)
            for i in range(len(processed_csv)):
                # Convierto cada elemento de la matriz a un float (originalmente es un String)
                for j in range(len(processed_csv[i])):
                    processed_csv[i][j] = float(processed_csv[i][j])
        # Devuelvo la matriz transformada
        return processed_csv

    @staticmethod
    def denormalize(normalized_value: float, max_value: float, min_value: float) -> float:
        """ Función que desnormaliza un valor dado su instancia máxima y mínima """
        return normalized_value * (max_value - min_value) + min_value

    def write_outputs(self, input_matrix: list, output_matrix: list, max_output: float, min_output: float) -> None:
        """ Función que escribe en un .txt (existente o no) las salidas obtenidas por el ADALINE y las esperadas """
        with open(r'outputs.txt', 'w', encoding='utf-8') as output_file:
            output_file.write("ADALINE TEST OUTPUTS" + '\n')
            output_file.write("ADALINE-OUTPUT --> DESIRED" + '\n')
            for pattern_index in range(len(input_matrix)):
                # En cada línea del .txt se escribirá la salida del ADALINE y la salida esperada (para poder hacer una comparación visual)
                output_file.write('#' + str(pattern_index) + ": " + \
                str(ADALINE.denormalize(self.output(input_matrix[pattern_index])[0], max_output, min_output)) + \
                 " --> " + str(ADALINE.denormalize(output_matrix[pattern_index][0], max_output, min_output)) + '\n')
