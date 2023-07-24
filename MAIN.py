from ADALINE import *
from matplotlib import pyplot as plt
from best_model import model
from sys import exit
import time

def main():
    # Ejemplo de uso del ADALINE
    start_time = time.time()

    MAX_OUTPUT = 99.0 # Valor máximo (desnormalizado) de la salida usr
    MIN_OUTPUT = 0.0 #  Valor mínimo (desnormalizado) de la salida usr

    #starting_weights = [[0.11, 0.15, 0.11, 3, 14, 8, 0.2, 1, 0.6, 8, 0.9, 6, 13, 7, 13, 1, 0.11, 0.13, 1, 0.13, 7, 12]]

    print("Seleccione el modelo (de datos) que desea usar para entrenar y testear (1,..,7): " '\n')
    data_model = input()

    print("¿Desea cargar el modelo (umbral y pesos) óptimo? s/n" + '\n')
    selection = input()

    if selection == 's' or selection == 'S':
        adaline = ADALINE(21, True, "linear", 1, model)
    elif selection == 'n' or selection == 'N':
        adaline = ADALINE(21, True, "linear", 1)
    else:
        print("ERROR: Selección inválida" + '\n')
        exit()

    print("¿Desea probar o entrenar el ADALINE? p/e" + '\n')
    selection = input()

    if selection == 'p' or selection == 'P':
        input_test_matrix = ADALINE.socket('models/Modelo' + data_model + '/Testing/Testing_IN.csv')
        output_test_matrix = ADALINE.socket('models/Modelo' + data_model + '/Testing/Testing_OUT.csv')
        adaline.preprocess_data(input_test_matrix)
        print("Test-MSE: " + str(adaline.evaluate_MSE(input_test_matrix, output_test_matrix)))
        adaline.write_outputs(input_test_matrix, output_test_matrix, MAX_OUTPUT, MIN_OUTPUT)
        exit()
    elif selection == 'e' or selection == 'E':
        print("Introduzca la tasa de aprendizaje [0, 1] :" + '\n')
        learning_rate = float(input())
    else:
        print("ERROR: Selección inválida" + '\n')
        exit()

    # Modelo 6 da el error mínimo en test*
    x_matrix = ADALINE.socket('models/Modelo' + data_model + '/Training/Training_IN.csv')
    d_matrix = ADALINE.socket('models/Modelo' + data_model + '/Training/Training_OUT.csv')
    input_validate_matrix = ADALINE.socket('models/Modelo' + data_model + '/Validation/Validation_IN.csv')
    output_validate_matrix = ADALINE.socket('models/Modelo' + data_model + '/Validation/Validation_OUT.csv')
    input_test_matrix = ADALINE.socket('models/Modelo' + data_model + '/Testing/Testing_IN.csv')
    output_test_matrix = ADALINE.socket('models/Modelo' + data_model + '/Testing/Testing_OUT.csv')

    # Se preprocesa la matriz de entrada de datos de test (para poder propagarlos hacia adelante y calcular el MSE)
    adaline.preprocess_data(input_test_matrix)
    
    print("INITIAL WEIGHTS: " + '\n' + str(adaline.weights) + '\n')
    print("INITIAL TEST-MSE: " + str(adaline.evaluate_MSE(input_test_matrix, output_test_matrix)))

    # Una tasa de aprendizaje de 0.1 funciona notablemente bien para el problema en concreto
    results = adaline.learning_until_overtraining(x_matrix, d_matrix, learning_rate, input_validate_matrix, output_validate_matrix)
    print("FINAL VALIDATION-MSE: " + str(results[0]) + '\n')
    print("FINAL TRAINING-MSE: " + str(adaline.evaluate_MSE(x_matrix, d_matrix)) + '\n')
    print("FINAL TEST-MSE: " + str(adaline.evaluate_MSE(input_test_matrix, output_test_matrix)) + '\n')

    print("FINAL WEIGHTS: " + '\n' + str(adaline.weights) + '\n')

    # Escribe en un .txt los resultados (outputs) de la red para el conjunto test (de entrada)
    adaline.write_outputs(input_test_matrix, output_test_matrix, MAX_OUTPUT, MIN_OUTPUT)

    print("Elapsed time: " + str(time.time() - start_time) + " seconds")

    # Grafica la evolución del MSE de entrenamiento y validación a lo largo de los ciclos de aprendizaje
    plt.xlabel("LEARNING CYCLE")
    plt.ylabel("MSE")
    plt.plot(results[1][0], results[1][1], label="Training-MSE")
    plt.plot(results[2][0], results[2][1], label="Validation-MSE")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    main()
