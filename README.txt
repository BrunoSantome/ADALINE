1- Para probar la ejecución del ADALINE se necesita abrir un terminal en el directorio actual.

2- Introduce el siguiente comando: "python MAIN.py"

3- La ejecución comenzará preguntando cuál de los 7 modelos de datos disponibles se quiere usar.

4- El programa dará la opción de cargar el modelo (umbral y pesos) óptimo para poder probarlo con el conjunto de test seleccionado anteriormente.

5- Si no se escoge el modelo óptimo, el ADALINE se inicializará de forma randomizada.

6- El programa nos da la opción de probar el modelo actual (el random o el óptimo) para ver el MSE resultante (con test).

7- El programa nos da la opción de entrenar con el modelo de datos actual y obtener el MSE resultante (con test).

8- Si escogemos la opción de entrenar, tendremos que introducir la tasa de aprendizaje que deseamos utilizar.

NOTAS:

- Cada ejecución escribe un fichero output.txt donde se recogen las salidas del ADALINE y las salidas deseadas
- Estos ficheros .txt se sobreescriben por la última ejecución
- Cada ejecución de ENTRENAMIENTO devolverá por pantalla los MSE de entrenamiento, validación y test correspondientes. Además, mostrará una gráfica con la evolución del MSE de entrenamiento y de validación durante el transcurso de los diversos ciclos de aprendizaje.