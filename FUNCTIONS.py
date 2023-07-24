import math

class FUNCTIONS:
    """ CLASE QUE CONTIENE ÚNICAMENTE EL CONJUNTO DE FUNCIONES DE ACTIVACIÓN MÁS COMUNES Y SUS CORRESPONDIENTES DERIVADAS """
    def unit_step(z):
        """ Función escalón (entre 0 y 1) """
        if z < 0:
            return 0
        elif z == 0:
            return 0.5
        elif z > 0:
            return 1

    def sign(z):
        """ Función signo """
        if z < 0:
            return -1
        elif z == 0:
            return 0
        elif z > 0:
            return 1

    def linear(z):
        """ Función lineal """
        return z

    # DERIVATIVE
    def d_linear(z):
        """ Derivada de la función lineal """
        return 1

    def piece_wise_linear(z):
        """ Función para las SVM """
        if z >= 0.5:
            return 1
        elif z > -0.5 and z < 0.5:
            return z + 0.5
        elif z <= -0.5:
            return 0

    # DERIVATIVE
    def d_piece_wise_linear(z):
        """ Derivada de la función para las SVM """
        if z >= 0.5:
            return 0
        elif z > -0.5 and z < 0.5:
            return 1
        elif z <= -0.5:
            return 0

    def sigmoid(z):
        """ Función sigmoide """
        return 1/(1+pow(math.e, -z))

    # DERIVATIVE
    def d_sigmoid(z):
        """ Derivada de la función sigmoide """
        return FUNCTIONS.sigmoid(z)*(1 - FUNCTIONS.sigmoid(z))

    def hyperbolic_tangent(z):
        """ Función tangente hiperbólica """
        return (pow(math.e, z) - pow(math.e, -z))/(pow(math.e, z) + pow(math.e, -z))

    # DERIVATIVE
    def d_hyperbolic_tangent(z):
        """ Derivada de la tangente hiperbólica """
        return 1 - pow(FUNCTIONS.hyperbolic_tangent(z), 2)

    def relu(z):
        """ Función ReLU """
        return max(0, z)

    # DERIVATIVE
    def d_relu(z):
        """ Derivada de la función ReLU """
        if z > 0:
            return 1
        else:
            return 0

    def soft_plus(z):
        """ Función softplus """
        return math.log(1 + pow(math.e, z))

    # DERIVATIVE
    def d_soft_plus(z):
        """ Derivada de la función softplus """
        return FUNCTIONS.sigmoid(z)
