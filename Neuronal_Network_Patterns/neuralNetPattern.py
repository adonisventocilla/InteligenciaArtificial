"""
Importamos los metodos de la libreria Numpy a utilizar
exp: Calcular la exponencialidad de los elementos de la matriz de entrada.
dot: Método para realizar producto de matrices
random: Genera numeros aleatorios
array: Para el manejo de matrices
"""
from numpy import exp, dot, random, array
import math

class neural_network_patterns():
    def __init__(self):
        random.seed(1)
        #Definimos los pesos iniciales
        #Red neuronal con 3 entradas y 1 salida
        self.weights = 2*random.random((3,1))-1
    
    #La funcion sigmoide varia entre 0 y 1
    def sigmoide(self,x):
        return 1/(1+math.e ** -x)
    
    def train(self,inputs,outputs,num):
        for i in range(num):    #Iteracion
            output = self.prediction(inputs)    #salida
            error = outputs- output     #Calculo de error
            #Ajustamos los pesos
            #Formula = inputs(Transpuesta) * error * output * (1- output)
            ajuste = dot(inputs.T, error*output*(1-output))
            self.weights += ajuste

    def prediction(self, inputs):
        #Aplicamos la funcion sigmoide
        resultado = self.sigmoide(dot(inputs, self.weights))
        return resultado