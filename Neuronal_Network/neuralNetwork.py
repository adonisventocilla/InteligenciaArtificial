from numpy import array, dot, random

#Declaracion de la clase
class neural_network:
    def __init__(self):
        random.seed(1)
        #Pesos aleatorios: 2 entradas y 1 salida
        self.weights = 2*random.random((2,1))-1
    
    """
    inputs: son la entrada de los datos que ingresa como modelo
    outputs: son las solucion a los datos de entrada
    num: son la cantidad de veces que se va a entrenar
    """
    def train(self,inputs, outputs, num):
        for iteration in range(num):
            output = self.predict(inputs)   #salida de la neurona
            error = outputs - output    #Calculamos el error
            #Formula de ajuste: 0.01 *  Transpuesta de array(entradas: inputs) * error
            ajuste = 0.01*dot(inputs.T, error)
            self.weights += ajuste
    
    def predict(self, inputs):
        #Se calcula la salida de la neurona
        #inputs: la data
        #weights: los pesos aleatorios
        return dot(inputs,self.weights)