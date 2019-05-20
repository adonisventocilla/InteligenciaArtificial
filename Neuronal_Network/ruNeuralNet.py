from numpy import array
from neuralNetwork import neural_network

#Definicion de las entradas
x = array([[2,1],[1,1],[5,2],[10,3]])

#Definicion de los resultados de las entradas :: Salidas
y = array([[6,4,14,26]]).T

#entrenamos nuestra red
"""
AQUI LA RED NEURONAL DEBE DEDUCIR LA FORMULA:
(x+y)*2
A PARTIR DE LA INFORMACION DADA
"""
red = neural_network()
red.train(x,y,4000)

#Dato de prueba
test1 = array([12,1])

#Mostramos la salida de la red
print("El resultado es: %s" %red.predict(test1))
