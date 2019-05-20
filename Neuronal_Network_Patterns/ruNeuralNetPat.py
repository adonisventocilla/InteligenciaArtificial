from numpy import array
from neuralNetPattern import neural_network_patterns


#Instanciamos nuestra clase
red = neural_network_patterns()

#Datos de entrenamiento
inputs = array([[1,1,1],[1,0,1],[0,1,1]])
#Salida de los datos de entrenamiento
outputs = array([[1,1,0]]).T

#Entrenamiento de la red neuronal
# inputs = datos de entrada
# outputs = salida de cada entrada
# 10000 = cuantas veces se repetira el entrenamiento de la red
red.train(inputs, outputs, 100)
# Prueba 1
pre_1 = red.prediction(array([1,1,1]))[0]

# Prueba 2
pre_2 = red.prediction(array([0,1,0]))[0]

# round nos permitira redondear el numero
# y saber con más precisión la predicción
print("[1,1,1] = %s" % round(pre_1))
print("[0,1,0] = %s" % round(pre_2))