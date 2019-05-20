from perceptron import Perceptron
# Datos de entrenamiento
samples = [
    [0, 2],
    [-2, 2],
    [0, -2],
    [2, 0],
    [-2,2],
    [-2,-2],
    [2,-2],
    [2,2],
]

# Clasificación de los datos de entrenamiento (salidas que esperamos para cada conjunto de dato)
"""
[0,2] = 1
[-2,-2] = 1
[0,-2] = 0
...
"""
exit = [1, 1, -1, -1, 1, 1, -1, 1]

network = Perceptron(sample=samples, exit = exit, learn_rate=0.01, epoch_number=1000, bias=-1)

network.trannig()

while True:
    sample = []
    for i in range(2):
        sample.insert(i, float(input('Valor: ')))
    network.sort(sample)
    print("\n")