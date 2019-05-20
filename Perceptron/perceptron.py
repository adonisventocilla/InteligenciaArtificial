'''
 Perceptron
 w = w + N * (d(k) - y) * x(k)

 p1 = -1
 p2 = 1
'''

import random

class Perceptron:
    def __init__(self, sample, exit, learn_rate=0.01, epoch_number=1000, bias=-1):
        self.sample = sample
        self.exit = exit
        self.learn_rate = learn_rate
        self.epoch_number = epoch_number
        self.bias = bias
        self.number_sample = len(sample)
        self.col_sample = len(sample[0])
        self.weight = []

    def trannig(self):
        for sample in self.sample:
            sample.insert(0, self.bias)

        for i in range(self.col_sample):
           self.weight.append(random.random())

        self.weight.insert(0, self.bias)

        epoch_count = 0

        while True:
            erro = False
            for i in range(self.number_sample):
                u = 0
                for j in range(self.col_sample + 1):
                    u = u + self.weight[j] * self.sample[i][j]
                y = self.sign(u)
                if y != self.exit[i]:

                    for j in range(self.col_sample + 1):

                        self.weight[j] = self.weight[j] + self.learn_rate * (self.exit[i] - y) * self.sample[i][j]
                    erro = True
            epoch_count = epoch_count + 1
            if erro == False:
                break

    def sort(self, sample):
        sample.insert(0, self.bias)
        u = 0
        for i in range(self.col_sample + 1):
            u = u + self.weight[i] * sample[i]

        y = self.sign(u)

        if  y == -1:
            print('Clasificación: P1 (-1)')
        else:
            print('Clasificación: P2 (1)')

    def sign(self, u):
        return 1 if u >= 0 else -1