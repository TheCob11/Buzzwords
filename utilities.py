import numpy as np

class Activators:
    def sigmoid(x):
        return 1/(1+np.exp(-1*x.astype(float)))

    def sigPrime(x):
        return Activators.sigmoid(x)*(1-Activators.sigmoid(x))
    sigmoid.prime = sigPrime

    def reLU(x):
        return x*(x > 0)

    def reLUPrime(x):
        return 1*(x > 0)
    reLU.prime = reLUPrime

class XOR:
    def __init__(self, a=None, b=None):
        self.a = a if a is not None else np.random.randint(2)
        self.b = b if b is not None else np.random.randint(2)
        self.out = self.a^self.b
    def inOutTuple(self):
        return ([self.a,self.b], self.out)
    def generateData(dataSize):
        return np.array([XOR().inOutTuple() for i in range(dataSize)], dtype=object)
    testData = [([0, 0], 0), ([1, 0], 1), ([0, 1], 1), ([1, 1], 0)]
