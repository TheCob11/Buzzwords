import numpy as np
from utilities import Activators, XOR

'''
    n
1
    n   1
0
    n

y = correct
a = outLayer
z = weighted sum
p = prevLayer
C=(a-y)^2
dC/da = 2(a-y)
a = sig(z)
da/dz = sig'(z)
z=wp+b
dC/dz = 2(a-y) * sig'(z)
dC/dw = dC/da * da/dz * dz/dw 
dz/dw = p
dz/db = 1
dz/dp = w
dC/dw = dC/dz * p
dC/db = dC/dz
dC/dp = dC/dz * w
'''



class NNet:
    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(('{}:\n{}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def __init__(self, layerSizes: list[int], layerActivators=[Activators.reLU]):
        self.layerSizes = layerSizes
        self.weights = np.array([np.random.rand(currentLayerSize, prevLayerSize)
                                for currentLayerSize, prevLayerSize in zip(layerSizes[1:], layerSizes[:-1])], dtype=object)
        self.biases = np.array([np.random.rand(currentLayerSize, 1)
                               for currentLayerSize in layerSizes[1:]], dtype=object)
        layerActivators.append(Activators.sigmoid)
        if(len(layerActivators)<len(layerSizes)-1):
            midpt = (len(layerActivators)//2)-1
            layerActivators = layerActivators[:midpt]+[layerActivators[midpt]]*(len(layerSizes)-len(layerActivators)-1)+layerActivators[midpt:]
        self.layerActivators = layerActivators

    def feedForward(self, inputLayer):
        current = np.transpose([inputLayer])
        for i in range(len(n.layerSizes)-1):
            current = self.layerActivators[i](np.dot(self.weights[i], current)+self.biases[i])
        return current

    def train(self, data, learnRate, miniBatchSize, epochs):
        for i in range(epochs):
            np.random.shuffle(data)
            miniBatches = np.array_split(data, len(data)/miniBatchSize)
            for b in range(len(miniBatches)):
                print(f"Epoch {i}, batch {b}")
                self.updateMiniBatch(miniBatches[b], learnRate)
        print("ooga")

    def updateMiniBatch(self, miniBatch, learnRate):
        totalNabW = np.zeros_like(self.weights)
        totalNabB = np.zeros_like(self.biases)
        # logOutput = True
        for i in miniBatch:
            currentNabW, currentNabB = self.backProp(i[0], i[1])
            totalNabW += currentNabW*learnRate/len(miniBatch)
            totalNabB += currentNabB*learnRate/len(miniBatch)
            # logOutput = False
        self.weights -= totalNabW
        self.biases -= totalNabB

    def backProp(self, input, expected, logNumbers=False):
        currentA = np.transpose([input])
        activations = [currentA]
        preActivations = []
        for i in range(len(n.layerSizes)-1):
            preAct = np.dot(self.weights[i], currentA)+self.biases[i]
            preActivations.append(preAct)
            # print(preAct)
            # print(preActivations)
            currentA = self.layerActivators[i](preAct)
            activations.append(currentA)
        # print(preActivations)
        # print(activations)
        nablaW = np.zeros_like(self.weights)
        nablaB = np.zeros_like(self.biases)
        dCdz = (activations[-1] - np.transpose([expected]))
        if(logNumbers):
            print(activations[-1])
            print("\n")
            print(expected)
            print("\n")
            print(dCdz)
        nablaW[-1] = np.dot(dCdz, activations[-2].transpose())
        nablaB[-1] = dCdz
        for i in range(2, len(self.layerSizes)):
            dCdz = np.dot(self.weights[-i+1].transpose(),
                          dCdz) * self.layerActivators[-i].prime(preActivations[-i])
            nablaW[-i] = np.dot(dCdz, activations[-i-1].transpose())
            nablaB[-i] = dCdz
        return nablaW, nablaB

    def cost(self, output, expected):
        return -expected*np.log(output)+(1-expected)*np.log(1-output)

    def evaluate(self, testData):
        evaluation = np.empty((len(testData), 3), dtype=object)
        for i in range(len(testData)):
            evaluation[i] = (self.feedForward(testData[i][0]), testData[i][1], self.cost(self.feedForward(testData[i][0]), testData[i][1]))
        return evaluation



if __name__ == "__main__":
    n = NNet([2, 3, 1])
    n.train(XOR.generateData(100), 1, 10, 20)
    e = n.evaluate(XOR.testData)
    print(e)