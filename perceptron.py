import numpy as np

class Perceptron(object):
    def __init__(self,inputNeurons,inputLayerWeights=None):
        if inputLayerWeights is None:
            self.inputLayerWeights=np.random.uniform(inputNeurons)
        else:
            self.inputLayerWeights=inputLayerWeights
    def unitStepFunction(self,arg):
        return 1 if arg>0.5 else 0
    def __call__(self,inputLayer):
        outputFeed=self.inputLayerWeights*inputLayer
        outputArgument=outputFeed.sum()
        return self.unitStepFunction(outputArgument)
perceptron=Perceptron(2)
for x in [np.array([0, 0]), np.array([0, 1]), 
          np.array([1, 0]), np.array([1, 1])]:
    y = perceptron(np.array(x))
    print(x, y)

