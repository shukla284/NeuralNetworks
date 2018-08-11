import numpy as np

class Perceptron(object):
    def __init__(self,inputNeurons,learningRate=None,inputLayerWeights=None):
        if inputLayerWeights is None:
            self.inputLayerWeights=np.random.random((inputNeurons)) * 2 - 1
        else:
            self.inputLayerWeights=inputLayerWeights
        if learningRate is None:
            self.learningRate=0.1
        else:
            self.learningRate=learningRate

    def unitStepFunction(self,arg):
        return 1 if arg>0.5 else 0

    def __call__(self,inputLayer):
        outputFeed=self.inputLayerWeights*inputLayer
        outputArgument=outputFeed.sum()
        return self.unitStepFunction(outputArgument)

    def adjustWeights(self,targetResult,calculatedResult,inputLayer):
        error=targetResult-calculatedResult
        for neuron in range(len(inputLayer)):
            delta=error*inputLayer[neuron]*self.learningRate
            self.inputLayerWeights[neuron]=delta
def aboveLine(point,linearFunction):
    x,y=point
    return 1 if linearFunction(x)<y else 0
def lineFunction(x):
    return 2*x+3
points=np.random.randint(1,100,(100,2))
perceptron=Perceptron(2)
for point in points:
    perceptron.adjustWeights(aboveLine(point,lineFunction),perceptron(point),point)
    
#training completed with single layer perceptron
correct,wrong=0,0
for point in points:
    if perceptron(point)==aboveLine(point,lineFunction):
        correct+=1
    else:
        wrong+=1
print("Correctness"+str(correct)+" Wrongness "+str(wrong))  

