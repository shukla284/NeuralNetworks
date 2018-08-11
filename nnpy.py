import numpy as np

#sigmoid and derivative
def sigmoid(arg):
    return 1/(1+np.exp(-arg))
def derivativeSigmoid(arg):
    return arg*(1-arg)


inputLayer=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
outputLayer=np.array([[1],[1],[0]])

#number of rounds of fwd and back prop
epoch=5000
alpha=0.1
inputLayerNeurons=inputLayer.shape[1]
hiddenLayerNeurons=3
outputLayerNeurons=1

#initialization of weights
hiddenLayerWeights=np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hiddenLayerBias=np.random.uniform(size=(1,hiddenLayerNeurons))
outputLayerWeights=np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
outputLayerBias=np.random.uniform(size=(1,outputLayerNeurons))

for iteration in range(epoch):
#forward propagation
   hiddenLayerInput=np.dot(inputLayer,hiddenLayerWeights)+hiddenLayerBias
   hiddenLayerActivation=sigmoid(hiddenLayerInput)
   outputLayerInput=np.dot(hiddenLayerActivation,outputLayerWeights)+outputLayerBias
   output=sigmoid(outputLayerInput)
#backpropagation
   outputLayerError=outputLayer-output
   outputLayerGradient=derivativeSigmoid(output)
   hiddenLayerGradient=derivativeSigmoid(hiddenLayerActivation)
   deltaOutputLayer=outputLayerError*outputLayerGradient
   hiddenLayerError=np.dot(deltaOutputLayer,outputLayerWeights.T)
   deltaHiddenLayer=hiddenLayerError*hiddenLayerGradient

#updation of the weights
   outputLayerWeights+=np.dot(hiddenLayerActivation.T,deltaOutputLayer)*alpha
   outputLayerBias+=np.sum(deltaOutputLayer,axis=0,keepdims=True)*alpha
   hiddenLayerWeights+=np.dot(inputLayer.T,deltaHiddenLayer)*alpha
   hiddenLayerBias+=np.sum(deltaHiddenLayer,axis=0,keepdims=True)*alpha

   print(output)

















   


