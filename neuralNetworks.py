import numpy as np

class NeuralNetworks(object):
   def __init__(self,hiddenLayerSize=None,inputLayerSize=None,outputLayerSize=None):
      if hiddenLayerSize is None and inputLayerSize is None and outputLayerSize is None:
         self.hiddenLayerSize,self.inputLayerSize,self.outputLayerSize=3,2,1
         
      else:   
         self.hiddenLayerSize=hiddenLayerSize
         self.inputLayerSize= inputLayerSize
         self.outputLayerSize=outputLayerSize
      self.hiddenLayerWeights=np.random.uniform(size=(self.inputLayerSize,self.hiddenLayerSize))
      self.outputLayerWeights=np.random.uniform(size=(self.hiddenLayerSize,self.outputLayerSize))
   def printWeights(self):
      print(self.hiddenLayerWeights,self.outputLayerWeights)
   def sigmoid(self,z):
      return 1/(1+np.exp(-z))
   def derivativeSigmoid(self,z):
      return z*(1-z)	
   def forwardPropagation(self,inputMatrix):
      hiddenLayerInput=np.dot(inputMatrix,self.hiddenLayerWeights)
      hiddenLayerActivation=self.sigmoid(hiddenLayerInput)
      outputLayerFeed=np.dot(hiddenLayerActivation,self.outputLayerWeights)
      output=self.sigmoid(outputLayerFeed)
      return output
   def backPropagation()
neuralNetwork=NeuralNetworks()
x=np.array([[3,5],[5,1],[10,2]])
print (neuralNetwork.forwardPropagation(x))    
        	
		
 
