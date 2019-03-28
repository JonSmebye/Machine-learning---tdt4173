import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, dataIn, y, learning_rate):
        np.random.seed(1)
        self.weight1 = np.random.random([3,2])
        self.weight2 = np.random.random([3,1])
        self.z1=None
        self.z2=None
        self.activation1=None
        self.activation2=None
        self.Y = y
        ones = np.ones(len(dataX))
        self.dataInput = np.column_stack((ones, dataIn))
        self.learning_rate = learning_rate
        self.cost = []

    def sigmoid(self, x): 
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return (x)*(1 - (x))

    def forward(self):
        #compute activation values for hidden layer
        self.z1 = np.matmul(self.dataInput,self.weight1)
        self.activation1 = self.sigmoid(self.z1)
        #Add bias to hidden layer
        ones = np.ones(len(self.activation1))
        self.activation1 = np.column_stack((ones, self.activation1))
        #compute activation values for output layer
        self.z2 = np.matmul(self.activation1, self.weight2)
        self.activation2 = self.sigmoid(self.z2)

    def backProp(self):
        #compute backprop. for last layer
        delta2 = self.activation2 - self.Y
        grad2 = np.matmul(self.activation1.T,delta2)

        #append squared error to list for plot
        squaredError = 0.5*(np.mean(abs(delta2))**2)
        self.cost.append(squaredError)

        #Compute gradients for a^(L-1)*sigmoidDerived(z^(L))*(a^(L-1)-y)
        delta1 = delta2.dot(self.weight2.T)*self.sigmoid_derivative(self.activation1) 
        grad1 = np.matmul(self.dataInput.T, delta1)
        grad1 = np.delete(grad1,0,1)

        self.weight2 -= self.learning_rate*grad2
        self.weight1 -= self.learning_rate*grad1