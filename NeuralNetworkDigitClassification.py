import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets, svm, metrics
import pandas as pd
from sklearn.model_selection import train_test_split

class NeuralNetwork:

    def __init__(self, x ,y, learning_rate):
        self.x = x
        self.y = y
        self.neurons_hidden_layers = 200
        self.learning_rate = learning_rate

        self.w1 = np.random.randn(x.shape[1], self.neurons_hidden_layers)
        self.biasL1 = np.ones((1,self.neurons_hidden_layers))
        
        self.w2 = np.random.randn(self.neurons_hidden_layers,self.neurons_hidden_layers)
        self.biasL2 = np.ones((1,self.neurons_hidden_layers))

        self.w3 = np.random.randn(self.neurons_hidden_layers,self.neurons_hidden_layers)
        self.biasL3 = np.ones((1,self.neurons_hidden_layers))

        self.w4 = np.random.randn(self.neurons_hidden_layers, 10)
        self.biasL4 = np.ones((1,10))

        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1 - x)

    def forward(self, x=None):
        return_ = False
        if x is not None:
            return_ = True
            self.x = x
        #layer one input to hidden
        self.z1 = np.dot(self.x, self.w1) + self.biasL1
        self.a1 = self.sigmoid(self.z1)
        #layer two hidden to hidden
        self.z2 = np.dot(self.a1,self.w2) + self.biasL2
        self.a2 = self.sigmoid(self.z2)
        #layer three hidden to hidden
        self.z3 = np.dot(self.a2,self.w3) + self.biasL3
        self.a3 = self.sigmoid(self.z3)
        #layer four hidden to output
        self.z4 = np.dot(self.a3,self.w4) + self.biasL4
        self.a4 = self.sigmoid(self.z4)

        if return_:
            return np.argmax(self.a4)
        
    def backProp(self):
        error = self.a4 - self.y
        error = np.mean(abs(error))
        print(error)
        
        delta4 = (self.a4 -self.y)/(self.y.shape[0]) # normalize the result
        grad4 = np.dot(self.a3.T, delta4)

        error3 = np.dot(delta4, self.w4.T)
        delta3 = error3*self.sigmoid_derivative(self.a3)
        grad3 = np.dot(self.a2.T, delta3)

        error2 = np.dot(delta3, self.w3.T)
        delta2 = error2*self.sigmoid_derivative(self.a2)
        grad2 = np.dot(self.a1.T, delta2)

        error1 = np.dot(delta2, self.w2.T)
        delta1 = error1*self.sigmoid_derivative(self.a1)
        grad1 = np.dot(self.x.T, delta1)

        self.w4 -= self.learning_rate*grad4
        self.w3 -= self.learning_rate*grad3
        self.w2 -= self.learning_rate*grad2
        self.w1 -= self.learning_rate*grad1     

#import data
digits = datasets.load_digits(n_class=10,return_X_y=True)
data,target = digits
target = pd.get_dummies(target)
x_train, x_val, y_train, y_val = train_test_split(data, target, test_size = 0.2)
#normalize x values and make y array
x_train = x_train/16
x_val = x_val/16
y_train = np.array(y_train)
y_val = np.array(y_val)