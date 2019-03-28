import numpy as np
import matplotlib.pyplot as plt

def compute_error(x,y,w):
	z = np.dot(w, x.T)
	sigmoid_ = sigmoid(x,w)
	loglike = np.sum(y*np.log(sigmoid_) + (1-y)*np.log(1-sigmoid_))
	return -1/y.shape[0] * loglike

def sigmoid(x,w):
  z = np.dot(w.T,x.T)
  return 1/(1+np.exp(-z))

#one step in gradient
def step_gradient(X, y, w, learning_rate):
	#gradient descent
	small_sigma = sigmoid(X,w)
	error = small_sigma - y
	gradient = np.dot(error,X)
	# return weight
	new_W = learning_rate*gradient
	return new_W

def gradient_descent(X, y, weights, learning_rate, num_iterations):
	w = weights
	for i in range(num_iterations):
		w -= step_gradient(X,y,w,learning_rate)
	return w


#load training data
x1, x2, y = np.genfromtxt('cl_train_2.csv',delimiter=',',usecols=(0,1,2),unpack=True)
#load testing data
x1_test, x2_test, y_test= np.genfromtxt('cl_test_2.csv',delimiter=',',usecols=(0,1,2),unpack=True)
#set learning rate and num of iteations
learning_rate = .1
num_iterations = 1000

#add inductive bias
ones = np.ones(y.shape[0])
#add extra variables to capture strucure of data
x1_2 = np.power(x1,2)
x2_2 = np.power(x2,2)
X = np.column_stack((ones,x1,x2,x1_2,x2_2))
weights = np.zeros(X.shape[1])

#calulate weights
weights_trained = gradient_descent(X, y, weights, learning_rate,num_iterations)

ones = np.ones(y_test.shape[0])
X_test = np.column_stack(( ones, x1_test, x2_test))

x1_sorted = np.sort(x1)
x2_sorted = np.sort(x2)
x1_F,x2_F = np.meshgrid(x1_sorted,x2_sorted)

#function to plot
F = weights_trained[0]+weights_trained[1]*x1_F + weights_trained[2]*x2_F + weights_trained[3]*x1_F**2 + weights_trained[4]*x2_F**2

#plot training data
for i in range(y_test.shape[0]):
	if y[i] == 1:
		plt.plot(x1[i], x2[i], 'go')
	else:
		plt.plot(x1[i], x2[i],'yo')

plt.contour(
    x1_F, x2_F, F, [0]
)
plt.show()
#plot the testing data
for i in range(y_test.shape[0]):
	if y_test[i] == 1:
		plt.plot(x1_test[i], x2_test[i], 'bo')
	else:
		plt.plot(x1_test[i], x2_test[i],'ro')

plt.contour(
    x1_F, x2_F, F, [0]
)

plt.show()

