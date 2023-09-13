import random

import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            a[-1] = self.softmax(a[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, beta = 0.9):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Para almacenar los b_i y w_i al aplicar RMSprop
        self.rmsprop_b = [np.zeros(b.shape) for b in self.biases]
        self.rmsprop_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for t in range(len(self.weights)):
            self.rmsprop_b[t] = beta * self.rmsprop_b[t] + (1-beta) * (nabla_b[t]**2)
            self.rmsprop_w[t] = beta * self.rmsprop_w[t] + (1-beta) * (nabla_w[t]**2)
            self.biases[t] -= (eta / np.sqrt(self.rmsprop_b[t] + 1e-8)) * nabla_b[t]    # epsilon puede ir de 1e-9 a 1e-7
            self.weights[t] -= (eta / np.sqrt(self.rmsprop_w[t] + 1e-8)) * nabla_w[t]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cross_entropy(activations[-1], y) * \
            sigmoid_prime(zs[-1])           # cambiando self.cost_derivative por self.cross_entropy
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    '''
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    '''

    # cambiando la función de costo por categorical cross entropy
    def cross_entropy(self, output_activations, y):
        return -1/len(y) * np.sum(y * np.log(output_activations))
    
    # intento aplicar soft-max
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
