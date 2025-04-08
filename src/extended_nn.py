"""
Improved Basic Neural Networks

This script implements a feedforward neural network using stochastic gradient descent (SGD).
It is adapted from `network2.py` by Michael Nielsen (MIT License), originally created for educational purposes as part of
his open-source book: http://neuralnetworksanddeeplearning.com/

Compared to the original version, this implementation is extended and modularized to better support real-world experimentation.

Core features:
--------------
- Fully customizable feedforward neural network with sigmoid activation
- Cost functions: Cross-entropy (default) and quadratic loss
- Weight initialization (scaled Gaussian), plus option for standard initialization
- L1 and L2 regularization support with configurable lambda (λ)
- SGD with mini-batch updates and optional regularization
- Evaluation on training, validation, and test datasets
- Early stopping based on evaluation loss
- Learning rate scheduling: automatic decay on plateau
- Model saving and loading using JSON format

Additional Utilities:
---------------------
This codebase is designed to work with `common_utils.py`, which provides:
- Logging setup with timestamped logs
- Training/evaluation history plotting (cost and accuracy)
- CSV export of results
- Comparison plots across multiple tuning runs
"""

### Libraries
# Standard library
import random
import sys
import json
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# Plateau Learning Rate Scheduler
class PlateauLearningRateScheduler:
    def __init__(self, initial_eta, patience=10, decay=0.5, min_eta_ratio=1/128):
        self.initial_eta = initial_eta
        self.patience = patience
        self.decay = decay
        self.min_eta = initial_eta * min_eta_ratio
        self.best_cost = float('inf')
        self.epochs_without_improvement = 0
        self.eta = initial_eta

    def step(self, current_cost):
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.eta *= self.decay
            self.epochs_without_improvement = 0
            print(f"[Scheduler] Halved learning rate: {self.eta}")

        if self.eta <= self.min_eta:
            print(f"[Scheduler] Reached min learning rate: {self.eta}. Stopping.")
            return True  # Stop training
        return False  # Keep training


class QuadraticCost(object):
    # A class for the quadratic cost function
    # Assumes that the output ``a`` is the result of the sigmoid function
    # Assumes that the desired output ``y`` is a 1-of-K vector

    @staticmethod
    def fn(a, y):
        # return the cost associated with an output ``a`` and desired output ``y``
        y_vec = one_hot_vector(y)
        return 0.5*np.linalg.norm(a-y_vec)**2

    @staticmethod
    def delta(z, a, y):
        # return the error delta from the output layer
        y_vec = one_hot_vector(y)
        return (a-y_vec) * sigmoid_prime(z)

class CrossEntropyCost(object):
    # A class for the cross-entropy cost function
    # Assumes that the output ``a`` is the result of the sigmoid function
    # Assumes that the desired output ``y`` is a 1-of-K vector
 
    @staticmethod
    def fn(a, y):
        # return the cost associated with an output ``a`` and desired output ``y``
        # nan_to_num is used to ensure numerical stability, in particular, if both ``a`` and ``y`` have a 1.0 in the same slot, then the expression (1-y)*np.log(1-a) returns nan. The np.nan_to_num ensures that that is converted to the correct value (0.0).
        y_vec = one_hot_vector(y)
        return np.sum(np.nan_to_num(-y_vec*np.log(a)-(1-y_vec)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        # return the error delta from the output layer
        # z is not used in the method, but it is included to make the interface consistent with the delta method for other cost classes
        y_vec = one_hot_vector(y)
        return (a-y_vec)
    

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        # added cost function argument
        self.num_layers = len(sizes)    
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost
    
    def default_weight_initializer(self):
        # Initialize biases to Gaussian with mean 0 and standard deviation 1
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]  
        # Initialize weights to Gaussian with mean 0 and standard deviation 1 over the square root of the number of weights connecting to the same neuron
        # x stands for the number of neurons in the previous layer
        # y stands for the number of neurons in the current layer
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    def basic_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda=0.0, 
            regularization_type='None',  # added regularization type argument
            evaluation_data=None, 
            test_data=None,
            monitor_evaluation_cost=False, 
            monitor_evaluation_accuracy=False, 
            monitor_training_cost=False, 
            monitor_training_accuracy=False,
            early_stopping_n=None,
            use_lr_schedule=False,
            verbose=True):
        
        if use_lr_schedule:
            scheduler = PlateauLearningRateScheduler(eta)

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []   
        training_cost, training_accuracy = [], []
        learning_rate_history = []
        # for early stopping
        best_cost = float('inf') # initialize the best cost to infinity
        epochs_without_improvement = 0 # initialize the counter for epochs without improvement

        for j in range(epochs):

            if verbose:
                print(f"Epoch {j} training started...\n")

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n, regularization_type)
            # Print the training complete message for each epoch
            if verbose:
                print(f"Epoch {j} training complete")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data) 
                training_accuracy.append(accuracy)
                if verbose:
                    print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)  
                if verbose:
                    print("Cost on training data: {0}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                if verbose:
                    print(f"Accuracy on evaluation data: {accuracy} / {n_data}")
            
            # check if evaluation is needed
            evaluation_needed = evaluation_data and any([monitor_evaluation_cost, early_stopping_n, use_lr_schedule])
            if evaluation_needed:
                cost = self.total_cost(evaluation_data, lmbda)

            if monitor_evaluation_cost:
                evaluation_cost.append(cost)
                if verbose:
                    print(f"Cost on evaluation data: {cost}")
        
            # === Early Stopping ===
            if early_stopping_n:
                if cost < best_cost:
                    best_cost = cost
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_n:
                    print(f"Early stopping: No improvement for {early_stopping_n} consecutive epochs.")
                    break

            # === Learning Rate Scheduler ===
            if use_lr_schedule:
                stop = scheduler.step(cost)
                eta = scheduler.eta
                if stop:
                    break

            learning_rate_history.append(eta)
            # print a newline to separate the epochs
            if verbose:
                print()

        test_cost = None
        test_accuracy = None
        if test_data:
            test_cost, test_accuracy = self.model_test(test_data)
            if verbose:
                print("Final model evaluation on test data:")
                print(f"Test cost: {test_cost}")
                print(f"Test accuracy: {test_accuracy} / {len(test_data)}")

        # Return all tracked metrics
        return {
            "evaluation_cost": evaluation_cost,
            "evaluation_accuracy": evaluation_accuracy,
            "training_cost": training_cost,
            "training_accuracy": training_accuracy,
            "test_cost": test_cost,
            "test_accuracy": test_accuracy,
            "learning_rate_history": learning_rate_history
            }

    def update_mini_batch(self, mini_batch, eta, lmbda, n, regularization_type):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weights with l2 regularization 
        if regularization_type == 'l2':
            self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
        # update weights with l1 regularization 
        elif regularization_type == 'l1':
            self.weights = [np.sign(w) * np.maximum(np.abs(w) - (eta * lmbda / len(mini_batch)), 0)
                            for w in self.weights]
        else:  # no regularization
            self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    
    def backprop(self, x, y):
        # a backprop function that returns the gradient of the cost function
        # initialize the gradient of the biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x # the input activation
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
             z = np.dot(w, activation) + b
             zs.append(z)
             activation = sigmoid(z)
             activations.append(activation)
        # backward pass
        # output layer
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data):
        
        results = [(np.argmax(self.feedforward(x)), y)
                   for (x, y) in data]
        
        return(sum(int(x == y) for (x, y) in results))
    
    def total_cost(self, data, lmbda, regularization_type = 'None'):
        cost = 0.0  
        for x, y in data:     
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        # add regularization term
        if regularization_type == 'l2':
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        elif regularization_type == 'l1':
            cost += (lmbda/len(data))*sum(np.linalg.norm(w, 1) for w in self.weights)
        return cost
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def model_test(self, test_data):
        """Evaluate the model on the test data."""
        test_cost = self.total_cost(test_data, lmbda=0.0, regularization_type='None')  # No regularization for test
        test_accuracy = self.accuracy(test_data)
        print(f"Test cost: {test_cost}")
        print(f"Test accuracy: {test_accuracy} / {len(test_data)}")
        return test_cost, test_accuracy

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def one_hot_vector(j):
    # Return a 10-dimensional unit vector with a 1.0 in the j'th position and zeroes elsewhere. 
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
