"""
Load different datasets for training and testing, based on the MNIST format.

Adapted from mnist_loader.py by Michael Nielsen (MIT License)
"""

### Libraries
# Standard library
import pickle
import gzip
# Third-party libraries
import numpy as np

### MNIST dataset
def mnist_load_data():
    # Load the MNIST dataset 
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = "latin1")
    f.close()
    return (training_data, validation_data, test_data)

def mnist_load_data_wrapper():
    # Load the MNIST dataset and return a tuple containing ``(training_data, validation_data, test_data)``.
    tr_d, va_d, te_d = mnist_load_data()
    # Process the training data
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_data = list(zip(training_inputs, tr_d[1]))

    # Process the validation data
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # Process the test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


