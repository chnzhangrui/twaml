# import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
# from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

class Net(object):
    ''' Define a simple neural network '''

    def describe(self): return self.__class__.__name__
    def __init__(self, name = 'mynet', layer_number = 1, node_number = 20, hidden_activation = 'relu', output_activation = 'sigmoid'):
        self.name = name
        self.layer_number = layer_number
        self.node_number = node_number
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def build(self, input_dimension = None, base_directory = './', lr = 0.02, momentum = 0.8, plot = True):
        self.input_dimension = input_dimension
        self.model = Sequential()

        # Input layer
        self.model.add(Dense(10, activation = self.hidden_activation, input_dim = self.input_dimension, name = self.name + '_input'))
        # Hidden layer
        for i in range(self.layer_number):
            self.model.add(Dense(10, activation = self.hidden_activation, name = self.name + '_layer' + str(i+2)))
        # Output layer
        self.model.add(Dense(1, activation = self.output_activation, name = self.name + '_output'))
        sgd = SGD(lr=lr, momentum=momentum)
        self.model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            plot_model(self.model, to_file = self.output_path + self.name + '.png')
            self.model.summary()

