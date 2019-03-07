# import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
# from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

class DeepNet(object):
    ''' Define a deep forward neural network '''

    def describe(self): return self.__class__.__name__
    def __init__(self, name = 'deepnet', single_input = False, layer_number = 1, node_number = 20, hidden_activation = 'relu', output_activation = 'sigmoid'):
        self.name = name
        self.single_input = single_input
        self.layer_number = layer_number
        self.node_number = node_number
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def build(self, input_dimension = None, base_directory = './', lr = 0.02, momentum = 0.8, plot = True):
        self.input_dimension = 1 if self.single_input else input_dimension
        self.dnn = Sequential(name = self.name)

        # Input layer
        self.dnn.add(Dense(10, activation = self.hidden_activation, input_dim = self.input_dimension, name = self.name + '_input'))
        # Hidden layer
        for i in range(self.layer_number):
            self.dnn.add(Dense(10, activation = self.hidden_activation, name = self.name + '_layer' + str(i+2)))
        # Output layer
        self.dnn.add(Dense(1, activation = self.output_activation, name = self.name + '_output'))
        sgd = SGD(lr=lr, momentum=momentum)
        self.dnn.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            plot_model(self.dnn, to_file = self.output_path + self.name + '.png')
            self.dnn.summary()

    def make_trainable(self, flag):
        self.dnn.trainable = flag
        self.dnn.compile

class AdvNet(object):
    ''' Define the adversarial neural network which is just a generator followed by a discriminator.
    Notice that the weights of the discriminator have been frozen.
    [generator]: separation signal from background
    [discriminator]: predict nuisance parameter
    '''

    def describe(self): return self.__class__.__name__
    def __init__(self, name = 'advnet', generator = None, discriminator = None):
        self.name = name
        self.Gen = generator
        self.Dis = discriminator
        assert self.Dis.single_input

    def build(self, input_dimension = None, base_directory = './', lr = 0.02, momentum = 0.8, plot = True):
        ''' Freeze the weight of the discriminator '''
        # self.Dis.trainable = False

        self.adv = Sequential()
        self.adv.add(self.Gen.dnn)
        self.adv.add(self.Dis.dnn)

        sgd = SGD(lr=lr, momentum=momentum)
        self.adv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            plot_model(self.adv, to_file = self.output_path + self.name + '.png')
            self.adv.summary()
