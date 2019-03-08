# import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
# from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

class DeepNet(object):
    ''' Define a deep forward neural network '''

    def describe(self): return self.__class__.__name__
    def __init__(self, name = 'deepnet', single_input = False, hidden_Nlayer = 1, hidden_Nnode = 20, hidden_activation = 'relu', output_activation = 'sigmoid'):
        self.name = name
        self.single_input = single_input
        self.hidden_Nlayer = hidden_Nlayer
        assert (self.hidden_Nlayer > 0)
        self.hidden_Nnode = hidden_Nnode
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def build(self, input_dimension = None, base_directory = './', lr = 0.02, momentum = 0.8, plot = True):
        self.input_dimension = 1 if self.single_input else input_dimension

        # Input layer
        self.inputLayer = Input(shape=(self.input_dimension,), name = self.name + '_layer0_input')
        # Hidden layer
        self.outputLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_layer1')(self.inputLayer)
        for i in range(self.hidden_Nlayer - 1):
            self.outputLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_layer' + str(i+2))(self.outputLayers)
        # Output layer
        self.outputLayers = Dense(1, activation = self.output_activation, name = self.name + '_output')(self.outputLayers)
        self.dnn = Model(inputs=[self.inputLayer], outputs=[self.outputLayers], name = self.name)

        sgd = SGD(lr = lr, momentum = momentum)
        self.dnn.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            plot_model(self.dnn, to_file = self.output_path + self.name + '.png')
            print('zhang aaa', len(self.dnn._layers), self.single_input, self.input_dimension)
            self.dnn.summary()


    def make_trainable(self, flag):
        self.dnn.trainable = flag
        self.dnn.compile

class AdvNet(object):
    ''' Define the adversarial neural network part which is just a generator followed by a discriminator.
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

        self.adv = self.Gen.dnn
        self.adv.name = 'Adversarial'
        self.adv.add(self.Dis.dnn)

        sgd = SGD(lr = lr, momentum = momentum)
        self.adv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            plot_model(self.adv, to_file = self.output_path + self.name + '.png')
            print('zhang bbb', len(self.adv._layers))
            self.adv.summary()


class CompNet(object):
    ''' Define the composed neural network of generator and adversarial.
    [generator]: separation signal from background
    [adversarial]: contains generator and discriminator
    '''

    def describe(self): return self.__class__.__name__
    def __init__(self, name = 'compnet', generator = None, adversary = None):
        self.name = name
        self.Gen = generator
        self.Adv = adversary
        assert (self.Gen.dnn.inputs is self.Adv.adv.inputs)

    def build(self, base_directory = './', lam = 10, lr = 0.02, momentum = 0.8, plot = True):
        ''' Freeze the weight of the discriminator '''

        self.com = Model(inputs = self.Adv.adv.inputs, outputs = [self.Gen.dnn.outputs[0], self.Adv.adv.outputs[0]])
        print('zhang ccc', len(self.com._layers), )
        self.com.summary()

        def loss(c):
            def _loss(z_true, z_pred):
                return c * K.binary_crossentropy(z_true, z_pred)
            return _loss

        sgd = SGD(lr = lr, momentum = momentum)
        self.com.compile(loss=[loss(c=1.0), loss(c=-lam)], optimizer = sgd, metrics=['accuracy'])

        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            plot_model(self.com, to_file = self.output_path + self.name + '.png')
            print('zhang ddd', len(self.com._layers), )
            self.com.summary()
