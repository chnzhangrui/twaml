import os
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.losses import mean_squared_error, mean_absolute_error
# from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.layers.normalization import BatchNormalization


class DeepNet(object):
    ''' Define a deep forward neural network '''

    def describe(self): return self.__class__.__name__
    def __init__(self, name, problem, build_dis, hidden_Nlayer, hidden_Nnode, dropout_rate, hidden_activation, output_activation = 'sigmoid'):
        self.name = name
        self.problem = problem
        self.build_dis = build_dis
        self.hidden_Nlayer = hidden_Nlayer
        assert (self.hidden_Nlayer > 0)
        self.hidden_Nnode = hidden_Nnode
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

    @staticmethod
    def make_trainable(network, flag):
        network.trainable = flag
        for layer in network.layers:
            layer.trainable = flag
        network.compile

    def build(self, input_dimension, lr, momentum):
        self.input_dimension = input_dimension

        # Input layer
        self.input_GLayer = Input(shape=(self.input_dimension,), name = self.name + '_layer0')
        # Hidden layer
        self.output_GLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_layer1')(self.input_GLayer)
        for i in range(self.hidden_Nlayer - 1):
            self.output_GLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_layer' + str(i+2))(self.output_GLayers)
            if self.dropout_rate != 0:
                self.output_GLayers = Dropout(self.dropout_rate)(self.output_GLayers)
        # Output layer
        self.output_GLayers = Dense(1, activation = self.output_activation, name = self.name + '_output')(self.output_GLayers)
        # Define model with above layers
        self.generator = Model(inputs=[self.input_GLayer], outputs=[self.output_GLayers], name = self.name)

        if self.problem == 0:
            sgd = SGD(lr = lr, momentum = momentum)
            self.generator.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
        else:
            self.generator.compile(loss = 'mean_squared_error', optimizer = Adam(), metrics=['mse'])

    def plot(self, base_directory = './'):
        self.output_path = '/'.join([base_directory, self.describe()]) + '/'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        print('\n\n==> \033[92mNetwork' + (' 1' if self.build_dis else '') + '\033[0m')
        self.generator.summary()
        plot_model(self.generator, to_file = self.output_path + self.name + '_generator.png')
        if self.build_dis:
            print('\n\n==> \033[92mNetwork 2\033[0m')
            self.discriminator.summary()
            print('\n\n==> \033[92mCombined network\033[0m')
            self.adversary.summary()
            plot_model(self.discriminator, to_file = self.output_path + self.name + '_discriminator.png')
            plot_model(self.adversary, to_file = self.output_path + self.name + '_adversary.png')

class AdvNet(DeepNet):
    ''' Define adversarial neural networks '''

    def __init__(self, hidden_auxNlayer, hidden_auxNnode, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.hidden_auxNlayer = hidden_auxNlayer
        self.hidden_auxNnode = hidden_auxNnode

    def build(self, input_dimension, lam, lr, momentum, alr, amomentum):
        self.input_dimension = input_dimension

        # Input layer
        self.input_GLayer = Input(shape=(self.input_dimension,), name = self.name + '_layer0')
        # Hidden layer
        self.output_GLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_Gen_l1')(self.input_GLayer)
        for i in range(self.hidden_Nlayer - 1):
            self.output_GLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_Gen_l' + str(i+2))(self.output_GLayers)
            if self.dropout_rate != 0:
                self.output_GLayers = Dropout(self.dropout_rate)(self.output_GLayers)
        # Output layer
        self.output_GLayers = Dense(1, activation = self.output_activation, name = self.name + '_Gen_output')(self.output_GLayers)
        # Define model with above layers
        self.generator = Model(inputs=[self.input_GLayer], outputs=[self.output_GLayers], name = self.name + '_Gen')

        def binary_loss(c):
            def _loss(z_true, z_pred):
                return c * K.binary_crossentropy(z_true, z_pred)
            return _loss

        def mse_loss(c):
            def _loss(z_true, z_pred):
                return c * mean_squared_error(z_true, z_pred)
            return _loss

        def mae_loss(c):
            def _loss(z_true, z_pred):
                return c * mean_absolute_error(z_true, z_pred)
            return _loss

        sgd = SGD(lr = lr, momentum = momentum)
        asgd = SGD(lr = alr, momentum = amomentum)
        self.generator.compile(loss = binary_loss(c = 1.0), optimizer = sgd, metrics=['accuracy'])


        ''' Predict NPs '''
        self.output_DLayers = self.output_GLayers
        for i in range(self.hidden_auxNlayer - 1):
            self.output_DLayers = Dense(self.hidden_auxNnode, activation = self.hidden_activation, name = self.name + '_Dis_l' + str(i+1))(self.output_DLayers)
            self.output_DLayers = BatchNormalization()(self.output_DLayers)
        self.output_DLayers = Dense(1, activation = self.output_activation, name = self.name + '_Dis_output')(self.output_DLayers)
        self.output_DLayers = BatchNormalization()(self.output_DLayers)
        self.discriminator = Model(inputs=[self.input_GLayer], outputs=[self.output_DLayers], name = self.name + '_Dis')

        self.adversary = Model(inputs=[self.input_GLayer], outputs=[self.generator(self.input_GLayer), self.discriminator(self.input_GLayer)])

        self.make_trainable(self.discriminator, False)
        self.make_trainable(self.generator, True)
        self.lam = lam
        if self.problem == 0:
            self.adversary.compile(loss = [binary_loss(c = 1.0), binary_loss(c = -lam)], optimizer = sgd, metrics = ['accuracy'])
        elif self.problem == 1:
            self.adversary.compile(loss = [binary_loss(c = 1.0), mse_loss(c = -lam)], optimizer = sgd, metrics = ['accuracy', 'mse'])
        else:
            self.adversary.compile(loss = [binary_loss(c = 1.0), mae_loss(c = -lam)], optimizer = sgd, metrics = ['accuracy', 'mae'])

        self.make_trainable(self.discriminator, True)
        self.make_trainable(self.generator, False)
        if self.problem == 0:
            self.discriminator.compile(loss = binary_loss(c = 1.0), optimizer = sgd, metrics=['accuracy'])
        elif self.problem == 1:
            self.discriminator.compile(loss = mse_loss(c = lam), optimizer = asgd, metrics=['mse', 'mae'])
        else:
            self.discriminator.compile(loss = mae_loss(c = lam), optimizer = Adam(lr=alr), metrics=['mae', 'mse'])
