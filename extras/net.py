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
    def __init__(self, name = 'deepnet', build_dis = False, hidden_Nlayer = 1, hidden_Nnode = 20, hidden_activation = 'relu', output_activation = 'sigmoid'):
        self.name = name
        self.build_dis = build_dis
        self.hidden_Nlayer = hidden_Nlayer
        assert (self.hidden_Nlayer > 0)
        self.hidden_Nnode = hidden_Nnode
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    @staticmethod
    def make_trainable(network, flag):
        network.trainable = flag
        network.compile
        for layer in network.layers:
            layer.trainable = flag

    def build(self, input_dimension = None, base_directory = './', lr = 0.02, momentum = 0.8, plot = True):
        self.input_dimension = input_dimension

        # Input layer
        self.input_GLayer = Input(shape=(self.input_dimension,), name = self.name + '_layer0')
        # Hidden layer
        self.output_GLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_layer1')(self.input_GLayer)
        for i in range(self.hidden_Nlayer - 1):
            self.output_GLayers = Dense(self.hidden_Nnode, activation = self.hidden_activation, name = self.name + '_layer' + str(i+2))(self.output_GLayers)
        # Output layer
        self.output_GLayers = Dense(1, activation = self.output_activation, name = self.name + '_output')(self.output_GLayers)
        # Define model with above layers
        self.generator = Model(inputs=[self.input_GLayer], outputs=[self.output_GLayers], name = self.name)

        sgd = SGD(lr = lr, momentum = momentum)
        self.generator.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])

        if self.build_dis:
            ''' If it is a build_dis, i.e. to predict NPs '''
            self.output_DLayers = self.output_GLayers
            self.output_DLayers = Dense(100, activation="relu", name='Net_r_layer1')(self.output_DLayers)
            self.output_DLayers = Dense(1, activation="sigmoid", name='Net_r_output')(self.output_DLayers)
            self.discriminator = Model(inputs=[self.input_GLayer], outputs=[self.output_DLayers], name='Net_r_model')
            self.discriminator.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])

            def loss(c):
                def _loss(z_true, z_pred):
                    return c * K.binary_crossentropy(z_true, z_pred)
                return _loss

            self.adversary = Model(inputs=[self.input_GLayer], outputs=[self.generator(self.input_GLayer), self.discriminator(self.input_GLayer)])

            self.make_trainable(self.discriminator, False)
            self.make_trainable(self.discriminator, True)
            lam = 10
            self.adversary.compile(loss = [loss(c = 1.0), loss(c = -lam)], optimizer = sgd, metrics = ['accuracy'])

        if plot:
            import os
            self.output_path = '/'.join([base_directory, self.describe()]) + '/'
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            print('\n\n==> \033[92mNetwork 1\033[0m')
            self.generator.summary()
            plot_model(self.generator, to_file = self.output_path + self.name + '_generator.png')
            if self.build_dis:
                print('\n\n==> \033[92mNetwork 2\033[0m')
                self.discriminator.summary()
                print('\n\n==> \033[92mCombined network\033[0m')
                self.adversary.summary()
                plot_model(self.discriminator, to_file = self.output_path + self.name + '_discriminator.png')
                plot_model(self.adversary, to_file = self.output_path + self.name + '_adversary.png')