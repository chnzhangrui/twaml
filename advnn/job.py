from net import DeepNet, AdvNet
from train import Train
from keras.callbacks import Callback
import os

class Job(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name, problem, nfold, train_fold, epochs, hidden_Nlayer, hidden_Nnode, lr, momentum, output, activation, dropout_rate, para_train={}):
        self.nfold = int(nfold)
        self.problem = int(problem)
        self.train_fold = int(train_fold)
        self.epochs = int(epochs)
        self.hidden_Nlayer = int(hidden_Nlayer)
        self.hidden_Nnode = int(hidden_Nnode)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.name = self.output if name is None else name
        self.output = 'job_{}__l{}n{}_lr{}mom{}_{}_k{}_dp{}_e{}_plb{}'.format(self.name, self.hidden_Nlayer, self.hidden_Nnode, self.lr, self.momentum, self.activation, self.nfold, self.dropout_rate, self.epochs, self.problem) if output is None else output

        self.para_train = para_train
        para_train['base_directory'] = self.output
        print('\033[92m[INFO]\033[0m', '\033[92mJobname \033[0m', self.output)

    def run(self):

        ''' An instance of Train for data handling '''
        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = self.nfold)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self.deepnet = DeepNet(name = self.name, problem = self.problem, build_dis = False, hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode, hidden_activation = self.activation, dropout_rate = self.dropout_rate)
        self.deepnet.build(input_dimension = self.trainer.shape, lr = self.lr, momentum = self.momentum)
        self.deepnet.plot(base_directory = self.output)
        self.trainer.setNetwork(self.deepnet.generator)
        
        ''' Run the training '''
        self.result = self.trainer.train(mode = 0, epochs = self.epochs, fold = self.train_fold)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()

    def saveModel(self, prefix):
        # serialize model to JSON
        model_json = self.trainer.network.to_json()
        with open(prefix + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.trainer.network.save_weights(prefix + '.h5')
        print('Saved', prefix, 'to disk')

class JobAdv(Job):
    def __init__(self, preTrain_epochs, hidden_auxNlayer, hidden_auxNnode, n_iteraction, lam, alr, amomentum, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.preTrain_epochs = int(preTrain_epochs)
        self.hidden_auxNlayer = int(hidden_auxNlayer)
        self.hidden_auxNnode = int(hidden_auxNnode)
        self.n_iteraction = int(n_iteraction)
        self.lam = float(lam)
        self.alr = float(alr)
        self.amomentum = float(amomentum)
        self.output = '{}__E{}_L{}N{}_alr{}mom{}_it{}_Loss{}_lam{}'.format(self.output, self.preTrain_epochs, self.hidden_auxNlayer, self.hidden_auxNnode, self.alr, self.amomentum, self.n_iteraction, self.problem, self.lam)
        self.para_train['base_directory'] = self.output
        print('\033[92m[INFO]\033[0m', '\033[92mJobname \033[0m', self.output)
        
    def run(self):

        ''' An instance of Train for data handling '''
        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = self.nfold)

        ''' An instance of AdvNet for network construction and pass it to Train '''
        self.advnet = AdvNet(name = self.name, problem = self.problem, build_dis = True, hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode,
            hidden_activation = self.activation, hidden_auxNlayer = self.hidden_auxNlayer, hidden_auxNnode = self.hidden_auxNnode, dropout_rate = self.dropout_rate)
        self.advnet.build(input_dimension = self.trainer.shape, lam = self.lam, lr = self.lr, momentum = self.momentum, alr = self.alr, amomentum = self.amomentum)
        self.advnet.plot(base_directory = self.output)

        class Evaluate(Callback):
            def __init__(self, name, trainer, mode):
                self.name = name
                self.trainer = trainer
                self.mode = mode
            def on_epoch_end(self, epoch, logs):
                print('\033[92m[INFO]\033[0m', '\033[92mCheckpoint \033[0m', self.name, epoch, self.mode, logs)
#                self.trainer.plotResults(self.name + str(epoch), mode)
    
        ''' pre-training '''
        if self.preTrain_epochs != 0:
            prefix = 'pre-gen'
            print('\033[92m[INFO]\033[0m', '\033[92mpre-training generator (1st) with epochs\033[0m', self.preTrain_epochs)
            AdvNet.make_trainable(self.advnet.discriminator, False)
            AdvNet.make_trainable(self.advnet.generator, True)
            self.trainer.setNetwork(self.advnet.generator)
            self.result = self.trainer.train(mode = 1, epochs = self.preTrain_epochs, fold = self.train_fold, callbacks=[Evaluate(prefix, self.trainer, 'y')])
            self.trainer.plotLoss(self.result, prefix)
            self.trainer.plotResults(prefix, 'y')

            prefix = 'pre-dis'
            print('\033[92m[INFO]\033[0m', '\033[92mpre-training discriminator (2nd) with epochs\033[0m', self.preTrain_epochs)
            AdvNet.make_trainable(self.advnet.discriminator, True)
            AdvNet.make_trainable(self.advnet.generator, False)
            self.trainer.setNetwork(self.advnet.discriminator)
            self.result = self.trainer.train(mode = 2, epochs = 1, fold = self.train_fold, callbacks=[Evaluate(prefix, self.trainer, 'z')])
            self.trainer.plotLoss(self.result, prefix, True)
            self.trainer.plotResults(prefix, 'z')
        else:
            print('\033[91m[INFO]\033[0m', '\033[91mpre-training skipped!\033[0m')

        self.output_path = '/'.join([self.output, self.describe()]) + '/'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        ''' Iterative training '''
        for i in range(1, self.n_iteraction+1):

            print('\033[92m[INFO] Going to train\033[0m', i, '\033[92miteration, generator (1st) with epochs\033[0m', self.epochs)
            AdvNet.make_trainable(self.advnet.discriminator, False)
            AdvNet.make_trainable(self.advnet.generator, True)
            self.trainer.setNetwork(self.advnet.adversary)
            self.result = self.trainer.train(mode = 3, epochs = self.epochs, fold = self.train_fold)
            self.trainer.plotIteration(i)

            prefix = 'iter-gen' + str(i)
            mode = 'y'
            print('\033[92m[DEBUG] Going to inspect predction by\033[0m', prefix, '\033[92mwith mode\033[0m', mode)
            self.trainer.setNetwork(self.advnet.generator)
            if (i % 5 == 0):
                self.trainer.plotResults(prefix, mode)

            if (i % 5 == 0):
                self.saveModel(self.output_path + self.trainer.name + '_' + str(i))

            print('\033[92m[INFO] Going to train\033[0m', i, '\033[92miteration, discriminator (2nd) with epochs\033[0m', 1)
            AdvNet.make_trainable(self.advnet.discriminator, True)
            AdvNet.make_trainable(self.advnet.generator, False)
            self.trainer.setNetwork(self.advnet.discriminator)
            self.result = self.trainer.train(mode = 2, epochs = 1, fold = self.train_fold)

            prefix = 'iter-dis' + str(i)
            mode = 'z'
            print('\033[92m[INFO] Going to inspect predction by\033[0m', prefix, '\033[92mwith mode\033[0m', mode)
            if (i % 5 == 0) or (i<2):
                self.trainer.plotResults(prefix, mode)

            self.trainer.setNetwork(self.advnet.adversary)
            self.trainer.plotIteration(i+0.5)

        print('\033[92m[INFO]\033[0m', self.n_iteraction, '\033[92mIteration done, storing and plotting results.\033[0m')
        self.trainer.setNetwork(self.advnet.adversary)
        self.trainer.saveLoss()
        self.trainer.setNetwork(self.advnet.generator)
        self.trainer.plotResults('iter-genFinal')

