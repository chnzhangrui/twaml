from net import DeepNet, AdvNet, CompNet
from train import Train

class Job(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name = None, output = None, nfold = 3, train_fold = 0, epochs = 10, layer_number = 4, node_number = 20, lr = 0.02, momentum = 0.8, activation = 'relu'):
        self.output = 'l{}n{}_lr{}_mom{}_{}_f{}_e{}'.format(layer_number, node_number, lr, momentum, activation, nfold, epochs) if output is None else output
        self.name = self.output if name is None else name
        self.nfold = nfold
        # Train the "train_fold"-th fold
        self.train_fold = train_fold
        self.epochs = epochs
        self.layer_number = layer_number
        self.node_number = node_number
        self.lr = lr
        self.momentum = momentum
        self.activation = activation

    def run(self):
        # ''' An instance of Train for data handling '''
        para_train = {'name': '2j2b',
            'base_directory': self.output,
            'signal_h5': 'tW_DR_2j2b.h5',
            'signal_name': 'tW_DR_2j2b',
            'signal_tree': 'wt_DR_nominal',
            'weight_name': 'EventWeight',
            'backgd_h5': 'ttbar_2j2b.h5',
            'backgd_name': 'ttbar_2j2b',
            'backgd_tree': 'tt_nominal'}
        para_train_v28 = {'name': '2j2b',
            'base_directory': self.output,
            'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
            'signal_name': 'tW_DR',
            'signal_tree': 'wt_DR_nominal',
            'weight_name': 'weight_nominal',
            'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/ttbar_2j2b.h5',
            'backgd_name': 'ttbar',
            'backgd_tree': 'tt_nominal'}
        self.trainer = Train(**para_train_v28)
        self.trainer.split(nfold = 3)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self.deepnet = DeepNet(layer_number = self.layer_number, node_number = self.node_number, hidden_activation = self.activation)
        self.deepnet.build(input_dimension = self.trainer.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)
        self.trainer.getNetwork(self.deepnet.dnn)
        
        ''' Run the training '''
        self.result = self.trainer.train(epochs = self.epochs, fold = self.train_fold)
        self.trainer.evaluate(self.result)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()


    def run2(self):


        para_train_Gen = {'name': '2j2b',
            'base_directory': self.output,
            'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
            'signal_name': 'tW_DR',
            'signal_tree': 'wt_DR_nominal',
            'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/ttbar_2j2b.h5',
            'backgd_name': 'ttbar',
            'backgd_tree': 'tt_nominal',
            'weight_name': 'weight_nominal'}
        self.trainer_Adv = Train(**para_train_Gen)
        self.trainer_Adv.split(nfold = 3)

        para_train_Dis = {'name': 'NP',
            'base_directory': self.output,
            'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
            'signal_name': 'tW_DR',
            'signal_tree': 'wt_DR_nominal',
            'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DS_2j2b.h5',
            'backgd_name': 'tW_DS',
            'backgd_tree': 'wt_DS_nominal',
            'weight_name': 'weight_nominal'}
        self.trainer_Com = Train(**para_train_Dis)
        self.trainer_Com.split(nfold = 3)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self._Generator = DeepNet(name = 'Adv_Gen', layer_number = self.layer_number, node_number = self.node_number, hidden_activation = self.activation)
        self._Generator.build(input_dimension = self.trainer_Adv.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)
    
        self._Discriminator = DeepNet(name = 'Adv_Dis', single_input = True, layer_number = self.layer_number, node_number = self.node_number, hidden_activation = self.activation)
        self._Discriminator.build(input_dimension = self.trainer_Adv.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)

        self.advnet = AdvNet(generator = self._Generator, discriminator = self._Discriminator)
        self.advnet.build(input_dimension = self.trainer_Adv.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)

        self.compnet = CompNet(generator = self._Generator, adversary = self.advnet)
        self.compnet.build(lam = 10)

        for i in range(2):
            print('zhangr', i)
            self.advnet.Dis.make_trainable(False)
            self.advnet.Gen.make_trainable(True)
            print(self.trainer_Com.shape, self.trainer_Adv.shape)
            print('zhang eee', len(self.compnet.com._layers), vars(self.compnet.com))
            self.compnet.com.summary()
            print('zhang fff', len(self.advnet.adv._layers), vars(self.advnet.adv))
            self.advnet.adv.summary()
            self.trainer_Com.getNetwork(self.compnet.com)
            print('zhangrui 0')
            self.result = self.trainer_Com.train(epochs = self.epochs, fold = self.train_fold)

            # self.advnet.Dis.make_trainable(True)
            # self.advnet.Gen.make_trainable(False)
            # self.advnet.adv.summary()
            # self.trainer_Adv.getNetwork(self.advnet.adv)
            # self.result = self.trainer_Adv.train(epochs = self.epochs, fold = self.train_fold)



# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 100, lr = 0.01, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 30, lr = 0.01, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 100, lr = 0.01, momentum = 0.4)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 30, lr = 0.01, momentum = 0.4)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 50, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 30, lr = 0.02, momentum = 0.8) # the best 75.2
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 20, node_number = 30, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 10, node_number = 30, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 10, node_number = 50, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 10, node_number = 100, lr = 0.03, momentum = 0.8) # the best 76.7
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 10, node_number = 150, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 5, node_number = 30, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 5, node_number = 50, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 5, node_number = 100, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 5, node_number = 30, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 5, node_number = 50, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, layer_number = 5, node_number = 100, lr = 0.03, momentum = 0.8)
# job.run()
job = Job(output = 'test2', nfold = 3, train_fold = 0, epochs = 1, layer_number = 1, node_number = 10, lr = 0.01, momentum = 0.8)
job.run2()
