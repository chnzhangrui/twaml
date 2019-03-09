from net import DeepNet
from train import Train

class Job(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name = None, output = None, nfold = 3, train_fold = 0, epochs = 10, hidden_Nlayer = 4, hidden_Nnode = 20,
            lr = 0.02, momentum = 0.8, activation = 'relu', para_train = {}):
        self.output = 'l{}n{}_lr{}_mom{}_{}_f{}_e{}'.format(hidden_Nlayer, hidden_Nnode, lr, momentum, activation, nfold, epochs) if output is None else output
        self.name = self.output if name is None else name
        para_train['base_directory'] = self.output
        self.nfold = nfold
        # Train the "train_fold"-th fold
        self.train_fold = train_fold
        self.epochs = epochs
        self.hidden_Nlayer = hidden_Nlayer
        self.hidden_Nnode = hidden_Nnode
        self.lr = lr
        self.momentum = momentum
        self.activation = activation
        self.para_train = para_train

    def run(self):
        # ''' An instance of Train for data handling '''

        self.trainer = Train(**para_train)
        self.trainer.split(nfold = 3)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self.deepnet = DeepNet(hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode, hidden_activation = self.activation)
        self.deepnet.build(input_dimension = self.trainer.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)
        self.trainer.getNetwork(self.deepnet.generator)
        
        ''' Run the training '''
        self.result = self.trainer.train(epochs = self.epochs, fold = self.train_fold)
        self.trainer.evaluate(self.result)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()


class JobAdv(Job):
    def __init__(self, n_iteraction = 1, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.n_iteraction = n_iteraction

    def run(self):

        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = 3)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self.advnet = DeepNet(name = 'AdvNN', build_dis = True, hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode, hidden_activation = self.activation)
        self.advnet.build(input_dimension = self.trainer.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)
    
        ''' Run the training '''
        for i in range(self.n_iteraction):
            DeepNet.make_trainable(self.advnet.discriminator, False)
            DeepNet.make_trainable(self.advnet.generator, True)
            self.trainer.getNetwork(self.advnet.adversary)
            self.result = self.trainer.train(mode = 2, epochs = self.epochs, fold = self.train_fold)

            DeepNet.make_trainable(self.advnet.discriminator, True)
            DeepNet.make_trainable(self.advnet.generator, False)
            self.trainer.getNetwork(self.advnet.discriminator)
            self.result = self.trainer.train(mode = 1, epochs = self.epochs, fold = self.train_fold)

        self.trainer.evaluate(self.result)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()

# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 100, lr = 0.01, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 30, lr = 0.01, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 100, lr = 0.01, momentum = 0.4)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 30, lr = 0.01, momentum = 0.4)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 50, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 30, lr = 0.02, momentum = 0.8) # the best 75.2
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 20, hidden_Nnode = 30, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 10, hidden_Nnode = 30, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 10, hidden_Nnode = 50, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 10, hidden_Nnode = 100, lr = 0.03, momentum = 0.8) # the best 76.7
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 10, hidden_Nnode = 150, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 5, hidden_Nnode = 30, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 5, hidden_Nnode = 50, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 5, hidden_Nnode = 100, lr = 0.02, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 5, hidden_Nnode = 30, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 5, hidden_Nnode = 50, lr = 0.03, momentum = 0.8)
# job.run()
# job = Job(nfold = 3, train_fold = 0, epochs = 500, hidden_Nlayer = 5, hidden_Nnode = 100, lr = 0.03, momentum = 0.8)
# job.run()

para_train = {'name': '2j2b',
    'signal_h5': 'tW_DR_2j2b.h5',
    'signal_name': 'tW_DR_2j2b',
    'signal_tree': 'wt_DR_nominal',
    'weight_name': 'EventWeight',
    'backgd_h5': 'ttbar_2j2b.h5',
    'backgd_name': 'ttbar_2j2b',
    'backgd_tree': 'tt_nominal'}
para_train_v28 = {'name': '2j2b',
    'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
    'signal_name': 'tW_DR',
    'signal_tree': 'wt_DR_nominal',
    'weight_name': 'weight_nominal',
    'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/ttbar_2j2b.h5',
    'backgd_name': 'ttbar',
    'backgd_tree': 'tt_nominal'}

para_train_Adv = {'name': 'NP',
    'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
    'signal_name': 'tW_DR',
    'signal_tree': 'wt_DR_nominal',
    'no_syssig': False,
    'syssig_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DS_2j2b.h5',
    'syssig_name': 'tW_DS',
    'syssig_tree': 'wt_DS', 
    'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/ttbar_2j2b.h5',
    'backgd_name': 'ttbar',
    'backgd_tree': 'tt_nominal',
    'weight_name': 'weight_nominal',
    'variables': ['mass_lep1jet2', 'mass_lep1jet1'],}

job = JobAdv(n_iteraction = 2, output = 'test1', nfold = 3, train_fold = 0, epochs = 1, hidden_Nlayer = 3, hidden_Nnode = 10, lr = 0.01, momentum = 0.8, para_train = para_train_Adv)
job.run()
