from net import DeepNet, AdvNet
from train import Train

class Job(object):
    def describe(self): return self.__class__.__name__
    # def __init__(self, name = None, output = None, nfold = 3, train_fold = 0, epochs = 10, hidden_Nlayer = 4, hidden_Nnode = 20,
    #         lr = 0.02, momentum = 0.8, activation = 'relu', para_train = {}):
    def __init__(self, para_net = {}, para_train = {}):
        self.para_train = para_train

        self.nfold = para_net['nfold']
        self.train_fold = para_net['train_fold']
        self.epochs = para_net['epochs']
        self.hidden_Nlayer = para_net['hidden_Nlayer']
        self.hidden_Nnode = para_net['hidden_Nnode']
        self.lr = para_net['lr']
        self.momentum = para_net['momentum']
        self.activation = 'relu' if 'activation' not in para_net else para_net['activation']
        self.output = 'l{}n{}_lr{}_mom{}_{}_f{}_e{}'.format(self.hidden_Nlayer, self.hidden_Nnode, self.lr, self.momentum, self.activation, self.nfold, self.epochs) if 'output' not in para_net else para_net['output']
        self.name = self.output if 'name' not in para_net else para_net['name']
        para_train['base_directory'] = self.output

    def run(self):
        # ''' An instance of Train for data handling '''

        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = self.nfold)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self.deepnet = DeepNet(hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode, hidden_activation = self.activation)
        self.deepnet.build(input_dimension = self.trainer.shape, lr = self.lr, momentum = self.momentum)
        self.deepnet.plot(base_directory = self.output)
        self.trainer.getNetwork(self.deepnet.generator)
        
        ''' Run the training '''
        self.result = self.trainer.train(epochs = self.epochs, fold = self.train_fold)
        self.trainer.evaluate(self.result)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()


class JobAdv(Job):
    def __init__(self, hidden_auxNlayer = 1, hidden_auxNnode = 5, n_iteraction = 1, lam = 10, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.hidden_auxNlayer = hidden_auxNlayer
        self.hidden_auxNnode = hidden_auxNnode
        self.n_iteraction = n_iteraction
        self.lam = lam

    def run(self):

        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = 3)

        ''' An instance of AdvNet for network construction and pass it to Train '''
        self.advnet = AdvNet(name = 'AdvNN', build_dis = True, hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode,
            hidden_activation = self.activation, hidden_auxNlayer = self.hidden_auxNlayer, hidden_auxNnode = self.hidden_auxNnode)
        self.advnet.build(input_dimension = self.trainer.shape, lam = self.lam, lr = self.lr, momentum = self.momentum)
        self.advnet.plot(base_directory = self.output)
    
        ''' Run the training '''
        for i in range(self.n_iteraction):
            AdvNet.make_trainable(self.advnet.discriminator, False)
            AdvNet.make_trainable(self.advnet.generator, True)
            self.trainer.getNetwork(self.advnet.adversary)
            self.result = self.trainer.train(mode = 2, epochs = self.epochs, fold = self.train_fold)

            AdvNet.make_trainable(self.advnet.discriminator, True)
            AdvNet.make_trainable(self.advnet.generator, False)
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


para_train_sim = {'name': '2j2b',
    'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
    'signal_name': 'tW_DR',
    'signal_tree': 'wt_DR_nominal',
    'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/ttbar_2j2b.h5',
    'backgd_name': 'ttbar',
    'backgd_tree': 'tt_nominal',
    'weight_name': 'weight_nominal',
    'variables': ['mass_lep1jet2', 'mass_lep1jet1', 'deltaR_lep1_jet1', 'mass_lep2jet1', 'pTsys_lep1lep2met', 'pT_jet2', 'mass_lep2jet2'],
    }

para_net_sim = {
    'nfold': 3,
    'train_fold': 0,
    'epochs': 500,
    'hidden_Nlayer': 3,
    'hidden_Nnode': 10,
    'lr': 0.01,
    'momentum': 0.8,
    }

# job = Job(para_net = para_net_sim, para_train = para_train_sim)
# job.run()

para_train_Adv = {**para_train_sim,
    'name': 'NP',
    'no_syssig': False,
    'syssig_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DS_2j2b.h5',
    'syssig_name': 'tW_DS',
    'syssig_tree': 'wt_DS',
    }

para_net_Adv = {**para_net_sim,
    'hidden_auxNlayer': 2,
    'hidden_auxNnode': 5,
    # 'pre_epochs': 20,
    'n_iteraction': 2,
    'lam': 10,
}
job = JobAdv(para_net = para_net_Adv, para_train = para_train_Adv)
job.run()