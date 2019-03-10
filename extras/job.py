from net import DeepNet, AdvNet
from train import Train

class Job(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name, nfold, train_fold, epochs, hidden_Nlayer, hidden_Nnode, lr, momentum, output, activation, para_train={}):
        self.nfold = int(nfold)
        self.train_fold = int(train_fold)
        self.epochs = int(epochs)
        self.hidden_Nlayer = int(hidden_Nlayer)
        self.hidden_Nnode = int(hidden_Nnode)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.activation = activation
        self.output = 'l{}n{}_lr{}mom{}_{}_k{}_e{}'.format(self.hidden_Nlayer, self.hidden_Nnode, self.lr, self.momentum, self.activation, self.nfold, self.epochs) if output is None else output
        self.name = self.output if name is None else name

        self.para_train = para_train
        para_train['base_directory'] = self.output

    def run(self):

        ''' An instance of Train for data handling '''
        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = self.nfold)

        ''' An instance of DeepNet for network construction and pass it to Train '''
        self.deepnet = DeepNet(name = self.name, build_dis = False, hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode, hidden_activation = self.activation)
        self.deepnet.build(input_dimension = self.trainer.shape, lr = self.lr, momentum = self.momentum)
        self.deepnet.plot(base_directory = self.output)
        self.trainer.getNetwork(self.deepnet.generator)
        
        ''' Run the training '''
        self.result = self.trainer.train(mode = 0, epochs = self.epochs, fold = self.train_fold)
        self.trainer.evaluate(self.result)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()


class JobAdv(Job):
    def __init__(self, preTrain_epochs, hidden_auxNlayer, hidden_auxNnode, n_iteraction, lam, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.preTrain_epochs = preTrain_epochs
        self.hidden_auxNlayer = hidden_auxNlayer
        self.hidden_auxNnode = hidden_auxNnode
        self.n_iteraction = n_iteraction
        self.lam = lam
        self.output = '{}__E{}_L{}N{}_it{}_lam{}'.format(self.output, self.preTrain_epochs, self.hidden_auxNlayer, self.hidden_auxNnode, self.n_iteraction, self.lam)

    def run(self):

        ''' An instance of Train for data handling '''
        self.trainer = Train(**self.para_train)
        self.trainer.split(nfold = self.nfold)

        ''' An instance of AdvNet for network construction and pass it to Train '''
        self.advnet = AdvNet(name = self.name, build_dis = True, hidden_Nlayer = self.hidden_Nlayer, hidden_Nnode = self.hidden_Nnode,
            hidden_activation = self.activation, hidden_auxNlayer = self.hidden_auxNlayer, hidden_auxNnode = self.hidden_auxNnode)
        self.advnet.build(input_dimension = self.trainer.shape, lam = self.lam, lr = self.lr, momentum = self.momentum)
        self.advnet.plot(base_directory = self.output)
    
        ''' pre-training '''
        AdvNet.make_trainable(self.advnet.generator, True)
        AdvNet.make_trainable(self.advnet.discriminator, False)
        self.trainer.getNetwork(self.advnet.generator)
        self.result = self.trainer.train(mode = 0, epochs = self.preTrain_epochs, fold = self.train_fold)

        AdvNet.make_trainable(self.advnet.generator, False)
        AdvNet.make_trainable(self.advnet.discriminator, True)
        self.trainer.getNetwork(self.advnet.discriminator)
        self.result = self.trainer.train(mode = 1, epochs = self.preTrain_epochs, fold = self.train_fold)

        ''' Iterative training '''
        for i in range(self.n_iteraction):
            AdvNet.make_trainable(self.advnet.generator, True)
            AdvNet.make_trainable(self.advnet.discriminator, False)
            self.trainer.getNetwork(self.advnet.adversary)
            self.result = self.trainer.train(mode = 2, epochs = self.epochs, fold = self.train_fold)

            AdvNet.make_trainable(self.advnet.generator, False)
            AdvNet.make_trainable(self.advnet.discriminator, True)
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

