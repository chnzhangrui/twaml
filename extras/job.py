from net import Net
from train import Train

class Job(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name = None, output = None, nfold = 3, train_fold = 0, epochs = 10, layer_number = 4, node_number = 20, lr = 0.02, momentum = 0.8):
        self.output = 'l{}n{}_lr{}_mom{}'.format(layer_number, node_number, lr, momentum) if output is None else output
        self.name = self.output if name is None else name
        self.nfold = nfold
        # Train the "train_fold"-th fold
        self.train_fold = train_fold
        self.epochs = epochs
        self.layer_number = layer_number
        self.node_number = node_number
        self.lr = lr
        self.momentum = momentum

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

        ''' An instance of Net for network construction and pass it to Train '''
        self.net = Net(layer_number = self.layer_number, node_number = self.node_number)
        self.net.build(input_dimension = self.trainer.shape, base_directory = self.output, lr = self.lr, momentum = self.momentum)
        self.trainer.model(self.net.model)
        
        ''' Run the training '''
        self.result = self.trainer.train(epochs = self.epochs, fold = self.train_fold)
        self.trainer.evaluate(self.result)
        self.trainer.plotLoss(self.result)
        self.trainer.plotResults()

job = Job(nfold = 3, train_fold = 0, epochs = 50, layer_number = 10, node_number = 50, lr = 0.01, momentum = 0.8)
job.run()
