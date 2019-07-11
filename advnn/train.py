import numpy as np
import keras
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from twaml.data import from_pytables
from twaml.data import scale_weight_sum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import pickle


class Train(object):
    def describe(self): return self.__class__.__name__
    def __init__(self, name = 'zero_jet', base_directory = '../h5', signal_h5 = 'sig_one_jet.h5', signal_name = 'sig', signal_tree = 'wt_DR_nominal', signal_latex = r'H$\rightarrow\mu\mu$',
            backgd_h5 = 'bkg_zero_jet.h5', backgd_name = 'bkg', backgd_tree = 'tt_nominal', backgd_latex =  r'Data sideband', weight_name = 'weight',
            variables = ['Z_PT_FSR', 'Z_Y_FSR', 'Muons_CosThetaStar', 'm_mumu'],
            has_syst = False, syssig_h5 = 'tW_DS_2j2b.h5', syssig_name = 'tW_DS_2j2b', syssig_tree = 'tW_DS', syssig_latex = r'$tW$ DS',
            has_mass = True, reg_variable = 'm_mumu', reg_latex = r'm_\mu\mu',
            ):
        self.name = name
        self.signal_label, self.backgd_label, self.center_label, self.syssig_label = 1, 0, 1, 0
        self.signal_latex, self.backgd_latex = signal_latex, backgd_latex
        self.signal = from_pytables(signal_h5, signal_name, tree_name = signal_tree, weight_name = weight_name, label = self.signal_label, auxlabel = self.center_label)
        self.backgd = from_pytables(backgd_h5, backgd_name, tree_name = backgd_tree, weight_name = weight_name, label = self.backgd_label, auxlabel = self.center_label)
        self.signal.keep_columns(variables)
        self.backgd.keep_columns(variables)
        self.has_syst = has_syst
        self.syssig_latex = None if not self.has_syst else syssig_latex
        self.losses_test = {'L_gen': [], 'L_dis': [], 'L_diff': []}
        self.losses_train = {'L_gen': [], 'L_dis': [], 'L_diff': []}
        self.has_mass = has_mass
        self.reg_variable = reg_variable
        self.reg_latex = reg_latex

        if self.has_syst:
            self.syssig = from_pytables(syssig_h5, syssig_name, tree_name = syssig_tree, weight_name = weight_name, label = self.signal_label, auxlabel = self.syssig_label)
            self.syssig.keep_columns(variables)

            # Append syssig to signal
            self.signal.append(self.syssig)


        # Equalise signal weights to background weights
        scale_weight_sum(self.signal, self.backgd)
        self.X_raw = np.concatenate([self.signal.df.to_numpy(), self.backgd.df.to_numpy()])
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X_raw)
        self.y = np.concatenate([self.signal.label_asarray(), self.backgd.label_asarray()])
        self.z = np.concatenate([self.signal.auxlabel_asarray(), self.backgd.auxlabel_asarray()])
        if self.has_mass:
            signal = from_pytables(signal_h5, signal_name, tree_name = signal_tree, weight_name = weight_name, label = self.signal_label, auxlabel = self.center_label)
            backgd = from_pytables(backgd_h5, backgd_name, tree_name = backgd_tree, weight_name = weight_name, label = self.backgd_label, auxlabel = self.center_label)
            signal.keep_columns([reg_variable])
            backgd.keep_columns([reg_variable])
            self.z = np.concatenate([signal.df.to_numpy(), backgd.df.to_numpy()])
        self.w = np.concatenate([self.signal.weights, self.backgd.weights])

        self.output_path = '/'.join([base_directory, self.describe()]) + '/'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        print('[INFO]', self.describe(), self.signal.df.__getitem__, self.backgd.df.__getitem__)
        print('[INFO]', '-'*20)

        #store the content
        with open(self.output_path + self.name + '_event.pkl', 'wb') as pkl:
            pickle.dump(scaler, pkl)

    @property
    def shape(self):
        return self.signal.shape[1]

    def setNetwork(self, net):
        self.network = net

    def split(self, nfold, seed = 666):
        ''' Split sample to training and test portions using KFold '''
 
        self.nfold = nfold
        kfolder = KFold(n_splits = self.nfold, shuffle = True, random_state = seed)

        self.X_train, self.X_test = {}, {}
        self.y_train, self.y_test = {}, {}
        self.z_train, self.z_test = {}, {}
        self.w_train, self.w_test = {}, {}
        for i, (train_idx, test_idx) in enumerate(kfolder.split(self.X)):
            self.X_train[i], self.X_test[i] = self.X[train_idx], self.X[test_idx]
            self.y_train[i], self.y_test[i] = self.y[train_idx], self.y[test_idx]
            self.z_train[i], self.z_test[i] = self.z[train_idx], self.z[test_idx]
            self.w_train[i], self.w_test[i] = self.w[train_idx], self.w[test_idx]

    def train(self, mode, epochs, fold):
        '''
        mode = 0: one target mode => signal vs backgd
               1: one target mode => signal vs syssig
               2: two target mode => signal+syssig vs backgd and signal+backgd vs syssig
        '''
        self.epochs = epochs
        self.fold = fold
        checkpoint = keras.callbacks.ModelCheckpoint(self.output_path + self.name + '_model_{epoch:04d}.h5', period=int(self.epochs/10.)) 
        # serialize model to JSON
        model_json = self.network.to_json()
        with open(self.output_path + self.name + '_model.json', 'w') as json_file:
            json_file.write(model_json)

        if mode == 0:
            return self.network.fit(self.X_train[self.fold], self.y_train[self.fold], sample_weight = self.w_train[self.fold], batch_size = 512,
                    validation_data = (self.X_test[self.fold], self.y_test[self.fold], self.w_test[self.fold]), epochs = self.epochs, callbacks=[checkpoint])
        elif mode == 1:
            assert (self.has_syst)
            return self.network.fit(self.X_train[self.fold], self.z_train[self.fold], sample_weight = self.w_train[self.fold], batch_size = 512,
                    validation_data = (self.X_test[self.fold], self.z_test[self.fold], self.w_test[self.fold]), epochs = 1)

        elif mode == 2:
            assert (self.has_syst)
            return self.network.fit(self.X_train[self.fold],  [self.y_train[self.fold], self.z_train[self.fold]], sample_weight = [self.w_train[self.fold], self.w_train[self.fold]], batch_size = 512,
                    validation_data = (self.X_test[self.fold], [self.y_test[self.fold], self.z_test[self.fold]], [self.w_test[self.fold], self.w_test[self.fold]]), epochs = self.epochs)

    def evaluate(self):
        print('Evaluating ...', self.has_syst, self.name)
        if not self.has_syst:
            self.network.evaluate(self.X_train[self.fold], self.y_train[self.fold], sample_weight = self.w_train[self.fold], verbose=0)
            self.network.evaluate(self.X_test[self.fold], self.y_test[self.fold], sample_weight = self.w_test[self.fold], verbose=0)
        else:
            loss_train = self.network.evaluate(self.X_train[self.fold],  [self.y_train[self.fold], self.z_train[self.fold]], sample_weight = [self.w_train[self.fold], self.w_train[self.fold]], verbose=0)
            loss_test = self.network.evaluate(self.X_test[self.fold], [self.y_test[self.fold], self.z_test[self.fold]], sample_weight = [self.w_test[self.fold], self.w_test[self.fold]], verbose=0)
            return loss_train, loss_test


    def plotLoss(self, result):
        ''' Plot loss functions '''
        if self.epochs < 2:
            print('Only', self.epochs, 'epochs, no need for plotLoss.')
            return

        # Summarise history for accuracy
        plt.plot(result.history['acc'])
        plt.plot(result.history['val_acc'])
        plt.title('network accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.output_path + self.name + '_acc' + '.pdf', format='pdf')
        plt.clf()
        # Summarise history for loss
        plt.plot(result.history['loss'])
        plt.plot(result.history['val_loss'])
        plt.title('network loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig(self.output_path + self.name + '_loss' + '.pdf', format='pdf')
        plt.clf()

    def plotResults(self, xlo = 0., xhi = 1, nbin = 20):
        from sklearn.metrics import roc_curve, auc

        train_predict = self.network.predict(self.X_train[self.fold])
        test_predict = self.network.predict(self.X_test[self.fold])

        train_FP, train_TP, train_TH = roc_curve(self.y_train[self.fold], train_predict)
        test_FP, test_TP, test_TH = roc_curve(self.y_test[self.fold], test_predict)
        train_AUC = auc(train_FP, train_TP)
        test_AUC = auc(test_FP, test_TP)

        plt.title('Receiver Operating Characteristic')
        plt.plot(train_FP, train_TP, 'g--', label='Train AUC = %2.1f%%'% (train_AUC * 100))
        plt.plot(test_FP, test_TP, 'b', label='Test  AUC = %2.1f%%'% (test_AUC * 100))
        plt.legend(loc='lower right')
        
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.,1.])
        plt.ylim([-0.,1.])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(self.output_path + self.name + '_ROC' + '.pdf', format='pdf')
        plt.clf()

        names = ['Absolute', 'Normalised']
        for density in [0, 1]:
            plt.subplot(1, 2, density + 1)

            plt.hist(train_predict[self.y_train[self.fold] == self.signal_label], range = [xlo, xhi], bins = nbin, histtype = 'step', density = density, label='Training ' + self.signal_latex)
            plt.hist(train_predict[self.y_train[self.fold] == self.backgd_label], range = [xlo, xhi], bins = nbin, histtype = 'step', density = density, label='Training ' + self.backgd_latex)
            plt.hist(test_predict[self.y_test[self.fold] == self.signal_label],   range = [xlo, xhi], bins = nbin, histtype = 'step', density = density, label='Test ' + self.signal_latex, linestyle = 'dashed')
            plt.hist(test_predict[self.y_test[self.fold] == self.backgd_label],   range = [xlo, xhi], bins = nbin, histtype = 'step', density = density, label='Test ' + self.backgd_latex, linestyle = 'dashed')
            plt.ylim(0, plt.gca().get_ylim()[1] * 1.5)
            plt.legend()
            plt.xlabel('Response', horizontalalignment = 'left', fontsize = 'large')
            plt.title(names[density])

        plt.savefig(self.output_path + self.name + '_response' + '.pdf', format='pdf')
        plt.clf()

        with open(self.output_path + self.name + '_ROC' + '.txt', 'w') as f:
            f.write('Train AUC = %2.1f %%\n'% (train_AUC * 100))
            f.write('Test  AUC = %2.1f %%\n'% (test_AUC * 100))


    def plotIteration(self, it):
        if not self.has_syst:
            return

        loss_train, loss_test = self.evaluate()
        self.losses_test['L_gen'].append(loss_test[1][None][0])
        self.losses_test['L_dis'].append(-loss_test[2][None][0])
        self.losses_test['L_diff'].append(loss_test[0][None][0])
        self.losses_train['L_gen'].append(loss_train[1][None][0])
        self.losses_train['L_dis'].append(-loss_train[2][None][0])
        self.losses_train['L_diff'].append(loss_train[0][None][0])

        def plot_twolosses():
            idxes = ['L_gen', 'L_dis', 'L_diff']
            latex = [r'$L_{\mathrm{gen}}$', r'$\lambda \cdot L_{\mathrm{dis}}$', r'$L_{\mathrm{gen}} - \lambda \cdot L_{\mathrm{dis}}$']
            for idx in range(len(self.losses_test)):
                ax = plt.subplot(3, 1, idx + 1)

                plt.plot(np.arange(len(self.losses_train[idxes[idx]])), self.losses_test[idxes[idx]], '--', label = r'Test ')
                plt.plot(np.arange(len(self.losses_train[idxes[idx]])), self.losses_train[idxes[idx]], '', label = r'Train')
                
                plt.legend(loc='upper right')
                plt.ylabel(latex[idx], fontsize='large')
                plt.grid()

            plt.xlabel('Number of iterations', horizontalalignment='left', fontsize='large')
            plt.subplots_adjust(left=0.18, right=0.95, top=0.95, hspace = 0.4)
            plt.savefig(self.output_path + self.name + '_iter.pdf', format = 'pdf')
            plt.clf()

        if not it % 2:
            plot_twolosses()

    def saveLoss(self):
        import json
        with open(self.output_path + self.name + '_iter.json', 'w') as fp:
            print('Saved', fp.name, 'to disk')
            json.dump(self.losses_test, fp)
            json.dump(self.losses_train, fp)
