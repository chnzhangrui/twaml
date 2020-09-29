from keras.models import model_from_json
from argparse import ArgumentParser
from root_numpy import tree2array, array2tree, array2root
from shutil import copyfile
from ROOT import TFile, TDirectory
import numpy as np
import pickle

def loadModel(json_file, h5_file):
    # load json and create model
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_file)
    return loaded_model

def apply(json, h5, pkl, root_in, root_out, treename):
    inFile = TFile(root_in)
    print(inFile.GetName())

    copyfile(root_in, root_out)
    print ('copy', root_in, root_out)
    outBranch = 'DNN_response'

    ''' implement ANN on each ttree '''
    for name in [treename]:
        print('Implementing:', name)

        ''' *array* is [eventVariables, EventWeight]; *event* is [eventVariables]; *weight* is [EventWeight]'''
        tree_list = []
        variables = []
        weight = []
        array = []
        test_event = np.array

        if 'zero_jet' in root_in:
            tree_list.append(inFile.Get('zero_jet'))
            variables.append(tree2array(tree_list[-1], branches=['Z_PT_FSR_scaled', 'Z_Y_FSR', 'Muons_CosThetaStar'], selection='1'))
        elif 'one_jet' in root_in:
            variables.append(tree2array(tree_list[-1], branches=['Z_PT_FSR_scaled', 'Z_Y_FSR', 'Muons_CosThetaStar', 'Jets_PT_Lead', 'Jets_Eta_Lead', 'DeltaPhi_mumuj1'], selection='1'))
        elif 'two_jet' in root_in:
            variables.append(tree2array(tree_list[-1], branches=['Z_PT_FSR_scaled', 'Z_Y_FSR', 'Muons_CosThetaStar', 'Jets_PT_Lead', 'Jets_Eta_Lead', 'DeltaPhi_mumuj1', 'Jets_PT_Sub', 'Jets_Eta_Sub', 'DeltaPhi_mumuj2', 'Jets_PT_jj', 'Jets_Y_jj', 'DeltaPhi_mumujj', 'Jets_Minv_jj', 'Event_MET' ], selection='1'))
        else:
            raise RuntimeError('Unknown jet bin: {}'.format(root_in))
        weight.append(tree2array(tree_list[-1], branches=[ 'weight' ], selection='1'))
        array.append([list(elem) for elem in zip(variables[-1], weight[-1])])

        event__list = []
        for ivar in variables:
            for i in ivar:
                event__list.append(list(i))
        test_event = np.vstack(event__list)


        #load the content
        scaler = pickle.load(open(pkl, 'rb'))
        test_event_transfered = scaler.transform(test_event)
        predicttest__ANN = loadModel(json, h5).predict(test_event_transfered)


        weight__list = []
        for iwegt in weight:
            for i in iwegt:
                weight__list.append(list(i))
        test_weight = np.vstack(weight__list)

        response__test_event = np.column_stack((test_event, test_weight, predicttest__ANN))

        variable_array = response__test_event[:,-1]
        variable_array.dtype = [(outBranch, response__test_event.dtype)]
        array2root(variable_array, treename=name, filename=root_out)


    print(root_out, 'is saved and closed.')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input ROOT file", metavar="FILE", default='data_zero_jet.root')
    parser.add_argument("--treename", dest="treename", help="Input Tree name", metavar="FILE", default='zero_jet')
    parser.add_argument("--output", dest="output", help="Output ROOT file", metavar="FILE", default='ANN_data_zero_jet.root')
    parser.add_argument("--json", dest="json", help="Model json file", metavar="FILE", default='/Users/zhangrui/Work/Code/ML/ANN/training/job__l5n50_lr0.01mom0.8_elu_k3_dp0.2_e1_plb1__E20_L5N10_it100_Loss1_lam1.0/JobAdv/0j_100.json')
    parser.add_argument("--h5", dest="h5", help="Model weight h5 file", metavar="FILE", default='/Users/zhangrui/Work/Code/ML/ANN/training/job__l5n50_lr0.01mom0.8_elu_k3_dp0.2_e1_plb1__E20_L5N10_it100_Loss1_lam1.0/JobAdv/0j_100.h5')
    parser.add_argument("--pkl", dest="pkl", help="Event stored pkl file after transformation", metavar="FILE", default='/Users/zhangrui/Work/Code/ML/ANN/training/job__l5n50_lr0.01mom0.8_elu_k3_dp0.2_e1_plb1__E20_L5N10_it100_Loss1_lam1.0/Train/0j_event.pkl')
    args = parser.parse_args()
    apply(root_in = args.input, treename = args.treename, root_out = args.output, json = args.json, h5 = args.h5, pkl = args.pkl)
