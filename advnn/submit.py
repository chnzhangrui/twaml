from batch import Batch

import argparse
import sys, os

def parse_options(args):
    parser = argparse.ArgumentParser(description='Neural network training', prog='submit')
    
    subcommands = parser.add_subparsers(dest='command')

    _run = subcommands.add_parser('_run', help='Run training with current setting')
    _run.add_argument('_run', nargs='*')

    htc = subcommands.add_parser('htc', help='Generate HTCondor scripts')
    run = subcommands.add_parser('all', help='Run trainings locally')

    return parser.parse_args(args)

if __name__ == '__main__':
    base_directory = os.getcwd()
    
    inputs = {'name': '0j',
        'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5/low-massscaled/sig_zero_jet.h5',
        'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5/low-massscaled/bkg_zero_jet.h5',
        'syssig_h5': '/cephfs/user/rzhang/Wtr21/run/v28/h5files/tW_DS_2j2b.h5',
        #'variables': ['Z_PT_FSR', 'Z_Y_FSR', 'Muons_CosThetaStar', 'Muons_PT_Lead', 'Muons_PT_Sub', 'Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub'],
        'variables': ['Z_PT_FSR_scaled', 'Z_Y_FSR', 'Muons_CosThetaStar', 'Muons_PT_Lead_scaled', 'Muons_PT_Sub_scaled', 'Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub'],
    }
    if 'macproruizhang2019' not in os.uname()[1] and 'macbook' not in os.uname()[1].lower():
        inputs['signal_h5'] = '/afs/cern.ch/user/z/zhangr/work/Hmumu/h5/low-massscaled/sig_zero_jet.h5'
        inputs['backgd_h5'] = '/afs/cern.ch/user/z/zhangr/work/Hmumu/h5/low-massscaled/bkg_zero_jet.h5'

    jobname = sys.argv[1]
    batch = Batch(jobname, base_directory, inputs)
    args = parse_options(sys.argv[2:])

    ''' Grid search '''
    DNN_job_array = {
        'hidden_Nlayer': ['5', '10'],
        'hidden_Nnode': ['10', '30', '50'],
        'lr': ['0.001', '0.005'],
        'activation': ['elu'],
        'dropout_rate': ['0.2', '0.5'],
    }
    ANN_job_array = {
        'hidden_Nlayer': ['5'],
        'hidden_Nnode': ['50'],
        'lr': ['0.005'],
        'activation': ['elu'],
        'dropout_rate': ['0.2'],
        'preTrain_epochs': ['2', '5'],
        'hidden_auxNlayer': ['5'],
        'hidden_auxNnode': ['20'],
        'problem': ['1'],
        'n_iteraction': ['100'],
        'epochs': ['1','3'],
        'alr': ['0.0000001'],
        'amomentum': ['0.6'],
        'lam': ['0.1', '0.2', '0.3', '0.4', '0.5'],
    }

    if jobname == 'DNN':
        job_array = DNN_job_array
    elif jobname == 'ANN' or jobname == 'ANNReg':
        job_array = ANN_job_array
    else:
        raise RuntimeError('Unknown job name: {}'.format(jobname))


    if args.command == '_run':
        batch._run(args._run)

    elif args.command == 'htc':
        batch.create_jdl(job_array)
        batch.create_wrap()

    elif args.command == 'all':
        batch.create_jdl(job_array, local_run = True)

    else:
        raise RuntimeError('Unknown command: {}'.format(args.command))
