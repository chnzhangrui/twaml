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
    
    inputs = {'name': '2j2b',
        'signal_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DR_2j2b.h5',
        'backgd_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/ttbar_2j2b.h5',
        'variables': ['mass_lep1jet2', 'mass_lep1jet1', 'deltaR_lep1_jet1', 'mass_lep2jet1', 'pTsys_lep1lep2met', 'pT_jet2', 'mass_lep2jet2'],
    }

    batch = Batch(base_directory, inputs)

    args = parse_options(sys.argv[1:])

    ''' Grid search '''
    job_array = {
        'hidden_Nlayer': ['10', '20', '5'],
        'hidden_Nnode': ['30', '50', '100'],
        'lr': ['0.03', '0.05'],
        'activation': ['elu', 'relu'],
    }

    if args.command == '_run':
        batch._run(args._run)

    elif args.command == 'htc':
        batch.create_jdl(job_array)
        batch.create_wrap()

    elif args.command == 'all':
        batch.create_jdl(job_array, local_run = True)

    else:
        raise RuntimeError('Unknown command: {}'.format(args.command))