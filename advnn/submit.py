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
        'signal_h5': '/afs/cern.ch/user/z/zhangr/work/Hmumu/h5/sig_zero_jet.h5',
        'backgd_h5': '/afs/cern.ch/user/z/zhangr/work/Hmumu/h5/bkg_zero_jet.h5',
        'syssig_h5': '/cephfs/user/rzhang/Wtr21/run/v28/h5files/tW_DS_2j2b.h5',
        'variables': ['Z_PT_FSR', 'Z_Y_FSR', 'Muons_CosThetaStar'],
    }

    jobname = sys.argv[1]
    batch = Batch(jobname, base_directory, inputs)
    args = parse_options(sys.argv[2:])

    ''' Grid search '''
    job_array = {
        'hidden_Nlayer': ['3', '5', '10'],
        'hidden_Nnode': ['10', '30', '50'],
        'lr': ['0.001', '0.0005'],
        'activation': ['elu'],
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
