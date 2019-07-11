from job import Job, JobAdv
import itertools    
from itertools import chain
import os

def update_dict(orig_dict, new_dict):
    for k, v in new_dict.items():
        if k in orig_dict:
            orig_dict[k] = v

class Batch(object):
    def __init__(self, jobname, base_directory, inputs):
        self.jobname = jobname

        self.base_directory = base_directory + '/'
        self.para_train_sim = {'name': '2j2b',
            'signal_h5': 'sig_zero_jet.h5',
            'signal_name': 'sig',
            'signal_tree': 'AppInputs',
            'backgd_h5': 'bkg_zero_jet.h5',
            'backgd_name': 'bkg',
            'backgd_tree': 'AppInputs',
            'weight_name': 'weight',
            'variables': ['Z_PT_FSR', 'Z_Y_FSR', 'Muons_CosThetaStar'],
            }
        update_dict(self.para_train_sim, inputs)

        self.para_net_sim = {
            'name': 'simple',
            'nfold': 3,
            'train_fold': 0,
            'epochs': 100,
            'hidden_Nlayer': 10,
            'hidden_Nnode': 100,
            'lr': 0.03,
            'momentum': 0.8,
            'output': None,
            'activation': 'elu',
            'dropout_rate': '0.0',
            }
        update_dict(self.para_net_sim, inputs)

        self.para_train_Adv = {**self.para_train_sim,
            'name': 'NP',
            'has_syst': True,
            'syssig_h5': '/Users/zhangrui/Work/Code/ML/ANN/h5files/tW_DS_2j2b.h5',
            'syssig_name': 'tW_DS',
            'syssig_tree': 'wt_DS',
            }
        update_dict(self.para_train_Adv, inputs)

        self.para_net_Adv = {**self.para_net_sim,
            'name': 'ANN',
            'epochs': 2,
            'hidden_auxNlayer': 2,
            'hidden_auxNnode': 5,
            'preTrain_epochs': 20,
            'n_iteraction': 100,
            'lam': 10,
        }
        update_dict(self.para_net_Adv, inputs)

        self.wrappe = self.base_directory + self.jobname + '_wrap.sh'
        self.htcjdl = self.base_directory + self.jobname + '_htc.jdl'

        self.req_memory = 6
        self.req_cores = 6

    def create_jdl(self, job_array, local_run = False):
        log = 'log'
        if not os.path.exists('/'.join([self.base_directory, log])):
            os.makedirs('/'.join([self.base_directory, log]))
        def prefix_jdl():
                ''' Steering file for HTCondor (Prefix). '''

                return '\n'.join(l[20:] for l in """
                    Executable              = {wrappe}
                    Universe                = vanilla
                    Transfer_executable     = True
                    Request_memory          = {req_memory}
                    Request_cpus            = {req_cores}
                    # Disk space in kiB, if no unit is specified!
                    Request_disk            = 4 GB
                    +JobFlavour             = "workday"
                    +AccountingGroup        = "group_u_ATLASWISC.all"
                    # Specify job input and output
                    Error                   = {log}/err.$(ClusterId).$(Process)
                    Output                  = {log}/out.$(ClusterId).$(Process)
                    Log                     = {log}/log.$(ClusterId).$(Process)
                    # Additional job requirements (note the plus signs)
                    # Choose OS (options: "SL6", "CentOS7", "Ubuntu1804")
                    +ContainerOS            = "SL6"
                    """.split('\n')[1:]).format(
                        wrappe = self.wrappe,
                        req_memory = '{} GB'.format(self.req_memory),
                        req_cores = str(self.req_cores),
                        log = log)


        def jobarr_jdl(setting):
            ''' Steering file for HTCondor (Job array). '''

            return '\n'.join(l[20:] for l in """
                    
                    # Submit 1 job
                    arguments               = {arguments}
                    Queue 1
                    """.split('\n')[1:]).format(
                        arguments = ' '.join(list(chain(*setting.items()))))

        if not local_run:
            with open(self.htcjdl, 'w+') as f:
                f.write(prefix_jdl())

        def product_dict(**kwargs):
            keys = kwargs.keys()
            vals = kwargs.values()
            for instance in itertools.product(*vals):
                yield dict(zip(keys, instance))
                
        settings = list(product_dict(**job_array))
        for setting in settings:
            if not local_run:
                with open(self.htcjdl, 'a+') as f:
                    f.write(jobarr_jdl(setting))
            elif self.jobname == 'ANN':
                # If local run adversarial neural network
                update_dict(self.para_net_Adv, setting)
                job = JobAdv(**self.para_net_Adv, para_train = self.para_train_Adv)
                job.run()
            else:
                update_dict(self.para_net_sim, setting)
                job = Job(**self.para_net_sim, para_train = self.para_train_sim)
                job.run()

    def create_wrap(self):

        def wrapper():
            ''' Executable file for HTCondor. '''

            return '\n'.join(l[8:] for l in """
        #!/bin/bash
        source ~/.bashrc
        conda activate twaml
        dest={base_directory}
        source $dest../.venvs/twaml-venv-lxpus7/bin/activate
        python {program}/submit.py {mode} _run $*
        \cp -r job__* $dest/
        unset dest
        """.split('\n')[1:]).format(
                base_directory = self.base_directory,
                program = os.path.dirname(os.path.abspath(__file__)),
                mode = self.jobname)

        with open(self.wrappe, 'w+') as f:
            f.write(wrapper())


    def _run(self, param):
        setting = {param[i]: param[i+1] for i in range(0, len(param), 2)}
        if self.jobname == 'ANN':
            update_dict(self.para_net_Adv, setting)
            job = JobAdv(**self.para_net_Adv, para_train = self.para_train_Adv)
            job.run()
        else:
            update_dict(self.para_net_sim, setting)
            job = Job(**self.para_net_sim, para_train = self.para_train_sim)
            job.run()
