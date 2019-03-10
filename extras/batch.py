from job import Job, JobAdv
import itertools    
from itertools import chain

class Batch(object):
    def __init__(self, base_directory, inputs):
        self.base_directory = base_directory + '/'
        self.para_train_sim = {'name': '2j2b',
            'signal_h5': 'tW_DR_2j2b.h5',
            'signal_name': 'tW_DR',
            'signal_tree': 'wt_DR_nominal',
            'backgd_h5': 'ttbar_2j2b.h5',
            'backgd_name': 'ttbar',
            'backgd_tree': 'tt_nominal',
            'weight_name': 'weight_nominal',
            'variables': ['mass_lep1jet2'],
            }
        self.para_train_sim.update(inputs)

        self.para_net_sim = {
            'name': 'simple',
            'nfold': 3,
            'train_fold': 0,
            'epochs': 500,
            'hidden_Nlayer': 10,
            'hidden_Nnode': 100,
            'lr': 0.03,
            'momentum': 0.8,
            'output': None,
            'activation': 'elu',
            }

        self.wrappe = self.base_directory + 'wrap.sh'
        self.htcjdl = self.base_directory + 'htc.jdl'

        self.req_memory = 4
        self.req_cores = 4

    def create_jdl(self, job_array, local_run = False):

        def prefix_jdl():
                ''' Steering file for HTCondor (Prefix). '''

                return '\n'.join(l[8:] for l in """
        Executable              = {wrappe}
        Universe                = vanilla
        Transfer_executable     = True
        Request_memory          = {req_memory}
        Request_cpus            = {req_cores}
        # Disk space in kiB, if no unit is specified!
        Request_disk            = 2 GB
        # Additional job requirements (note the plus signs)
        # Choose OS (options: "SL6", "CentOS7", "Ubuntu1804")
        +ContainerOS            = "SL6"
        """.split('\n')[1:]).format(
                    wrappe = self.wrappe,
                    req_memory = '{} GB'.format(self.req_memory),
                    req_cores = str(self.req_cores))


        def jobarr_jdl(setting):
            ''' Steering file for HTCondor (Job array). '''

            return '\n'.join(l[8:] for l in """
            
        # Submit 1 job
        arguments               = {arguments}
        Queue 1
        """.split('\n')[1:]).format(
                    arguments = ' '.join(list(chain(*setting.items()))))

        print('zhang', local_run, self.htcjdl)
        if not local_run:
            print('zhang write')
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
            else:
                # If local run
                self.para_net_sim.update(setting)
                job = Job(**self.para_net_sim, para_train = self.para_train_sim)
                job.run()

    def create_wrap(self):

        def wrapper():
            ''' Executable file for HTCondor. '''

            return '\n'.join(l[8:] for l in """
        #!/bin/bash

        python ../twaml/extras/submit.py _run $*
            """.split('\n')[1:]).format('')

        with open(self.wrappe, 'w+') as f:
            f.write(wrapper())


    def _run(self, param):
        setting = {param[i]: param[i+1] for i in range(0, len(param), 2)}
        self.para_net_sim.update(setting)
        job = Job(**self.para_net_sim, para_train = self.para_train_sim)
        job.run()