Getting Started
===============

.. toctree::
   :maxdepth: 2

Requirements
------------
Run on lxplus7
Create HTCondor job scripts:

.. code-block:: none

    $ python ../twaml/advnn/submit.py DNN htc

Submit to lxplus HTCondor:

.. code-block:: none

    $ condor_submit DNN_htc.jdl

Local run:

.. code-block:: none

    $ source DNN_wrap.sh hidden_Nlayer 5 hidden_Nnode 30 lr 0.01 activation elu

Inference:

.. code-block:: none

    $ py apply.py --input ../cernbox/twaml/data_zero_jet.root --output out_data_zero_jet.root --json ../cernbox/twaml/job__l5n50_lr0.001mom0.8_elu_k3_dp0.5_e100/Train/0j_model.json  --h5 ../cernbox/twaml/job__l5n50_lr0.001mom0.8_elu_k3_dp0.5_e100/Train/0j_model_0010.h5 --pkl ../cernbox/twaml/job__l5n50_lr0.001mom0.8_elu_k3_dp0.5_e100/Train/0j_event.pkl
