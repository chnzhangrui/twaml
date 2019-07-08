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
