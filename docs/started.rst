Getting Started
===============

.. toctree::
   :maxdepth: 2

Requirements
------------

It is highy recommended to use the `Anaconda
<https://www.anaconda.com/>`_ python distribution. Most of the
required libraries (outside of the bleeding edge machine learning
packages) will be included with Anaconda environments.

The simplest way to get up and running with ``twaml`` is to use the ``environment.yml`` file.

.. code-block:: none

   $ cd /path/to/twaml
   $ conda env create -f environment.yml
   $ conda activate twaml

The bare requirements for data handling, plotting, and testing include
(enforced by ``requirements.txt``, see file for verions):

- uproot
- pandas
- scikit-learn
- matplotlib
- h5py
- pytables
- numexpr (to ensure pandas.eval acceleration)

Since twaml adopts "`live at the head
<https://www.youtube.com/watch?v=tISy7EJQPzI>`_", requirement versions
may be fluid.

For training and testing models (not enforced by ``requirements.txt``)
you'll probably want:

- tensorflow
- pytorch
- xgboost

For building documentation

- sphinx
- sphinx_rtd_theme
- sphinx-autodoc-typehints
- sphinxcontrib-programoutput


Base Setup in a venv
--------------------

A base setup without the machine learning libraries just requires a
pip installation of the ``twaml``.

.. code-block:: none

   $ python3 -m venv ~/.venvs/twaml-venv
   $ source ~/.venvs/twaml-venv/bin/activate
   $ cd /path/to/twaml
   $ pip install .

This will make the ``twaml.data`` and ``twaml.viz`` APIs accessible.


Example GPU Anaconda Setup
--------------------------

Start with a fresh Anaconda virtual environment:

.. code-block:: none

   $ cd /path/to/twaml
   $ conda env create -f environment.yml
   $ conda activate twaml
   $ conda install pytorch torchvision cuda100 -c pytorch ## requires recent nvidia linux drivers
   $ conda install tensorflow-gpu ## or just tensorflow
   $ pip install .
