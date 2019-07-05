Command Line Applications
=========================

.. toctree::
   :maxdepth: 2

.. currentmodule:: twaml._apps

``twaml-root2pytables``
-----------------------

Convert a set of ROOT files into a single pytables HDF5 file.

.. command-output:: twaml-root2pytables --help

An example that uses the default ``--tree-name`` and
``--weight-name``, while only saving the branches ``b1`` and ``b2``
and requiring the branch ``elmu`` to be true and ``pT_lep1`` to be
greater than 50 to save the event.

.. code-block:: none

   $ twaml-root2pytables -i file.root -o file.h5 --branches b1 b2 \
         --selection "(elmu == True) & (pT_lep1 > 50)"

The docs for the function that is being called:

.. autofunction:: root2pytables
