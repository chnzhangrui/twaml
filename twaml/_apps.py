"""
twaml command line applications
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from twaml.data import dataset
import logging


def root2pytables():
    log = logging.getLogger("root2pytables")
    """command line application which converts a set of ROOT files into a
    pytables hdf5 file via the ``twaml.data.root_dataset`` function
    and the ``to_pytables`` member function of the
    ``twaml.data.dataset`` class.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert ROOT files to a pytables hdf5 dataset "
            "via twaml.data.root_dataset and "
            "twaml.data.dataset.to_pytables"
        )
    )
    parser.add_argument(
        "-i",
        "--input-files",
        type=str,
        nargs="+",
        required=True,
        help="input ROOT files",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="dataset name (required when reading back into twaml.data.dataset)",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        required=True,
        help="Output h5 file (existing file will be overwritten)",
    )
    parser.add_argument(
        "-b",
        "--branches",
        type=str,
        nargs="+",
        required=False,
        help="branches to save (defaults to all)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        required=False,
        default="WtLoop_nominal",
        help="tree name",
    )
    parser.add_argument(
        "--weight-name",
        type=str,
        required=False,
        default="weight_nominal",
        help="weight branch name",
    )
    parser.add_argument(
        "--extra-weights",
        type=str,
        nargs="+",
        required=False,
        help="extra weights to save",
    )
    parser.add_argument(
        "--true-branches",
        type=str,
        nargs="+",
        required=False,
        help="branches that must be true to pass selection",
    )
    parser.add_argument(
        "--detect-weights",
        action="store_true",
        help="detect weights in the dataset, --extra-weights overrides this",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        required=False,
        help="number of threads to use via ThreadPoolExecutor",
    )

    args = parser.parse_args()

    log.info(
        "Converting the following ROOT files to pytables ({}):".format(args.out_file)
    )
    for f in args.input_files:
        log.info("- {}".format(f))

    sel_dict = None
    if args.true_branches is not None:
        sel_dict = {bn: (np.equal, True) for bn in args.true_branches}
    xtor = ThreadPoolExecutor(args.nthreads) if args.nthreads > 1 else None
    ds = dataset.from_root(
        args.input_files,
        name=args.name,
        tree_name=args.tree_name,
        weight_name=args.weight_name,
        selection=sel_dict,
        branches=args.branches,
        extra_weights=args.extra_weights,
        detect_weights=args.detect_weights,
        executor=xtor,
    )
    ds.to_pytables(args.out_file)

    return 0
