# -*- coding: utf-8 -*-

"""twaml.data module

This module contains classes to abstract datasets using
pandas.DataFrames as the payload for feeding to machine learning
frameworks and other general data investigating

"""

import uproot
import pandas as pd
import h5py
import numpy as np
import re
from pathlib import PosixPath
from typing import List, Dict, Tuple, Optional


class dataset:
    """A class to define a dataset with a pandas.DataFrame as the payload
    of the class. The twaml.data module provides a set of functions to
    construct a dataset. The class constructor should be used only in
    very special cases.

    Attributes
    ----------
    files: List[PosixPath]
      List of files delivering the dataset
    name: str
      Name for the dataset
    tree_name: str
      All of our datasets had to come from a ROOT tree at some point
    weights: numpy.ndarray
      The array of event weights
    df: pandas.DataFrame
      The payload of the class, a dataframe
    addit_weights: Optional[pandas.DataFrame]
      Additional weights to have access too
    label: Optional[int]
      Optional dataset label (as an int)
    has_payload: bool
      Flag to know that the dataset actually wraps data
    cols: List[str]
      Column names as a list of strings
    shape: Tuple
      Shape of the main payload dataframe

    """

    def __init__(
        self,
        input_files: List[str],
        name: Optional[str] = None,
        tree_name: str = "WtLoop_nominal",
        weight_name: str = "weight_nominal",
        label: Optional[int] = None,
    ) -> None:
        """Default dataset creation

        Parameters
        ----------
        input_files: List[str]
          List of input files
        name: Optional[str]
          Name of the dataset (if none use first file name)
        tree_name: str
          Name of tree which this dataset originated from
        weight_name: str
          Name of the weight branch
        label: Optional[int]
          Give dataset an integer based label
        """
        self._weights = np.array([])
        self._df = pd.DataFrame({})
        self._addit_weights = None
        self.files = [PosixPath(f) for f in input_files]
        for f in self.files:
            assert f.exists(), "{} does not exist".format(f)
        if name is None:
            self.name = str(self.files[0].parts[-1])
        else:
            self.name = name
        self.weight_name = weight_name
        self.tree_name = tree_name
        self._label = label

    @property
    def has_payload(self) -> bool:
        has_df = not self._df.empty
        has_weights = self._weights.shape[0] > 0
        return has_df and has_weights

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new: pd.DataFrame) -> None:
        assert len(new) == len(self._weights), "df length != weight length"
        self._df = new

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, new: np.ndarray) -> None:
        assert len(new) == len(self._df), "weight length != frame length"
        self._weights = new

    @property
    def addit_weights(self) -> pd.DataFrame:
        return self._addit_weights

    @addit_weights.setter
    def addit_weights(self, new: pd.DataFrame) -> None:
        if new is not None:
            assert len(new) == len(self._df), "addit_weights length != frame length"
        self._addit_weights = new

    @property
    def label(self) -> Optional[int]:
        return self._label

    @label.setter
    def label(self, new: int) -> None:
        self._label = new

    @property
    def label_array(self) -> Optional[np.ndarray]:
        if self.label is None:
            return None
        return np.ones_like(self.weights, dtype=np.int64) * self.label

    @property
    def cols(self) -> List[str]:
        return list(self.df.columns)

    @property
    def shape(self) -> Tuple:
        """Get shape of dataset (shortcut to pd.DataFrame.shape)"""
        return self.df.shape

    @shape.setter
    def shape(self, new) -> None:
        raise NotImplementedError("Cannot set shape manually")

    def _set_df_and_weights(
        self, df: pd.DataFrame, w: np.ndarray, addit: Optional[pd.DataFrame] = None
    ) -> None:
        assert len(df) == len(w), "unequal length df and weights"
        self._df = df
        self._weights = w
        if addit is not None:
            assert len(df) == len(addit), "unequal length df and addit weights"
            self._addit_weights = addit

    def rm_weight_columns(self) -> None:
        """Remove all payload df columns which begin with ``weight_``

        If you are reading a dataset that was created retaining
        weights in the main payload, this is a useful function to
        remove them. The design of ``twaml.data.dataset`` expects
        weights to be separated from the payload's main dataframe.

        Internally this is done by calling
        ``pd.DataFrame.drop(..., inplace=True)`` on the payload

        """
        import re

        pat = re.compile("^weight_")
        rmthese = [c for c in self._df.columns if re.match(pat, c)]
        self._df.drop(columns=rmthese, inplace=True)

    def rm_columns_re(self, cols: List[str]) -> None:
        """Remove some columns from the payload based on regex paterns

        Uses ``pd.DataFrame.drop(..., inplace=True)``.

        Parameters
        ----------
        cols : List[str]
          List of regex paterns
        """
        import re

        for rm in cols:
            pat = re.compile(rm)
            rmthese = [c for c in self._df.columns if re.search(pat, c)]
            self._df.drop(columns=rmthese, inplace=True)

    def change_weights(self, wname: str) -> None:
        """Change the main weight of the dataset

        this function will swap the current main weight array of the
        dataset with one in the addit_weights frame.
        """
        assert self._addit_weights is not None, "additional weights do not exist"

        old_name = self.weight_name
        old_weights = self.weights
        self._addit_weights[old_name] = old_weights

        self.weights = self._addit_weights[wname].values
        self.weight_name = wname

        self._addit_weights.drop(columns=[wname], inplace=True)

    def append(self, other: "dataset") -> None:

        """Append a dataset to an exiting one

        We perform concatenations of the dataframes and weights to
        update the existing dataset's payload.

        if one dataset has additional weights and the other doesn't,
        the additional weights are dropped.

        Parameters
        ----------
        other : twanaet.data.dataset
          The dataset to append

        """
        assert self.has_payload, "Unconstructed df (self)"
        assert other.has_payload, "Unconstructed df (other)"
        assert self.weight_name == other.weight_name, "different weight names"
        assert self.shape[1] == other.shape[1], "different df columns"

        if self.addit_weights is not None and other.addit_weights is not None:
            assert (
                self.addit_weights.shape[1] == other.addit_weights.shape[1]
            ), "additional weights are different lengths"

        self._df = pd.concat([self._df, other.df])
        self._weights = np.concatenate([self._weights, other.weights])
        self.files = self.files + other.files

        if self.addit_weights is not None and other.addit_weights is not None:
            self._addit_weights = pd.concat([self._addit_weights, other.addit_weights])
        else:
            self._addit_weights = None

    def to_pytables(self, file_name: str) -> None:
        """Write payload to disk as an pytables h5 file with strict options

        The key in the file is the name of the dataset. The weights
        array is stored as a separate frame with the key being the
        weight_name attribute. If additional weights are present they
        are saved as well.

        An existing dataset label **is not stored**.

        Parameters
        ----------
        file_name : str
          output file name,

        """
        weights_frame = pd.DataFrame(dict(weights=self._weights))
        self._df.to_hdf(file_name, self.name, mode="w")
        weights_frame.to_hdf(file_name, self.weight_name, mode="a")
        if self._addit_weights is not None:
            self._addit_weights.to_hdf(
                file_name, "{}_addit_weights".format(self.name), mode="a"
            )

    def __add__(self, other: "dataset") -> "dataset":
        """Add two datasets together

        We perform concatenations of the dataframes and weights to
        generate a new dataset with the combined a new payload.

        if one dataset has additional weights and the other doesn't,
        the additional weights are dropped.

        """
        assert self.has_payload, "Unconstructed df (self)"
        assert other.has_payload, "Unconstructed df (other)"
        assert self.weight_name == other.weight_name, "different weight names"
        assert self.shape[1] == other.shape[1], "different df columns"

        if self.addit_weights is not None and other.addit_weights is not None:
            assert (
                self.addit_weights.shape[1] == other.addit_weights.shape[1]
            ), "additional weights are different lengths"

        new_weights = np.concatenate([self.weights, other.weights])
        new_df = pd.concat([self.df, other.df])
        new_files = [str(f) for f in (self.files + other.files)]
        new_ds = dataset(
            new_files,
            self.name,
            weight_name=self.weight_name,
            tree_name=self.tree_name,
            label=self.label,
        )

        if self.addit_weights is not None and other.addit_weights is not None:
            new_aw = pd.concat([self.addit_weights, other.addit_weights])
        else:
            new_aw = None

        new_ds._set_df_and_weights(new_df, new_weights, addit=new_aw)
        return new_ds

    def __len__(self) -> int:
        """length of the dataset"""
        return len(self.weights)

    def __repr__(self) -> str:
        """standard repr"""
        return "<twaml.data.dataset(name={}, shape={})>".format(self.name, self.shape)

    def __str__(self) -> str:
        """standard str"""
        return "dataset(name={})".format(self.name)


def root_dataset(
    input_files: List[str],
    name: Optional[str] = None,
    tree_name: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: List[str] = None,
    selection: Dict = None,
    label: Optional[int] = None,
    allow_weights_in_df: bool = False,
    addit_weights: Optional[List[str]] = None,
    detect_weights: bool = False,
) -> dataset:
    """Create a ROOT dataset

    Parameters
    ----------
    input_files: List[str]
        List of ROOT input_files to use
    name: str
        Name of the dataset (if none use first file name)
    tree_name: str
        Name of the tree in the file to use
    weight_name: str
        Name of the weight branch
    branches: List[str]
        List of branches to store in the dataset, if None use all
    selection: Dict
        A dictionary of selections to apply of the form:
        ``{branch_name: (numpy.ufunc, test_value)}``. the
        selections are combined using ``np.logical_and``
    label: Optional[int]
        Give the dataset an integer label
    allow_weights_in_df: bool
        Allow "^weight_" branches in the payload dataframe
    addit_weights: Optional[List[str]]
        Additional weights to store in a second dataframe.
    detect_weights: bool
        If True, fill the addit_weights df with all "^weight_"
        branches If ``addit_weights`` is not None, this option is
        ignored.

    Examples
    --------
    Example with a single file and two branches:

    >>> ds1 = root_dataset(['file.root'], name='myds',
    ...                    branches=['pT_lep1', 'pT_lep2'], label=1)

    Example with multiple input_files and a selection (uses all
    branches). The selection requires the branch ``nbjets == 1``
    and ``njets >= 1``, then label it 5.

    >>> flist = ['file1.root', 'file2.root', 'file3.root']
    >>> ds = root_dataset(flist, selection={'nbjets': (np.equal, 1),
    ...                                     'njets': (np.greater, 1)}
    >>> ds.label = 5

    Example using additional weights

    >>> ds = root_dataset(flist, name='myds', weight_name='weight_nominal',
    ...                   addit_weights=['weight_sys_radLo', ' weight_sys_radHi'])


    Example where we detect additional weights automatically

    >>> ds = root_dataset(flist, name='myds', weight_name='weight_nominal',
    ...                   detect_weights=True)

    """
    ds = dataset(
        input_files, name, tree_name=tree_name, weight_name=weight_name, label=label
    )

    uproot_trees = [uproot.open(file_name)[tree_name] for file_name in input_files]

    wpat = re.compile("^weight_")
    if addit_weights is not None:
        w_branches = addit_weights
    elif detect_weights:
        urtkeys = [k.decode("utf-8") for k in uproot_trees[0].keys()]
        w_branches = [k for k in urtkeys if re.match(wpat, k)]
        if weight_name in w_branches:
            w_branches.remove(weight_name)
    else:
        w_branches = None

    weight_list, frame_list, addit_frame_list = [], [], []
    for t in uproot_trees:
        raw_w = t.array(weight_name)
        raw_f = t.pandas.df(branches=branches, namedecode="utf-8")
        if not allow_weights_in_df:
            rmthese = [c for c in raw_f.columns if re.match(wpat, c)]
            raw_f.drop(columns=rmthese, inplace=True)

        if w_branches is not None:
            raw_aw = t.pandas.df(branches=w_branches, namedecode="utf-8")

        isel = np.ones((raw_w.shape[0]), dtype=bool)
        if selection is not None:
            selections = {k: v[0](t.array(k), v[1]) for k, v in selection.items()}
            for k, v in selections.items():
                isel = np.logical_and(isel, v)
        weight_list.append(raw_w[isel])
        frame_list.append(raw_f[isel])
        if w_branches is not None:
            addit_frame_list.append(raw_aw[isel])

    weights_array = np.concatenate(weight_list)
    df = pd.concat(frame_list)
    if w_branches is not None:
        aw_df = pd.concat(addit_frame_list)
    else:
        aw_df = None

    ds._set_df_and_weights(df, weights_array, addit=aw_df)

    return ds


def pytables_dataset(
    file_name: str,
    name: str,
    tree_name: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    label: Optional[int] = None,
) -> dataset:
    """Create an h5 dataset from pytables output generated from
    dataset.to_pytables

    The payload is extracted from the .h5 pytables files using the
    name of the dataset and the weight name. If the name of the
    dataset doesn't exist in the file you'll crash. Additional weights
    are retrieved if available.

    Parameters
    ----------
    file_name: str
        Name of h5 file containing the payload
    name: str
        Name of the dataset inside the h5 file
    tree_name: str
        Name of tree where dataset originated
    weight_name: str
        Name of the weight array inside the h5 file
    label: Optional[int]
        Give the dataset an integer label

    Examples
    --------

    >>> ds1 = pytables_dataset('ttbar.h5', 'ttbar')
    >>> ds1.label = 1 ## add label dataset after the fact

    """
    main_frame = pd.read_hdf(file_name, name)
    main_weight_frame = pd.read_hdf(file_name, weight_name)
    h5f = h5py.File(file_name, "r")
    if "{}_addit_weights".format(name) in h5f:
        addit_frame = pd.read_hdf(file_name, "{}_addit_weights".format(name))
    else:
        addit_frame = None
    w_array = main_weight_frame.weights.values
    ds = dataset(
        [file_name], name, weight_name=weight_name, tree_name=tree_name, label=label
    )
    ds._set_df_and_weights(main_frame, w_array, addit=addit_frame)
    return ds


def h5_dataset(
    file_name: str,
    name: str,
    columns: List[str],
    tree_name: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    label: Optional[int] = None,
) -> dataset:
    """Create a dataset from generic h5 input (loosely expected to be from
    the ATLAS Analysis Release utility ``ttree2hdf5``

    The name of the HDF5 dataset inside the file is assumed to be
    ``tree_name``. The ``name`` argument is something *you choose*.

    Parameters
    ----------
    file_name: str
        Name of h5 file containing the payload
    name: str
        Name of the dataset you would like to define
    columns: List[str]
        Names of columns (branches) to include in payload
    tree_name: str
        Name of tree dataset originates from (HDF5 dataset name)
    weight_name: str
        Name of the weight array inside the h5 file
    label: Optional[int]
        Give the dataset an integer label

    """
    ds = dataset(
        [file_name],
        name=name,
        weight_name=weight_name,
        tree_name=tree_name,
        label=label,
    )

    """Construct the payload from the h5 files"""
    f = h5py.File(file_name, mode="r")
    full_ds = f[tree_name]
    w_array = f[tree_name][weight_name]
    coldict = {}
    for col in columns:
        coldict[col] = full_ds[col]
    frame = pd.DataFrame(coldict)
    ds._set_df_and_weights(frame, w_array)
    return ds


def scale_weight_sum(to_update: "dataset", reference: "dataset") -> None:
    """
    Scale the weights of the `to_update` dataset such that the sum of
    weights are equal to the sum of weights of the `reference` dataset.

    Parameters
    ----------
    to_update : twanet.data.dataset
        dataset with weights to be scaled
    reference : twanet.data.dataset
        dataset to scale to

    """
    assert to_update.has_payload, "{} is without payload".format(to_update)
    assert reference.has_payload, "{} is without payload".format(reference)
    sum_to_update = to_update.weights.sum()
    sum_reference = reference.weights.sum()
    to_update.weights *= sum_reference / sum_to_update
