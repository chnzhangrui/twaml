"""
This module contains a class (and functions to load it) which
abstracts datasets using ``pandas.DataFrames`` as the payload for
feeding to machine learning frameworks and other general data
investigating.
"""

import re
from pathlib import PosixPath
from typing import List, Tuple, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging
import uproot
import pandas as pd
import h5py
import numpy as np
import yaml

log = logging.getLogger(__name__)

__all__ = ["dataset", "from_root", "from_pytables", "from_h5", "scale_weight_sum"]


class dataset:
    """A class to define a dataset with a :class:`pandas.DataFrame` as the
    payload of the class. The ``twaml.data`` module provides a set of
    static functions to construct a dataset. The class constructor
    should be used only in very special cases.

    ``datasets`` should `always` be constructed using one of three functions:

      - :meth:`from_root`
      - :meth:`from_pytables`
      - :meth:`from_h5`

    Attributes
    ----------
    name: str
      Name for the dataset
    weight_name: str
      Name of the branch which the weight array originates from
    tree_name: str
      All of our datasets had to come from a ROOT tree at some point
    selection_formula: Optional[str]
      A string (in :meth:`pandas.DataFrame.eval` form) that all of data in
      the dataset had to satisfy
    label: Optional[int]
      Optional dataset label (as an int)
    auxlabel: Optional[int]
      Optional auxiliary label (as an int) - sometimes we need two labels
    TeXlabel: Optional[str]
      LaTeX formatted name (for plot labels)
    """

    name = None
    weight_name = None
    tree_name = None
    selection_formula = None
    label = None
    auxlabel = None
    TeXlabel = None

    _weights = None
    _df = None
    _auxweights = None
    _files = None
    _wtloop_metas = None

    def _init_skeleton(
        self,
        input_files: List[str],
        name: Optional[str] = None,
        tree_name: str = "WtLoop_nominal",
        weight_name: str = "weight_nominal",
        label: Optional[int] = None,
        auxlabel: Optional[int] = None,
        TeXlabel: Optional[str] = None,
    ) -> None:
        """Default initialization - should only be called by internal
        staticmethods ``from_root``, ``from_pytables``, ``from_h5``

        Parameters
        ----------
        input_files:
          List of input files
        name:
          Name of the dataset (if none use first file name)
        tree_name:
          Name of tree which this dataset originated from
        weight_name:
          Name of the weight branch
        label:
          Give dataset an integer based label
        auxlabel:
          Give dataset an integer based auxiliary label
        TeXlabel:
          LaTeX form label
        """
        self._files = [PosixPath(f).resolve(strict=True) for f in input_files]
        if name is None:
            self.name = str(self.files[0].parts[-1])
        else:
            self.name = name
        self.weight_name = weight_name
        self.tree_name = tree_name
        self.label = label
        self.auxlabel = auxlabel
        self.TeXlabel = TeXlabel

    def has_payload(self) -> bool:
        """check if dataframe and weights are non empty"""
        has_df = not self._df.empty
        has_weights = self._weights.shape[0] > 0
        return has_df and has_weights

    @property
    def files(self) -> List[PosixPath]:
        """list of files which make up the dataset"""
        return self._files

    @files.setter
    def files(self, new) -> None:
        self._files = new

    @property
    def df(self) -> pd.DataFrame:
        """the payload of the dataset class"""
        return self._df

    @df.setter
    def df(self, new: pd.DataFrame) -> None:
        assert len(new) == len(self._weights), "df length != weight length"
        self._df = new

    @property
    def weights(self) -> np.ndarray:
        """array of event weights"""
        return self._weights

    @weights.setter
    def weights(self, new: np.ndarray) -> None:
        assert len(new) == len(self._df), "weight length != frame length"
        self._weights = new

    @property
    def auxweights(self) -> pd.DataFrame:
        """dataframe of auxiliary event weights"""
        return self._auxweights

    @auxweights.setter
    def auxweights(self, new: pd.DataFrame) -> None:
        if new is not None:
            assert len(new) == len(self._df), "auxweights length != frame length"
        self._auxweights = new

    @property
    def shape(self) -> Tuple:
        """shape of dataset (n events, n features)"""
        return self.df.shape

    @shape.setter
    def shape(self, new) -> None:
        raise NotImplementedError("Cannot set shape manually")

    @property
    def wtloop_metas(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """dictionary of metadata information (one for each file making up the dataset)"""
        return self._wtloop_metas

    @wtloop_metas.setter
    def wtloop_metas(self, new) -> None:
        self._wtloop_metas = new

    @property
    def initial_state(self) -> str:
        """retrieve initial state from the wtloop_metas information (if available)

        This will return 'unknown' if wtloop_metas information is
        unavailable, or a set if multiple different initial states are
        found.
        """
        if self.wtloop_metas is None:
            return "unknown"

        init_states = set()
        for _, v in self.wtloop_metas.items():
            init_states.add(v["initial_state"])
        if len(init_states) == 1:
            for elem in init_states:
                return elem
        else:
            return init_states

    @property
    def dsid(self) -> int:
        """retrieve the DSID from the wtloop_metas information (if available)

        This will return 999999 if wtloop_metas information is
        unavailable, or a set if multiple different DSIDs are found.
        """
        if self.wtloop_metas is None:
            return 999999

        dsids = set()
        for _, v in self.wtloop_metas.items():
            dsids.add(v["dsid"])
        if len(dsids) == 1:
            for elem in dsids:
                return elem
        else:
            return dsids

    def label_asarray(self) -> Optional[np.ndarray]:
        """retrieve a homogenuous array of labels (or ``None``) if no label"""
        if self.label is None:
            return None
        return np.ones_like(self.weights, dtype=np.int64) * self.label

    def auxlabel_asarray(self) -> Optional[np.ndarray]:
        """retrieve a homogenous array of auxiliary labels (or ``None``) if no auxlabel"""
        if self.auxlabel is None:
            return None
        return np.ones_like(self.weights, dtype=np.int64) * self.auxlabel

    def __len__(self) -> int:
        """length of the dataset"""
        return len(self.weights)

    def __repr__(self) -> str:
        """standard repr"""
        return f"<twaml.data.dataset(name={self.name}, shape={self.shape})>"

    def __str__(self) -> str:
        """standard str"""
        return f"dataset(name={self.name})"

    def __getitem__(self, idx) -> "dataset":
        """get subset based on boolean mask or array of indices"""
        new_df = self._df[idx]
        new_w = self._weights[idx]
        if self._auxweights is not None:
            new_aw = self._auxweights[idx]
        else:
            new_aw = None
        new_ds = dataset()
        new_ds._init_skeleton(
            self.files,
            self.name,
            weight_name=self.weight_name,
            tree_name=self.tree_name,
            label=self.label,
            auxlabel=self.auxlabel,
            TeXlabel=self.TeXlabel,
        )
        new_ds.wtloop_metas = self.wtloop_metas
        new_ds._set_df_and_weights(new_df, new_w, auxw=new_aw)
        return new_ds

    def _set_df_and_weights(
        self, df: pd.DataFrame, w: np.ndarray, auxw: Optional[pd.DataFrame] = None
    ) -> None:
        assert len(df) == len(w), "unequal length df and weights"
        self._df = df
        self._weights = w
        if auxw is not None:
            assert len(df) == len(auxw), "unequal length df and auxw weights"
            self._auxweights = auxw

    @staticmethod
    def _combine_wtloop_metas(meta1, meta2) -> Optional[dict]:
        if meta1 is not None and meta2 is not None:
            return {**meta1, **meta2}
        if meta1 is None and meta2 is not None:
            return {**meta2}
        if meta1 is not None and meta2 is None:
            return {**meta1}
        return None

    def keep_columns(self, cols: List[str]) -> None:
        """
        Drop all columns not included in ``cols``

        Parameters
        ----------
        cols: List[str]
          Columns to keep
        """
        self._df = self._df[cols]

    def aggressively_strip(self) -> None:
        """Drop all columns that should never be used in a classifier.

        This calls the following functions:

        - :meth:`rm_meta_columns`
        - :meth:`rm_region_columns`
        - :meth:`rm_chargeflavor_columns`
        - :meth:`rm_weight_columns`
        """
        self.rm_meta_columns()
        self.rm_region_columns()
        self.rm_chargeflavor_columns()
        self.rm_weight_columns()

    def rm_meta_columns(self) -> None:
        """Drop all columns are are considered meta data from the payload

        This includes runNumber, eventNumber, randomRunNumber

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload.
        """
        self.df.drop(columns=["runNumber", "randomRunNumber", "eventNumber"], inplace=True)

    def rm_region_columns(self) -> None:
        """Drop all columns that are prefixed with ``reg``, like ``reg2j2b``

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload.
        """
        rmthese = [c for c in self._df.columns if re.match(r"^reg[0-9]\w+", c)]
        self._df.drop(columns=rmthese, inplace=True)

    def rm_chargeflavor_columns(self) -> None:
        """Drop all columns that are related to charge and flavor

        This would be [elmu, elel, mumu, OS, SS]
        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload.
        """
        self.df.drop(columns=["OS", "SS", "elmu", "elel", "mumu"], inplace=True)

    def rm_weight_columns(self) -> None:
        """Remove all payload df columns which begin with ``weight_``

        If you are reading a dataset that was created retaining
        weights in the main payload, this is a useful function to
        remove them. The design of ``twaml.data.dataset`` expects
        weights to be separated from the payload's main dataframe.

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload

        """
        rmthese = [c for c in self._df.columns if re.match(r"^weight_\w+", c)]
        self._df.drop(columns=rmthese, inplace=True)

    def rm_columns(self, cols: List[str]) -> None:
        """Remove columns from the dataset

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload

        Parameters
        ----------
        cols: List[str]
          List of column names to remove

        """
        self._df.drop(columns=cols, inplace=True)

    def keep_weights(self, weights: List[str]) -> None:
        """Drop all columns from the aux weights frame that are not in ``weights``

        Parameters
        ----------
        weights: List[str]
          Weights to keep in the aux weights frame
        """
        self._auxweights = self._auxweights[weights]

    def change_weights(self, wname: str) -> None:
        """Change the main weight of the dataset

        this function will swap the current main weight array of the
        dataset with one in the ``auxweights`` frame (based on its
        name in the ``auxweights`` frame).

        Parameters
        ----------
        wname:
          name of weight in ``auxweight`` DataFrame to turn into the main weight.

        """
        assert self._auxweights is not None, "aux weights do not exist"

        old_name = self.weight_name
        old_weights = self.weights
        self._auxweights[old_name] = old_weights

        self.weights = self._auxweights[wname].to_numpy()
        self.weight_name = wname

        self._auxweights.drop(columns=[wname], inplace=True)

    def append(self, other: "dataset") -> None:
        """Append a dataset to an exiting one

        We perform concatenations of the dataframes and weights to
        update the existing dataset's payload.

        if one dataset has aux weights and the other doesn't,
        the aux weights are dropped.

        Parameters
        ----------
        other : twaml.data.dataset
          The dataset to append

        """
        assert self.has_payload, "Unconstructed df (self)"
        assert other.has_payload, "Unconstructed df (other)"
        assert self.weight_name == other.weight_name, "different weight names"
        assert self.shape[1] == other.shape[1], "different df columns"

        if self._auxweights is not None and other.auxweights is not None:
            assert (
                self._auxweights.shape[1] == other.auxweights.shape[1]
            ), "aux weights are different lengths"

        self._df = pd.concat([self._df, other.df])
        self._weights = np.concatenate([self._weights, other.weights])
        self._files = self._files + other._files
        self._wtloop_metas = self._combine_wtloop_metas(
            self._wtloop_metas, other._wtloop_metas
        )

        if self._auxweights is not None and other.auxweights is not None:
            self._auxweights = pd.concat([self._auxweights, other.auxweights])
        else:
            self._auxweights = None

    def to_pytables(self, file_name: str, to_hdf_kw: Optional[Dict[str, Any]] = None) -> None:
        """Write dataset to disk as a pytables h5 file

        This method saves a file using a strict twaml-compatible
        naming scheme. An existing dataset label **is not
        stored**. The properties of the class that are serialized to
        disk (and the associated key for each item):

        - ``df`` as ``{name}_payload``
        - ``weights`` as ``{name}_{weight_name}``
        - ``auxweights`` as ``{name}_auxweights``
        - ``wtloop_metas`` as ``{name}_wtloop_metas``

        These properties are wrapped in a pandas DataFrame (if they
        are not already) to be stored in a .h5 file. The
        :meth:`from_pytables` is designed to read in this output; so
        the standard use case is to call this function to store a
        dataset that was intialized via :meth:`from_root`.

        Internally this function uses :meth:`pandas.DataFrame.to_hdf`
        on a number of structures.

        Parameters
        ----------
        file_name:
          output file name,
        format:
          dict of keyword arguments fed to :meth:`pd.DataFrame.to_hdf`

        Examples
        --------

        >>> ds = twaml.dataset.from_root("file.root", name="myds",
        ...                              detect_weights=True, wtloop_metas=True)
        >>> ds.to_pytables("output.h5")
        >>> ds_again = twaml.dataset.from_pytables("output.h5")
        >>> ds_again.name
        'myds'

        """
        if to_hdf_kw is None:
            to_hdf_kw = {}
        log.info(f"Creating pytables dataset with name '{self.name}' in {file_name}")
        log.info(f"  selection used: '{self.selection_formula}'")
        log.info(f"  original source was:")
        for fname in self.files:
            log.info(f"   - {fname}")
        if PosixPath(file_name).exists():
            log.warning(f"{file_name} exists, overwriting")
        weights_frame = pd.DataFrame(dict(weights=self._weights))
        self._df.to_hdf(file_name, f"{self.name}_payload", mode="w", **to_hdf_kw)
        weights_frame.to_hdf(file_name, f"{self.name}_{self.weight_name}", mode="a")
        if self._auxweights is not None:
            self._auxweights.to_hdf(file_name, f"{self.name}_auxweights", mode="a")
        if self.wtloop_metas is not None:
            tempdict = {k: np.array([str(v)]) for k, v in self.wtloop_metas.items()}
            wtmetadf = pd.DataFrame.from_dict(tempdict)
            wtmetadf.to_hdf(file_name, f"{self.name}_wtloop_metas", mode="a")

    def __add__(self, other: "dataset") -> "dataset":
        """Add two datasets together

        We perform concatenations of the dataframes and weights to
        generate a new dataset with the combined a new payload.

        if one dataset has aux weights and the other doesn't,
        the aux weights are dropped.

        """
        assert self.has_payload, "Unconstructed df (self)"
        assert other.has_payload, "Unconstructed df (other)"
        assert self.weight_name == other.weight_name, "different weight names"
        assert self.shape[1] == other.shape[1], "different df columns"

        if self._auxweights is not None and other.auxweights is not None:
            assert (
                self._auxweights.shape[1] == other.auxweights.shape[1]
            ), "aux weights are different lengths"

        new_weights = np.concatenate([self.weights, other.weights])
        new_df = pd.concat([self.df, other.df])
        new_files = [str(f) for f in (self.files + other.files)]
        new_ds = dataset()
        new_ds._init_skeleton(
            new_files,
            self.name,
            weight_name=self.weight_name,
            tree_name=self.tree_name,
            label=self.label,
            auxlabel=self.auxlabel,
            TeXlabel=self.TeXlabel,
        )
        new_ds.wtloop_metas = self._combine_wtloop_metas(
            self.wtloop_metas, other.wtloop_metas
        )

        if self._auxweights is not None and other.auxweights is not None:
            new_aw = pd.concat([self._auxweights, other.auxweights])
        else:
            new_aw = None

        new_ds._set_df_and_weights(new_df, new_weights, auxw=new_aw)
        return new_ds

    def selection_masks(
        self, selections: Dict[str, str]
    ) -> Tuple[Dict[str, "np.ndarray"], Tuple[str]]:
        """Based on a dictionary of selections, calculate masks (boolean
        ararys) for each selection

        Parameters
        ----------
        selections:
          Dictionary of selections in the form ``{ name : selection }``.

        """
        masks = {}
        selstrs = []
        for sel_key, sel_val in selections.items():
            masks[sel_key] = np.asarray(self.df.eval(sel_val))
            selstrs.append(sel_val)
        return masks, tuple(selstrs)

    def selected_datasets(self, selections: Dict[str, str]) -> Dict[str, "dataset"]:
        """Based on a dictionary of selections, break the dataset into a set
        of multiple (finer grained) datasets.

        Warnings
        --------
        For large datasets this can get memory intensive quickly. A
        good alternative is :meth:`selection_masks` combined with the
        ``__getitem__`` implementation.

        Parameters
        ----------
        selections:
          Dictionary of selections in the form ``{ name : selection }``.

        Examples
        --------

        A handful of selections with all requiring ``OS`` and ``elmu``
        to be true, while changing the ``reg{...}`` requirement.

        >>> selections = { '1j1b' : '(reg1j1b == True) & (OS == True) & (elmu == True)',
        ...                '2j1b' : '(reg2j1b == True) & (OS == True) & (elmu == True)',
        ...                '2j2b' : '(reg2j2b == True) & (OS == True) & (elmu == True)',
        ...                '3j1b' : '(reg3j1b == True) & (OS == True) & (elmu == True)'}
        >>> selected_datasets = ds.selected_datasets(selections)

        """
        breaks = {}
        for sel_key, sel_val in selections.items():
            mask = self.df.eval(sel_val)
            new_df = self.df[mask]
            new_weights = self.weights[mask]
            new_auxweights = None
            if self.auxweights is not None:
                new_auxweights = self.auxweights[mask]
            new_meta = self.wtloop_metas
            new_ds = dataset()
            new_ds._init_skeleton(
                self.files,
                self.name,
                self.tree_name,
                self.weight_name,
                label=self.label,
                auxlabel=self.auxlabel,
                TeXlabel=self.TeXlabel,
            )
            new_ds._set_df_and_weights(new_df, new_weights, new_auxweights)
            new_ds.wtloop_metas = new_meta
            new_ds.selection_formula = sel_val
            breaks[sel_key] = new_ds

        return breaks


def from_root(
    input_files: Union[str, List[str]],
    name: Optional[str] = None,
    tree_name: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: List[str] = None,
    selection: Optional[str] = None,
    label: Optional[int] = None,
    auxlabel: Optional[int] = None,
    allow_weights_in_df: bool = False,
    aggressively_strip: bool = False,
    auxweights: Optional[List[str]] = None,
    detect_weights: bool = False,
    nthreads: Optional[int] = None,
    wtloop_meta: bool = False,
    TeXlabel: Optional[str] = None,
) -> "dataset":
    """
    Initialize a dataset from ROOT files

    Parameters
    ----------
    input_files:
        Single or list of ROOT input file(s) to use
    name:
        Name of the dataset (if none use first file name)
    tree_name:
        Name of the tree in the file to use
    weight_name:
        Name of the weight branch
    branches:
        List of branches to store in the dataset, if ``None`` use all
    selection:
        A string passed to pandas.DataFrame.eval to apply a selection
        based on branch/column values. e.g. ``(reg1j1b == True) & (OS == True)``
        requires the ``reg1j1b`` and ``OS`` branches to be ``True``.
    label:
        Give the dataset an integer label
    auxlabel:
        Give the dataset an integer auxiliary label
    allow_weights_in_df:
        Allow "^weight_\\w+" branches in the payload dataframe
    aggressively_strip:
        Call :meth:`twaml.data.dataset.aggressively_strip` during construction
    auxweights:
        Auxiliary weights to store in a second dataframe.
    detect_weights:
        If True, fill the auxweights df with all "^weight_"
        branches If ``auxweights`` is not ``None``, this option is
        ignored.
    nthreads:
        Number of threads to use reading the ROOT tree
        (see uproot.TTreeMethods_pandas.df)
    wtloop_meta:
        grab and store the `WtLoop_meta` YAML entries. stored as a dictionary
        of the form ``{ str(filename) : dict(yaml) }`` in the class variable
        ``wtloop_metas``.
    TeXlabel:
        A LaTeX format label for the dataset

    Examples
    --------
    Example with a single file and two branches:

    >>> ds1 = dataset.from_root(["file.root"], name="myds",
    ...                         branches=["pT_lep1", "pT_lep2"], label=1)

    Example with multiple input_files and a selection (uses all
    branches). The selection requires the branch ``nbjets == 1``
    and ``njets >= 1``, then label it 5.

    >>> flist = ["file1.root", "file2.root", "file3.root"]
    >>> ds = dataset.from_root(flist, selection='(nbjets == 1) & (njets >= 1)')
    >>> ds.label = 5

    Example using aux weights

    >>> ds = dataset.from_root(flist, name="myds", weight_name="weight_nominal",
    ...                        auxweights=["weight_sys_radLo", " weight_sys_radHi"])

    Example where we detect aux weights automatically

    >>> ds = dataset.from_root(flist, name="myds", weight_name="weight_nominal",
    ...                        detect_weights=True)

    Example using a ThreadPoolExecutor (16 threads):

    >>> ds = dataset.from_root(flist, name="myds", nthreads=16)

    """

    if isinstance(input_files, (str, bytes)):
        input_files = [input_files]
    else:
        try:
            iter(input_files)
        except TypeError:
            input_files = [input_files]
        else:
            input_files = list(input_files)

    executor = None
    if nthreads is not None:
        executor = ThreadPoolExecutor(nthreads)

    ds = dataset()
    ds._init_skeleton(
        input_files,
        name,
        tree_name=tree_name,
        weight_name=weight_name,
        label=label,
        auxlabel=auxlabel,
        TeXlabel=TeXlabel,
    )

    if wtloop_meta:
        meta_trees = {
            file_name: uproot.open(file_name)["WtLoop_meta"] for file_name in input_files
        }
        ds.wtloop_metas = {
            fn: yaml.full_load(mt.array("meta_yaml")[0]) for fn, mt in meta_trees.items()
        }

    uproot_trees = [uproot.open(file_name)[tree_name] for file_name in input_files]

    wpat = re.compile(r"^weight_\w+")
    if auxweights is not None:
        w_branches = auxweights
    elif detect_weights:
        urtkeys = [k.decode("utf-8") for k in uproot_trees[0].keys()]
        w_branches = [k for k in urtkeys if re.match(wpat, k)]
        if weight_name in w_branches:
            w_branches.remove(weight_name)
    else:
        w_branches = None

    frame_list, weight_list, aux_frame_list = [], [], []
    for t in uproot_trees:
        raw_w = t.array(weight_name)
        raw_f = t.pandas.df(branches=branches, namedecode="utf-8", executor=executor)
        if not allow_weights_in_df:
            rmthese = [c for c in raw_f.columns if re.match(wpat, c)]
            raw_f.drop(columns=rmthese, inplace=True)

        if w_branches is not None:
            raw_aw = t.pandas.df(branches=w_branches, namedecode="utf-8")

        if selection is not None:
            iselec = raw_f.eval(selection)
            raw_w = raw_w[iselec]
            raw_f = raw_f[iselec]
            if w_branches is not None:
                raw_aw = raw_aw[iselec]

        assert len(raw_w) == len(raw_f), "frame length and weight length different"
        weight_list.append(raw_w)
        frame_list.append(raw_f)
        if w_branches is not None:
            aux_frame_list.append(raw_aw)
            assert len(raw_w) == len(raw_aw), "aux weight length and weight length different"

    weights_array = np.concatenate(weight_list)
    df = pd.concat(frame_list)
    if w_branches is not None:
        aw_df = pd.concat(aux_frame_list)
    else:
        aw_df = None

    ds._set_df_and_weights(df, weights_array, auxw=aw_df)

    if aggressively_strip:
        ds.aggressively_strip()

    return ds


def from_pytables(
    file_name: str,
    name: str = "auto",
    tree_name: str = "none",
    weight_name: str = "auto",
    label: Optional[int] = None,
    auxlabel: Optional[int] = None,
    TeXlabel: Optional[str] = None,
) -> "dataset":
    """Initialize a dataset from pytables output generated from
    :meth:`dataset.to_pytables`

    The payload is extracted from the .h5 pytables files using the
    name of the dataset and the weight name. If the name of the
    dataset doesn't exist in the file you'll crash. Aux weights
    are retrieved if available.

    Parameters
    ----------
    file_name:
        Name of h5 file containing the payload
    name:
        Name of the dataset inside the h5 file. If ``"auto"`` (default),
        we attempt to determine the name automatically from the h5 file.
    tree_name:
        Name of tree where dataset originated (only for reference)
    weight_name:
        Name of the weight array inside the h5 file. If ``"auto"`` (default),
        we attempt to determine the name automatically from the h5 file.
    label:
        Give the dataset an integer label
    auxlabel:
        Give the dataset an integer auxiliary label
    TeXlabel:
        LaTeX formatted label

    Examples
    --------

    Creating a dataset from pytables where everything is auto detected:

    >>> ds1 = dataset.from_pytables("ttbar.h5")
    >>> ds1.label = 1 ## add label dataset after the fact

    """
    with h5py.File(file_name, "r") as f:
        keys = list(f.keys())
        if name == "auto":
            for k in keys:
                if "_payload" in k:
                    name = k.split("_payload")[0]
                    break
        if weight_name == "auto":
            for k in keys:
                if "_weight" in k:
                    weight_name = k.split(f"{name}_")[-1]
                    break

    main_frame = pd.read_hdf(file_name, f"{name}_payload")
    main_weight_frame = pd.read_hdf(file_name, f"{name}_{weight_name}")
    with h5py.File(file_name, "r") as f:
        if f"{name}_auxweights" in f:
            aux_frame = pd.read_hdf(file_name, f"{name}_auxweights")
        else:
            aux_frame = None
    w_array = main_weight_frame.weights.to_numpy()
    ds = dataset()
    ds._init_skeleton(
        [file_name],
        name,
        weight_name=weight_name,
        tree_name=tree_name,
        label=label,
        auxlabel=auxlabel,
        TeXlabel=TeXlabel,
    )

    ds._set_df_and_weights(main_frame, w_array, auxw=aux_frame)
    with h5py.File(file_name, "r") as f:
        if f"{name}_wtloop_metas" in f:
            wtloop_metas = pd.read_hdf(file_name, f"{name}_wtloop_metas")
            ds.wtloop_metas = {
                fn: yaml.full_load(wtloop_metas[fn].to_numpy()[0])
                for fn in wtloop_metas.columns
            }

    return ds


def from_h5(
    file_name: str,
    name: str,
    columns: List[str],
    tree_name: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    label: Optional[int] = None,
    auxlabel: Optional[int] = None,
    TeXlabel: Optional[str] = None,
) -> "dataset":
    """Initialize a dataset from generic h5 input (loosely expected to be
    from the ATLAS Analysis Release utility ``ttree2hdf5``

    The name of the HDF5 dataset inside the file is assumed to be
    ``tree_name``. The ``name`` argument is something *you
    choose*.

    Parameters
    ----------
    file_name:
        Name of h5 file containing the payload
    name:
        Name of the dataset you would like to define
    columns:
        Names of columns (branches) to include in payload
    tree_name:
        Name of tree dataset originates from (HDF5 dataset name)
    weight_name:
        Name of the weight array inside the h5 file
    label:
        Give the dataset an integer label
    auxlabel:
        Give the dataset an integer auxiliary label
    TeXlabel:
        LaTeX form label

    Examples
    --------

    >>> ds = dataset.from_h5("file.h5", "dsname", TeXlabel=r"$tW$",
    ...                      tree_name="WtLoop_EG_RESOLUTION_ALL__1up")

    """
    ds = dataset()
    ds._init_skeleton(
        [file_name],
        name=name,
        weight_name=weight_name,
        tree_name=tree_name,
        label=label,
        auxlabel=auxlabel,
        TeXlabel=TeXlabel,
    )

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
    """Scale the weights of the `to_update` dataset such that the sum of
    weights are equal to the sum of weights of the `reference`
    dataset.

    Parameters
    ----------
    to_update:
        dataset with weights to be scaled
    reference
        dataset to scale to

    """
    assert to_update.has_payload, f"{to_update} is without payload"
    assert reference.has_payload, f"{reference} is without payload"
    sum_to_update = to_update.weights.sum()
    sum_reference = reference.weights.sum()
    to_update.weights *= sum_reference / sum_to_update
