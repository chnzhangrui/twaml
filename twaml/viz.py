# -*- coding: utf-8 -*-

"""twaml.viz

A module to aid visualizing our datasets

"""

from .data import dataset
import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import numpy as np
import math


def compare_distributions(
    dist1,
    dist2,
    bins: Optional["np.ndarray"] = None,
    titles: List[str] = ["dist1", "dist2"],
    colors: List[str] = ["C0", "C1"],
    ratio: bool = True,
    weight1: Optional["np.ndarray"] = None,
    weight2: Optional["np.ndarray"] = None,
    **subplots_kw,
):
    """Compare two histogrammed distributons with matplotlib

    Parameters
    ----------
    dist1
      any mpl-histogrammable object (``np.ndarray``, ``pd.Series``, etc.)
    dist2
      any mpl-histogrammable object (``np.ndarray``, ``pd.Series``, etc.)
    bins: np.ndarray
      define the bin edges
    titles: List[str]
      labels for the distributions
    ratio: bool
      add a ratio plot
    weight1: Optional[np.ndarray]
      weights associated with dist1
    weight2: Optional[np.ndarray]
      weights associated with dist2
    subplots_kw: Dict
      all additional keywords to send to ``matplotlib.pyplot.subplots``

    Returns
    -------
    fig : matpotlib.figure.Figure

    ax : matplotlib.axes.Axes or array of them
      *ax* can be either a single matplotlib.axes.Axes object or an
      array of Axes objects if more than one subplot was created.  The
      dimensions of the resulting array can be controlled with the
      squeeze keyword, see above.
    h1
      the return of ``matplotlib.axes.Axes.hist`` for dist1
    h2
      the return of ``matplotlib.axes.Axes.hist`` for dist2
    """
    if ratio:
        fig, ax = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.025),
            **subplots_kw,
        )
        h1 = ax[0].hist(
            dist1,
            bins=bins,
            histtype="step",
            label=titles[0],
            color=colors[0],
            weights=weight1,
        )
        h2 = ax[0].hist(
            dist2,
            bins=h1[1],
            histtype="step",
            label=titles[1],
            color=colors[1],
            weights=weight2,
        )
        centers = (h1[1][:-1] + h1[1][1:]) / 2.0
        ax[1].plot(centers, h1[0] / h2[0], "k-")
        ax[1].plot([centers[0] - 10e3, centers[1] + 10e3], np.ones(2), "k--")
        ax[1].set_ylim([0, 2])
        ax[1].set_xlim([h1[1][0], h1[1][-1]])
    else:
        fig, ax = plt.subplots(**subplots_kw)
        h1 = ax.hist(dist1, bins=bins, histtype="step", label=titles[0], color=colors[0])
        h2 = ax.hist(dist2, bins=h1[1], histtype="step", label=titles[1], color=colors[1])
        ax.set_xlim([h1[1][0], h1[1][-1]])

    return fig, ax, h1, h2


def compare_columns(
    ds1: dataset,
    ds2: dataset,
    columns: Optional[List[str]] = None,
    names: Optional[Tuple[str, str]] = None,
    colors: Optional[Tuple[str, str]] = None,
    density=True,
    **subplots_kw,
):
    """generate a set of histograms comparing the distributions of a set
    of columns in two different datasets.

    Parameters
    ----------
    ds1: twaml.data.dataset
        The first dataset
    ds2: twaml.data.dataset
        The second dataset
    columns: Optional[List[str]]
        Columns to plot; if None, plot all
    names: Optional[Tuple[str,str]]
        Names for the legend, if None use the dataset ``name`` attributes
    colors: Optional[Tuple[str,str]]
        Colors for the histograms
    density: bool
        Feed to ``density`` parameter in ``matplotlib.pyplot.hist``
    subplots_kw: Dict
      all additional keywords to send to ``matplotlib.pyplot.subplots``
    """

    col1 = list(ds1.df.columns)
    col2 = list(ds2.df.columns)
    if columns is None:
        assert col1 == col2, "different columns"
        cols = col1
    else:
        cols = columns
        for c in cols:
            assert c in col1, f"{c} column not in ds1"
            assert c in col2, f"{c} column not in ds2"
    if names is None:
        names = (ds1.name, ds2.name)
    if colors is None:
        colors = ("C0", "C1")

    dim = math.sqrt(len(cols))
    dim1 = int(dim)
    if dim1 > dim:
        dim2 = dim1 + 1
    else:
        dim2 = dim1
    fig, axs = plt.subplots(dim1, dim2)

    w1, w2 = ds1.weights, ds2.weights
    for col, a in zip(cols, axs.flatten()):
        dist1 = ds1.df[col].to_numpy()
        dist2 = ds2.df[col].to_numpy()
        print(dist1.dtype, dist2.dtype)
        xmin = min(np.min(dist1), np.min(dist2))
        xmax = max(np.max(dist1), np.max(dist1))
        if (
            dist1.dtype == np.dtype("i4")
            or dist1.dtype == np.dtype("i8")
            or dist1.dtype == np.dtype("u4")
            or dist1.dtype == np.dtype("u8")
        ):
            nbins = xmax - xmin
            xmax = xmax + 0.5
            xmin = xmin - 0.5
        else:
            nbins = 50
        a.hist(
            [dist1, dist2],
            bins=np.linspace(xmin, xmax, nbins + 1),
            weights=[w1, w2],
            density=density,
            histtype="step",
        )
    fig.savefig(f"{ds1.name}_{ds2.name}.pdf")
