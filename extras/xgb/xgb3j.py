from twaml.data import from_pytables
from twaml.data import scale_weight_sum
import yaml

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from datetime import datetime

import xgboost as xgb

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import time
import os
import sys

rseed = 414


def prepare_data(region="3jHb", delete_datasets=True):

    ttbar = from_pytables(
        f"/Users/ddavis/Desktop/newfullkincomb/ttbar_r{region}.h5", label=0, auxlabel=1
    )
    tW_DR = from_pytables(
        f"/Users/ddavis/Desktop/newfullkincomb/tW_DR_r{region}.h5", label=1, auxlabel=1
    )
    # tW_DS = from_pytables(f"/home/ddavis/ATLAS/data/h5s/tW_DS_r{region}.h5", label=1, auxlabel=0)

    # tW_DR.keep_columns(branches)
    # ttbar.keep_columns(branches)
    # tW_DS.keep_columns(branches)
    # scale_weight_sum(tW_DS, ttbar)
    scale_weight_sum(tW_DR, ttbar)
    tW_DR.weights *= 100
    # tW_DS.weights *= 50
    ttbar.weights *= 100

    X = pd.concat([ttbar.df, tW_DR.df]).to_numpy()  # , tW_DS.df]).to_numpy()
    w = np.concatenate([ttbar.weights, tW_DR.weights])  # , tW_DS.weights])
    y = np.concatenate(
        [ttbar.label_asarray(), tW_DR.label_asarray()]
    )  # , tW_DS.label_asarray()])
    z = np.concatenate(
        [ttbar.auxlabel_asarray(), tW_DR.auxlabel_asarray()]  # , tW_DS.auxlabel_asarray()]
    )

    if delete_datasets:
        del ttbar
        del tW_DR

    print("returning data")
    return (X, y, w, z)


def simple_xgb(sow, params=None):
    if params is None:
        params = {
            "max_depth": 3,
            "n_estimators": 150,
            "min_child_weight": sow * 0.005,
            "verbosity": 2,
        }
        params["n_job"] = 8
    model = xgb.XGBClassifier(**params)
    return model


def fit_model(model, X_train, y_train, w_train, X_test, y_test, w_test, output_dir=None):
    print("starting fit")
    if output_dir is None:
        timestamp = time.time()
        output_dir = datetime.fromtimestamp(timestamp).strftime("%Y%lr%d_%H%M%S")

    origdir = os.getcwd()
    os.mkdir(output_dir)
    os.chdir(output_dir)

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[w_train, w_test],
        eval_metric="logloss",
        verbose=True,
    )


def main():
    X, y, z, w = prepare_data()
    folder = KFold(n_splits=2, shuffle=True, random_state=rseed)
    top_dir = os.getcwd()

    for i, (test_idx, train_idx) in enumerate(folder.split(X)):
        X_train, y_train, w_train = X[train_idx], y[train_idx], w[train_idx]
        X_test, y_test, w_test = X[test_idx], y[test_idx], w[test_idx]

        sow = w_train.sum() + w_test.sum()

        simpmodel = simple_xgb(sow)

        fit_model(simpmodel, X_train, y_train, w_train, X_test, y_test, w_test)

        y_pred = simpmodel.predict(X_test)
        y_pred_train = simpmodel.predict(X_train)
        sig = y_pred[y_test == 1]
        sig_train = y_pred_train[y_train == 1]
        bkg = y_pred[y_test == 0]
        bkg_train = y_pred_train[y_train == 0]

        ax.hist(
            [sig, bkg],
            bins=np.linspace(bkg.min(), sig.max(), 41),
            density=True,
            label=["sig", "bkg"],
            histtype="step",
            weights=[w_test[y_test == 1], w_test[y_test == 0]],
        )
        ax.hist(
            [sig_train, bkg_train],
            bins=np.linspace(bkg.min(), sig.max(), 41),
            density=True,
            label=["sig train", "bkg train"],
            histtype="step",
            weights=[w_train[y_train == 1], w_train[y_train == 0]],
            linestyle="--",
        )
        ax.legend()
        fig.savefig(f"{outputdir}/hists.pdf")
        break

    os.chdir(top_dir)
    plt.cla()

    return 0


if __name__ == "__main__":
    main()
