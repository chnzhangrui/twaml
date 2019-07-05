import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from twaml.data import from_pytables
from twaml.data import scale_weight_sum
from twaml.viz import compare_columns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("pdf")

BRANCHES = [
    "nloosejets",
    "pT_jetL1",
    "mass_lep1jet2",
    "mass_lep1jet1",
    "deltaR_lep1_jet1",
    "mass_lep2jet1",
    "sigpTsys_lep1lep2jet1",
    "deltaR_lep1_jet2",
    "deltaR_lep1lep2_jet1jet2",
    "deltaR_lep2_jet1",
    "deltaR_lep2_jet2",
    "deltapT_lep2_jet2",
    "pT_jetF",
    "deltaR_lep1lep2jetC_jetF",
]


def get_combined():
    ttbar = from_pytables("ttbar.h5", "ttbar", label=0)
    tW_DR = from_pytables("tW_DR.h5", "tW_DR", label=0)
    tW_DS = from_pytables("tW_DS.h5", "tW_DS", label=1)
    scale_weight_sum(tW_DR, ttbar)
    scale_weight_sum(tW_DS, ttbar)
    tW_DR.weights *= 0.5
    tW_DS.weights *= 0.5
    print(ttbar.weights.sum(), tW_DR.weights.sum(), tW_DS.weights.sum())
    return ttbar, tW_DR, tW_DS


def train_2j2b():
    ttbar, tW_DR, tW_DS = get_combined()
    sow = ttbar.weights.sum() + tW_DR.weights.sum() + tW_DS.weights.sum()
    mwfl = sow * 0.01

    y = np.concatenate(
        [
            np.ones_like(tW_DR.weights),
            np.ones_like(tW_DS.weights),
            np.zeros_like(ttbar.weights),
        ]
    )
    X = np.concatenate([tW_DR.df.to_numpy(), tW_DS.df.to_numpy(), ttbar.df.to_numpy()])
    w = np.concatenate([tW_DR.weights, tW_DS.weights, ttbar.weights])

    folder = KFold(n_splits=3, shuffle=True, random_state=414)

    ttbar_dist = []
    ttbar_w_dist = []
    tW_dist = []
    tW_w_dist = []
    roc_aucs = []
    print("preproc done")
    print("starting folded training")
    for train_idx, test_idx in folder.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = w[train_idx], w[test_idx]

        param = {"max_depth": 3, "n_estimators": 150, "min_child_weight": mwfl}
        param["nthread"] = 4
        param["eval_metric"] = "auc"
        model = xgb.XGBClassifier(**param)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric="logloss",
            verbose=False,
        )
        # print(model.evals_result())
        predictions = model.predict(X_test)
        predict_pro = model.predict_proba(X_test)
        roc_aucs.append(roc_auc_score(y_test, predictions))
        tW_w_dist.append(w_test[y_test == 1])
        ttbar_w_dist.append(w_test[y_test == 0])
        ttbar_dist.append(predict_pro[y_test == 0])
        tW_dist.append(predict_pro[y_test == 1])
        print("just finished a fold")
    ttbar_dist_all = np.concatenate(ttbar_dist)
    ttbar_w_dist_all = np.concatenate(ttbar_w_dist)
    tW_dist_all = np.concatenate(tW_dist)
    tW_w_dist_all = np.concatenate(tW_w_dist)
    _ = plt.hist(
        tW_dist_all.T[1],
        bins=100,
        weights=tW_w_dist_all,
        density=True,
        histtype="step",
        label="tW",
    )
    _ = plt.hist(
        ttbar_dist_all.T[1],
        bins=_[1],
        density=True,
        weights=ttbar_w_dist_all,
        histtype="step",
        label="ttbar",
    )
    _ = plt.legend()
    plt.savefig("out2.pdf")
    del _


def play_with_plots():
    ttbar, tW_DR, tW_DS = get_combined()
    tW_BO = tW_DR + tW_DS
    compare_columns(ttbar, tW_BO)


if __name__ == "__main__":
    play_with_plots()
