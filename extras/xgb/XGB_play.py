import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from twaml.data import from_pytables
from twaml.data import scale_weight_sum
import matplotlib.pyplot as plt

ttbar = from_pytables("ttbar_1j1b.h5", "ttbar", label=0)
tW_DR = from_pytables("tW_DR_1j1b.h5", "tW_DR", label=1)
sow = ttbar.weights.sum() + tW_DR.weights.sum()
mwfl = sow * 0.01
scale_weight_sum(tW_DR, ttbar)

y = np.concatenate([tW_DR.label_asarray, ttbar.label_asarray])
X = np.concatenate([tW_DR.df.to_numpy(), ttbar.df.to_numpy()])
w = np.concatenate([tW_DR.weights, ttbar.weights])

folder = KFold(n_splits=3, shuffle=True, random_state=414)

ttbar_dist = []
tW_dist = []
tW_w_dist = []
ttbar_w_dist = []
roc_aucs = []
for train_idx, test_idx in folder.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train, w_test = w[train_idx], w[test_idx]

    param = {"max_depth": 4, "n_estimators": 150, "min_child_weight": mwfl}
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
plt.savefig("out.pdf")
del _
