from twaml.data import from_pytables
from twaml.data import scale_weight_sum
from twaml.pytorch import TworchDataset
from twaml.pytorch import SimpleNetwork
from twaml.pytorch import device

from torch.utils.data import DataLoader
import torch.optim
import torch.nn
import torch.nn.functional
import torch

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import time
import os
import sys


rseed = 414


def prepare_raw_data(branchinfo_file="../vars.yaml", region="2j2b"):
    with open(branchinfo_file, "r") as f:
        branches = yaml.load(f, Loader=yaml.FullLoader)
    branches = branches[region]

    ttbar = from_pytables(
        f"/home/ddavis/ATLAS/data/h5s/ttbar_r{region}.h5", label=0, auxlabel=1
    )
    tW_DR = from_pytables(
        f"/home/ddavis/ATLAS/data/h5s/tW_DR_r{region}.h5", label=1, auxlabel=1
    )
    tW_DS = from_pytables(
        f"/home/ddavis/ATLAS/data/h5s/tW_DS_r{region}.h5", label=1, auxlabel=0
    )

    tW_DR.keep_columns(branches)
    ttbar.keep_columns(branches)
    tW_DS.keep_columns(branches)
    scale_weight_sum(tW_DS, ttbar)
    scale_weight_sum(tW_DR, ttbar)
    tW_DR.weights *= 50
    tW_DS.weights *= 50
    ttbar.weights *= 100

    X = pd.concat([ttbar.df, tW_DR.df, tW_DS.df]).to_numpy()
    w = np.concatenate([ttbar.weights, tW_DR.weights, tW_DS.weights])
    y = np.concatenate([ttbar.label_asarray, tW_DR.label_asarray, tW_DS.label_asarray])
    z = np.concatenate(
        [ttbar.auxlabel_asarray, tW_DR.auxlabel_asarray, tW_DS.auxlabel_asarray]
    )

    return (X, y, w, z)



def train(model, train_loader, opt, epoch, log_interval=200):
    model.train()
    for batch_idx, (X, y, w) in enumerate(train_loader):
        opt.zero_grad()
        output = model(X)
        loss = torch.nn.functional.binary_cross_entropy(output, y, weight=w, reduction='none')
        loss.backward()
        opt.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y, w in test_loader:
            output = model(X)
            test_loss += torch.nn.functional.binary_cross_entropy(output, y, weight=w, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.long().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def light_it_up():
    X_raw, y_raw, w_raw, z_raw = prepare_raw_data()
    folder = KFold(n_splits=2, shuffle=True, random_state=rseed)

    for ifold, (test_idx, train_idx) in enumerate(folder.split(X_raw)):
        X_train, y_train, w_train = X_raw[train_idx], y_raw[train_idx], w_raw[train_idx]
        X_test, y_test, w_test = X_raw[test_idx], y_raw[test_idx], w_raw[test_idx]

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_train)

        trainDataset = TworchDataset(X_train, y_train, w_train)
        testDataset = TworchDataset(X_test, y_test, w_test)

        trainLoader = DataLoader(trainDataset, batch_size=512, shuffle=True)
        testLoader = DataLoader(testDataset, batch_size=512, shuffle=True)

        net = SimpleNetwork(trainDataset.n_features).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.8)

        for i in range(1, 51):
            train(net, trainLoader, optimizer, i)
            test(net, testLoader)



def main():
    light_it_up()
    return 0


if __name__ == "__main__":
    main()
