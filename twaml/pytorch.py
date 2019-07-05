import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

from typing import Optional


def get_device(option: str = "auto"):
    """helper function for getting pytorch device

    Paramaters
    ----------
    option:
      - if "auto" it tries to get the GPU and returns CPU if unavailable
      - if "cpu" just use CPU
    """
    if option == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif option == "cpu":
        return torch.device("cpu")


device = get_device("auto")


class TworchDataset(Dataset):
    """Training dataset container"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        z: Optional[np.ndarray] = None,
    ):
        """initialize a TwamlDataset

        Parameters
        ----------
        X:
          sample feature matrix
        y:
          sample target label
        w:
          sample weights
        z:
          sample auxiliary label (for Adversarial)
        """
        y = np.asarray(y, dtype=np.float32)
        if z is not None:
            z = np.asarray(z, dtype=np.float32)

        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.w = torch.from_numpy(weights).to(device)
        if z is not None:
            self.z = torch.from_numpy(z).to(device)
        else:
            self.z = None

        self.n_features = self.X.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.z is not None:
            return (self.X[idx], self.y[idx], self.w[idx], self.z[idx])
        else:
            return (self.X[idx], self.y[idx], self.w[idx])


class SimpleNetwork(torch.nn.Module):
    """simple deep linear layers"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

    """
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        return F.sigmoid(self.linear6(x))
    """
