import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, model_name: str):
        self.model = model_name
        self.index = []
        self.rmse_list = []
        self.mae_list = []
        self.nll_list = []
        self.r2_list = []
        self.nrmse_list = []
        self.picp_list = []
        self.mpiw_list = []

    def update_metrics(self, out: np.array, target: np.array, var: Union[np.array, None] = None,
                       time: Union[datetime, None] = None,
                       selected_metrics='all', nrmse_method='range'):

        if not isinstance(selected_metrics, list):
            selected_metrics = [selected_metrics]

        if time is not None:
            self.index.append(time)

        if 'rmse' in selected_metrics or 'all' in selected_metrics:
            self.rmse_list.append(rmse(out, target))

        if 'mae' in selected_metrics or 'all' in selected_metrics:
            self.mae_list.append(mae(out, target))

        if 'nll' in selected_metrics or 'all' in selected_metrics:
            if var is not None:
                self.nll_list.append(nll(out, var, target))

        if 'r2' in selected_metrics or 'all' in selected_metrics:
            self.r2_list.append(r2(out, target))

        if 'nrmse' in selected_metrics or 'all' in selected_metrics:
            self.nrmse_list.append(nrmse(out, target, nrmse_method))

        if 'picp' in selected_metrics or 'all' in selected_metrics:
            if var is not None:
                self.picp_list.append(picp(out, var, target))

        if 'mpiw' in selected_metrics or 'all' in selected_metrics:
            if var is not None:
                self.mpiw_list.append(mpiw(var))

    def convert2df(self):
        df = pd.DataFrame({self.model + '-' + 'rmse': self.rmse_list,
                           self.model + '-' + 'mae': self.mae_list,
                           self.model + '-' + 'r2': self.r2_list,
                           self.model + '-' + 'nrmse': self.nrmse_list,
                           })

        if len(self.nll_list) == len(self.rmse_list) == len(self.picp_list) == len(self.rmse_list):
            df.insert(loc=0, column=self.model + '-' + 'nll', value=np.array(self.nll_list))
            df.insert(loc=0, column=self.model + '-' + 'picp', value=np.array(self.picp_list))
            df.insert(loc=0, column=self.model + '-' + 'mpiw', value=np.array(self.mpiw_list))

        if len(self.index) > 0 and len(self.index)==len(self.rmse_list):
            df.insert(loc=0, column='time', value=np.array(self.index))


        return df

    def print_metrics_mean(self):
        print('model:', self.model)
        print('mae  :', np.mean(self.mae_list))
        print('rmse :', np.mean(self.rmse_list))
        print('nrmse:', np.mean(self.nrmse_list))
        print('r2   :', np.mean(self.r2_list))

        if len(self.nll_list) == len(self.mpiw_list) == len(self.picp_list) == len(self.rmse_list):
            print('nll  :', np.mean(self.nll_list))
            print('picp:', np.mean(self.picp_list))
            print('mpiw:', np.mean(self.mpiw_list))


def rmse(y: np.array, target: np.array) -> float:
    res = np.sqrt(np.mean((y - target) ** 2))
    return res


def mae(y: np.array, target: np.array) -> float:
    res = np.mean(np.abs(y - target))
    return res.item()


def nll(mu: np.array, var: np.array, target: np.array) -> float:
    """
    f = (1/(sqrt(2*pi)*sigma))*exp(-(x-mu)^2/(2*sigma^2))
    -2*log(f) = log(2*pi) + log(sigma^2) + (x-mu)^2/sigma^2
    sigma^2 = var
    -2*log(f) = log(2*pi) + log(var) + (x-mu)^2/var
    log(f) = (-1/2)*[log(2*pi) + log(var) + (x-mu)^2/var]
    -log(f) = 0.5*[log(2*pi) + log(var) + (x-mu)^2/var]
    """
    res = np.log(var) + np.log(2 * np.pi) + (mu - target) ** 2 / var
    res = np.mean(res)
    res = res * 0.5
    return res


def r2(y_pred: np.array, target: np.array) -> float:
    target = np.array(target)
    y_pred = np.array(y_pred)
    y_mean = np.mean(target)
    ss_res = np.sum((target - y_pred) ** 2)
    ss_tot = np.sum((target - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def nrmse(y_pred: np.array, target: np.array, method="range") -> float:
    rmse_ = rmse(target, y_pred)
    if method == "range":
        normalization_factor = np.max(target) - np.min(target)
    elif method == "std":
        normalization_factor = np.std(target)
    elif method == "mean":
        normalization_factor = np.mean(target)
    else:
        raise ValueError("method must be 'range', 'std' or 'mean'.")

    if normalization_factor == 0:
        raise ValueError("Normalization factor is zero, unable to calculate NRMSE.")

    return rmse_ / normalization_factor


def picp(y_pred: np.array, var: np.array, target: np.array):
    std_dev = np.sqrt(var)
    lower_bound = y_pred - 3 * std_dev
    upper_bound = y_pred + 3 * std_dev

    within_interval = (target >= lower_bound) & (target <= upper_bound)
    picp_value = np.mean(within_interval)

    return picp_value


def mpiw(var: np.array):
    std_dev = np.sqrt(var)
    return np.mean(std_dev*6)

def plot_metrics(metricsDict: Metrics):

    index = metricsDict.index

    if not metricsDict.nll_list:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
        ax4.plot(index, metricsDict.nll_list, label='nll')
        ax4.legend()

    ax1.plot(index, metricsDict.rmse_list, label='rmse')
    ax1.plot(index, metricsDict.mae_list, label='mae')
    ax1.legend()

    ax2.plot(index, metricsDict.r2_list, label='r2')
    ax2.legend()

    ax3.plot(index, metricsDict.nrmse_list, label='nrmse')
    ax3.legend()

    plt.tight_layout()
    plt.legend()
    plt.show()
