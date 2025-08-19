import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader, random_split

# Global Variables
DataPath = 'E:/Datasets/CFPP/'
STARTTIME = datetime(year=2023, month=2, day=23, hour=10, minute=40, second=0)
ENDTIME = datetime(year=2023, month=6, day=20, hour=17, minute=30, second=0)


def getCodes(labels: list):
    codes = []
    LabelCode = []
    with open(DataPath + 'Label_Code.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            LabelCode.append(row)

    for label in labels:
        for label_code in LabelCode:
            if label_code[0] == label:
                codes.append(label_code[1])

    return codes


def getNormDataSection(labels: list, start: datetime, end: datetime):
    Data = getData(labels)
    Offset = np.zeros(shape=(len(labels),))
    Scale = np.ones(shape=(len(labels),))

    for idx in range(len(labels)):
        Data[:, idx], offset, scale = normalizer(Data[:, idx])
        Offset[idx] = offset
        Scale[idx] = scale

    index_start = int((start - STARTTIME).total_seconds() / 60)
    index_end = int((end - STARTTIME).total_seconds() / 60)
    Data = Data[index_start:index_end+1]

    timeIndex = []
    t, delta = start, timedelta(minutes=1)
    while t <= end:
        timeIndex.append(t)
        t += delta

    return timeIndex, Data, Offset, Scale


def getDataSection(labels: list, start: datetime, end: datetime):
    Data = getData(labels)
    index_start = int((start - STARTTIME).total_seconds() / 60)
    index_end = int((end - STARTTIME).total_seconds() / 60)
    Data = Data[index_start:index_end+1]

    timeIndex = []
    t, delta = start, timedelta(minutes=1)
    while t <= end:
        timeIndex.append(t)
        t += delta

    return timeIndex, Data


def getData(labels):
    codes = getCodes(labels)
    L = int((ENDTIME - STARTTIME).total_seconds() / 60) + 1
    Data = np.empty(shape=(L, len(codes)))
    for idx, code in enumerate(codes):
        series = np.load(DataPath + 'NPY/' + code + '.npy', allow_pickle=True)
        Data[:, idx] = series
    Data = Data.astype(np.float32)
    return Data



def get_Offset_Scale(label):
    code = getCodes(label)
    series = np.load(DataPath + 'NPY/' + code[0] + '.npy', allow_pickle=True)
    scale = np.max(series) - np.min(series)
    if scale == 0:
        return 0, 1
    else:
        return np.min(series), scale


def normalizer(series):
    scale = series.max() - series.min()
    if scale == 0:
        series = np.zeros(len(series))
        return series, 0, 1
    else:
        offset = series.min()
        series = (series - series.min()) / scale
        return series, offset, scale


def plot(labels,  startTime=None, endTime=None, normalize=False, splitFlag=None):
    if startTime is None:
        startTime = STARTTIME
    if endTime is None:
        endTime = ENDTIME

    if normalize:
        TimeIndex, Data, _, _ = getNormDataSection(labels, startTime, endTime)
    else:
        TimeIndex, Data = getDataSection(labels, startTime, endTime)

    plt.rcParams['font.family'] = 'Microsoft YaHei'
    if splitFlag is True:
        rows = len(labels)
        fig, axs = plt.subplots(nrows=rows, ncols=1, sharex='all')
        for t in range(rows):
            axs[t].plot_metrics(TimeIndex, Data[:, t], label=labels[t])
            axs[t].legend(loc='upper left')
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex='all')
        for t in range(len(labels)):
            axs.plot_metrics(TimeIndex, Data[:, t], label=labels[t])
            axs.legend(loc='upper left')
    plt.show()


def write_data_to_xlsx(labels, xlsxName, startTime=None, endTime=None, normalize=False):
    if startTime is None:
        startTime = STARTTIME
    if endTime is None:
        endTime = ENDTIME

    if normalize:
        TimeIndex, Data, _, _ = getNormDataSection(labels, startTime, endTime)
    else:
        TimeIndex, Data = getDataSection(labels, startTime, endTime)

    xlsxData = {'time': TimeIndex}
    for idx, label in enumerate(labels):
        xlsxData[label] = Data[:, idx]

    df = pd.DataFrame(xlsxData)
    df.to_excel(xlsxName)


class CFPP_Dataset(Dataset):
    def __init__(self, inputLabels: list, outLabels: list, start: datetime, end: datetime, step: int, winLength: int):
        _, self.u, _, _ = getNormDataSection(inputLabels, start, end)
        _, self.y, _, _ = getNormDataSection(outLabels, start, end)
        self.step = step
        self.win = winLength
        self.len = (len(self.u) - winLength) // step + 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        s = index * self.step
        e = s + self.win
        return self.u[s:e], self.y[s:e]


def get_dataloader(dataSet, ratio, trainBs, validBs):
    ratio = int(len(dataSet) * ratio)
    testRation = len(dataSet) - ratio
    train_set, test_set = random_split(dataset=dataSet, lengths=[ratio, testRation])

    train_DataLoader = DataLoader(train_set, batch_size=trainBs, shuffle=True, drop_last=True)
    valid_DataLoader = DataLoader(test_set, batch_size=validBs, shuffle=True, drop_last=True)

    return train_DataLoader, valid_DataLoader

