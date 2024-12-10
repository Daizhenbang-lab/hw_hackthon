import numpy as np
import os
import torch
from torch.utils.data import Dataset


class ChannelTrainSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ChannelTestSet(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


def load_labelled_data(H_data_path, input_pos_path):
    # load input pos data
    pos_data = np.loadtxt(input_pos_path)
    labelled_pos = pos_data[:, 0].astype(int) - 1 # 1-indexed to 0-indexed
    pos_data = pos_data[:, 1:]

    # read all_input data
    H = np.load(H_data_path)

    # select the labelled data
    H = H[labelled_pos, :, :, :]

    return ChannelTrainSet(H, pos_data)


def load_all_data(H_data_path):
    H = np.load(H_data_path)
    return ChannelTestSet(H)


def load_eval_data(H_data_path, ground_truth_path):
    # load input pos data
    pos_data = np.loadtxt(ground_truth_path)

    # read all_input data
    H = np.load(H_data_path)

    return ChannelTrainSet(H, pos_data)


if __name__ == '__main__':
    # # Test the dataset
    # eval_set = load_eval_data(
    #     'data/Dataset1CfgData1.txt',
    #     'data/Round1InputData1.txt',
    #     'data/Dataset1GroundTruth1.txt',
    #     'data/evaltemp1_1.npy')
    # print(len(eval_set))
    # print(eval_set[0])

    dataset_idx = 0
    data_idx = 1
