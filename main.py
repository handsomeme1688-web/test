import torch
import random
import numpy as np
from datapro import Simdata_pro, load_data

from train import train_test


class Config:
    def __init__(self):
        self.datapath = './dataset/'
        self.kfold = 5
        self.batchSize = 64   # 64
        self.ratio = 0.2
        self.epoch = 6  # 12
        self.view = 3
        self.nei_size = [512, 32]  # Select the appropriate the sampling size of multi-hop neighbor.
        self.hop = 2
        self.feture_size = 256
        self.edge_feature = 9
        self.atthidden_fea = 128
        self.sim_class = 3
        self.md_class = 3
        self.m_num = 853
        self.d_num = 591
        self.Dropout = 0.7
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.device = torch.device('cuda')


def main():
    param = Config()
    simData = Simdata_pro(param)
    train_data = load_data(param)
    result = train_test(simData, train_data, param, state='valid')


if __name__ == "__main__":
    main()
