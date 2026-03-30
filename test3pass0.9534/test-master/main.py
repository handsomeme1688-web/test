import random
import numpy as np
import torch
from datapro import Simdata_pro, load_data

from train import train_test


class Config:
    def __init__(self):
        self.datapath = './dataset/'
        self.seed = 42
        self.kfold = 5
        self.batchSize = 64
        self.ratio = 0.2
        self.epoch = 12
        self.patience = 5
        self.calibration_ratio = 0.1
        self.view = 3
        self.sim_topk = 96
        self.nei_size = [64, 16]
        self.hop = 2
        self.feture_size = 256
        self.edge_feature = 9
        self.atthidden_fea = 128
        self.sim_class = 3
        self.md_class = 3
        self.m_num = 853
        self.d_num = 591
        self.Dropout = 0.2
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.train_unlabeled_ratio = 1
        self.eval_unlabeled_ratio = 1
        self.threshold = 0.5
        self.rn_quantile = 0.2
        self.loss_pu_weight = 1.0
        self.loss_sup_weight = 1.0
        self.loss_rank_weight = 0.5
        self.valid_use_full_data = True
        self.aux_knn_k = 32
        self.enable_fusion = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    param = Config()
    set_seed(param.seed)
    simData = Simdata_pro(param)
    train_data = load_data(param)
    train_test(simData, train_data, param, state='valid')


if __name__ == "__main__":
    main()
