import csv
import os

import numpy as np
import torch
import torch.utils.data.dataset as Dataset


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = [[float(i) for i in row] for row in reader]
    return torch.tensor(md_data, dtype=torch.float32)


def Simdata_processing(param):
    dataset = dict()
    dataset['mm_f'] = read_csv(os.path.join(param.datapath, 'm_fs.csv'))
    dataset['mm_s'] = read_csv(os.path.join(param.datapath, 'm_ss.csv'))
    dataset['mm_g'] = read_csv(os.path.join(param.datapath, 'm_gs.csv'))
    dataset['dd_t'] = read_csv(os.path.join(param.datapath, 'd_ts.csv'))
    dataset['dd_s'] = read_csv(os.path.join(param.datapath, 'd_ss.csv'))
    dataset['dd_g'] = read_csv(os.path.join(param.datapath, 'd_gs.csv'))
    return dataset


def Simdata_pro(param):
    dataset = dict()
    dataset['mm_f'] = torch.from_numpy(
        np.loadtxt(os.path.join(param.datapath, 'm_fs.csv'), dtype=np.float32, delimiter=',')
    )
    dataset['mm_s'] = torch.from_numpy(
        np.loadtxt(os.path.join(param.datapath, 'm_ss.csv'), dtype=np.float32, delimiter=',')
    )
    dataset['mm_g'] = torch.from_numpy(
        np.loadtxt(os.path.join(param.datapath, 'm_gs.csv'), dtype=np.float32, delimiter=',')
    )
    dataset['dd_t'] = torch.from_numpy(
        np.loadtxt(os.path.join(param.datapath, 'd_ts.csv'), dtype=np.float32, delimiter=',')
    )
    dataset['dd_s'] = torch.from_numpy(
        np.loadtxt(os.path.join(param.datapath, 'd_ss.csv'), dtype=np.float32, delimiter=',')
    )
    dataset['dd_g'] = torch.from_numpy(
        np.loadtxt(os.path.join(param.datapath, 'd_gs.csv'), dtype=np.float32, delimiter=',')
    )
    return dataset


def _shuffle_rows(rng, array):
    indices = rng.permutation(array.shape[0])
    return array[indices]


def load_data(param):
    md_matrix = np.loadtxt(os.path.join(param.datapath, 'm_d.csv'), dtype=np.float32, delimiter=',')
    md_class = np.loadtxt(os.path.join(param.datapath, 'm_d_edge.csv'), dtype=np.float32, delimiter=',')

    rng = np.random.default_rng(seed=param.seed)
    pos_edges = np.column_stack(np.where(md_matrix == 1)).astype(np.int64)
    unlabeled_edges = np.column_stack(np.where(md_matrix == 0)).astype(np.int64)

    pos_edges = _shuffle_rows(rng, pos_edges)
    unlabeled_edges = _shuffle_rows(rng, unlabeled_edges)

    test_pos_count = max(1, int(pos_edges.shape[0] * param.ratio))
    train_pos_edges = pos_edges[test_pos_count:]
    test_pos_edges = pos_edges[:test_pos_count]

    eval_ratio = max(1, int(param.eval_unlabeled_ratio))
    train_ratio = max(1, int(param.train_unlabeled_ratio))

    test_unlabeled_count = min(unlabeled_edges.shape[0], test_pos_count * eval_ratio)
    test_unlabeled_edges = unlabeled_edges[:test_unlabeled_count]

    remaining_unlabeled = unlabeled_edges[test_unlabeled_count:]
    train_unlabeled_count = min(remaining_unlabeled.shape[0], train_pos_edges.shape[0] * train_ratio)
    train_unlabeled_edges = remaining_unlabeled[:train_unlabeled_count]

    full_unlabeled_count = min(unlabeled_edges.shape[0], pos_edges.shape[0] * eval_ratio)
    full_unlabeled_edges = unlabeled_edges[:full_unlabeled_count]
    full_edges = np.vstack((pos_edges, full_unlabeled_edges))
    full_labels = np.hstack((
        np.ones(pos_edges.shape[0], dtype=np.float32),
        np.zeros(full_unlabeled_edges.shape[0], dtype=np.float32),
    ))

    train_edges = np.vstack((train_pos_edges, train_unlabeled_edges))
    train_labels = np.hstack((
        np.ones(train_pos_edges.shape[0], dtype=np.float32),
        np.zeros(train_unlabeled_edges.shape[0], dtype=np.float32),
    ))

    test_edges = np.vstack((test_pos_edges, test_unlabeled_edges))
    test_labels = np.hstack((
        np.ones(test_pos_edges.shape[0], dtype=np.float32),
        np.zeros(test_unlabeled_edges.shape[0], dtype=np.float32),
    ))

    edge_idx_dict = dict()
    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_labels
    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_labels
    edge_idx_dict['train_Pos_Edges'] = train_pos_edges
    edge_idx_dict['train_Unlabeled_Edges'] = train_unlabeled_edges
    edge_idx_dict['test_Pos_Edges'] = test_pos_edges
    edge_idx_dict['test_Unlabeled_Edges'] = test_unlabeled_edges
    edge_idx_dict['full_Edges'] = full_edges
    edge_idx_dict['full_Labels'] = full_labels
    edge_idx_dict['md_class'] = md_class
    edge_idx_dict['true_md'] = md_matrix
    return edge_idx_dict


class EdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels, conf_matrix=None):
        self.Data = edges
        self.Label = labels
        self.conf_matrix = conf_matrix

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index].astype(np.int64)
        label = np.float32(self.Label[index])

        if label > 0.5:
            confidence = np.float32(1.0)
        elif self.conf_matrix is not None:
            confidence = np.float32(self.conf_matrix[int(data[0]), int(data[1])])
        else:
            confidence = np.float32(0.0)

        return data, label, confidence
