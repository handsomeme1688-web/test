# import numpy as np
# import torch
# import csv
# import torch.utils.data.dataset as Dataset
#
#
# def read_csv(path):
#     with open(path, 'r', newline='') as csv_file:
#         reader = csv.reader(csv_file)
#         md_data = []
#         md_data += [[float(i) for i in row] for row in reader]
#         return torch.Tensor(md_data)
#
#
# def Simdata_processing(param):
#     dataset = dict()
#
#     mm_funsim = read_csv(param.datapath + '/m_fs.csv')
#     dataset['mm_f'] = mm_funsim
#
#     mm_seqsim = read_csv(param.datapath + '/m_ss.csv')
#     dataset['mm_s'] = mm_seqsim
#
#     mm_gausim = read_csv(param.datapath + '/m_gs.csv')
#     dataset['mm_g'] = mm_gausim
#
#     dd_funsim = read_csv(param.datapath + '/d_ts.csv')
#     dataset['dd_t'] = dd_funsim
#
#     dd_semsim = read_csv(param.datapath + '/d_ss.csv')
#     dataset['dd_s'] = dd_semsim
#
#     dd_gausim = read_csv(param.datapath + '/d_gs.csv')
#     dataset['dd_g'] = dd_gausim
#
#     return dataset
#
#
# def Simdata_pro(param):
#     dataset = dict()
#     mm_funsim = np.loadtxt(param.datapath + 'm_fs.csv', dtype=np.float, delimiter=',')
#     mm_seqsim = np.loadtxt(param.datapath + 'm_ss.csv', dtype=np.float, delimiter=',')
#     mm_gausim = np.loadtxt(param.datapath + 'm_gs.csv', dtype=np.float, delimiter=',')
#     dd_funsim = np.loadtxt(param.datapath + 'd_ts.csv', dtype=np.float, delimiter=',')
#     dd_semsim = np.loadtxt(param.datapath + 'd_ss.csv', dtype=np.float, delimiter=',')
#     dd_gausim = np.loadtxt(param.datapath + 'd_gs.csv', dtype=np.float, delimiter=',')
#
#     dataset['mm_f'] = torch.FloatTensor(mm_funsim)
#     dataset['mm_s'] = torch.FloatTensor(mm_seqsim)
#     dataset['dd_t'] = torch.FloatTensor(dd_funsim)
#     dataset['dd_s'] = torch.FloatTensor(dd_semsim)
#
#     dataset['mm_g'] = torch.FloatTensor(mm_gausim)
#     dataset['dd_g'] = torch.FloatTensor(dd_gausim)
#
#     return dataset
#
#
# def load_data(param):
#     # Load the original miRNA-disease associations matrix.
#     md_matrix = np.loadtxt(param.datapath + '/m_d.csv', dtype=np.float32, delimiter=',')
#
#     rng = np.random.default_rng(seed=42)
#     pos_samples = np.where(md_matrix == 1)
#
#     pos_samples_shuffled = rng.permutation(pos_samples, axis=1)
#
#     rng = np.random.default_rng(seed=42)
#     neg_samples = np.where(md_matrix == 0)
#     neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
#
#     edge_idx_dict = dict()
#     n_pos_samples = pos_samples_shuffled.shape[1]
#     idx_split = int(n_pos_samples * param.ratio)
#
#     test_pos_edges = pos_samples_shuffled[:, :idx_split]
#     test_neg_edges = neg_samples_shuffled[:, :idx_split]
#     test_pos_edges = test_pos_edges.T
#     test_neg_edges = test_neg_edges.T
#     test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
#     test_true_label = np.array(test_true_label, dtype='float32')
#     test_edges = np.vstack((test_pos_edges, test_neg_edges))
#     # np.savetxt('./train_test/test_pos.csv', test_pos_edges, delimiter=',')
#     # np.savetxt('./train_test/test_neg.csv', test_neg_edges, delimiter=',')
#
#     train_pos_edges = pos_samples_shuffled[:, idx_split:]
#     train_neg_edges = neg_samples_shuffled[:, idx_split:]
#     train_pos_edges = train_pos_edges.T
#     train_neg_edges = train_neg_edges.T
#     train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
#     train_true_label = np.array(train_true_label, dtype='float32')
#     train_edges = np.vstack((train_pos_edges, train_neg_edges))
#     # np.savetxt('./train_test/train_pos.csv', train_pos_edges, delimiter=',')
#     # np.savetxt('./train_test/train_neg.csv', train_neg_edges, delimiter=',')
#
#     edge_idx_dict['train_Edges'] = train_edges
#     edge_idx_dict['train_Labels'] = train_true_label
#
#     edge_idx_dict['test_Edges'] = test_edges
#     edge_idx_dict['test_Labels'] = test_true_label
#
#     # Load the collected miRNA-disease associations matrix with edge attributes.
#     md_class = np.loadtxt(param.datapath + '/m_d_edge.csv', dtype=np.float32, delimiter=',')
#     edge_idx_dict['md_class'] = md_class
#     edge_idx_dict['true_md'] = md_matrix
#
#     return edge_idx_dict
#
#
# class EdgeDataset(Dataset.Dataset):
#     def __init__(self, edges, labels):
#         self.Data = edges
#         self.Label = labels
#
#     def __len__(self):
#         return len(self.Label)
#
#     def __getitem__(self, index):
#         data = self.Data[index]
#         label = self.Label[index]
#         return data, label
import numpy as np
import torch
import csv
import torch.utils.data.dataset as Dataset


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def Simdata_processing(param):
    dataset = dict()
    dataset['mm_f'] = read_csv(param.datapath + '/m_fs.csv')
    dataset['mm_s'] = read_csv(param.datapath + '/m_ss.csv')
    dataset['mm_g'] = read_csv(param.datapath + '/m_gs.csv')
    dataset['dd_t'] = read_csv(param.datapath + '/d_ts.csv')
    dataset['dd_s'] = read_csv(param.datapath + '/d_ss.csv')
    dataset['dd_g'] = read_csv(param.datapath + '/d_gs.csv')
    return dataset


def Simdata_pro(param):
    dataset = dict()
    dataset['mm_f'] = torch.FloatTensor(np.loadtxt(param.datapath + 'm_fs.csv', dtype=np.float32, delimiter=','))
    dataset['mm_s'] = torch.FloatTensor(np.loadtxt(param.datapath + 'm_ss.csv', dtype=np.float32, delimiter=','))
    dataset['dd_t'] = torch.FloatTensor(np.loadtxt(param.datapath + 'd_ts.csv', dtype=np.float32, delimiter=','))
    dataset['dd_s'] = torch.FloatTensor(np.loadtxt(param.datapath + 'd_ss.csv', dtype=np.float32, delimiter=','))
    dataset['mm_g'] = torch.FloatTensor(np.loadtxt(param.datapath + 'm_gs.csv', dtype=np.float32, delimiter=','))
    dataset['dd_g'] = torch.FloatTensor(np.loadtxt(param.datapath + 'd_gs.csv', dtype=np.float32, delimiter=','))
    return dataset


def load_data(param):
    """
    【论文级数据划分】：
    1. 训练集保留 PU 属性（正样本 vs 无标签样本）
    2. 测试集还原真实排序场景，避免 Easy Negative 污染
    """
    md_matrix = np.loadtxt(param.datapath + '/m_d.csv', dtype=np.float32, delimiter=',')
    rng = np.random.default_rng(seed=42)

    pos_samples = np.array(np.where(md_matrix == 1))
    unlabeled_samples = np.array(np.where(md_matrix == 0))

    pos_shuffled = rng.permutation(pos_samples, axis=1)
    unlabeled_shuffled = rng.permutation(unlabeled_samples, axis=1)

    n_pos = pos_shuffled.shape[1]
    idx_split = int(n_pos * param.ratio)

    # --- 测试集构造 ---
    test_pos_edges = pos_shuffled[:, :idx_split].T
    # 为了评估 AUC/AUPR，测试集的“负样本”实际上是未验证的候选对 (Candidates)
    # 我们抽取与正样本比例为 1:10 的候选对，模拟真实的极度不平衡检索场景
    test_unlabeled_edges = unlabeled_shuffled[:, :idx_split * 10].T
    test_labels = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_unlabeled_edges.shape[0])))
    test_edges = np.vstack((test_pos_edges, test_unlabeled_edges))

    # --- 训练集构造 ---
    train_pos_edges = pos_shuffled[:, idx_split:].T
    num_train_pos = train_pos_edges.shape[0]
    # 训练集也采用 1:10 的比例引入大量的无标签样本，让 PU 学习发挥作用
    train_unlabeled_edges = unlabeled_shuffled[:, idx_split * 10: idx_split * 10 + 10 * num_train_pos].T

    train_labels = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_unlabeled_edges.shape[0])))
    train_edges = np.vstack((train_pos_edges, train_unlabeled_edges))

    edge_idx_dict = dict()
    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = np.array(train_labels, dtype='float32')
    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = np.array(test_labels, dtype='float32')
    edge_idx_dict['md_class'] = np.loadtxt(param.datapath + '/m_d_edge.csv', dtype=np.float32, delimiter=',')
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
        data = self.Data[index]
        label = self.Label[index]

        conf = 0.0
        if self.conf_matrix is not None:
            m_idx, d_idx = int(data[0]), int(data[1])
            conf = self.conf_matrix[m_idx, d_idx]

        return data, label, np.float32(conf)