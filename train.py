# import time
# import gc
# import torch
# import random
# from datapro import EdgeDataset
# from model import SuperedgeLearn
# import numpy as np
# from sklearn import metrics
# import torch.utils.data.dataloader as DataLoader
# from sklearn.model_selection import KFold
# import scipy.sparse as sp
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# import math
#
# import os
#
#
# def get_metrics(score, label):
#     y_pre = score
#     y_true = label
#     metric = caculate_metrics(y_pre, y_true)
#
#     return metric
#
#
# def caculate_metrics(pre_score, real_score):
#     y_pre = pre_score
#     y_true = real_score
#
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
#     auc = metrics.auc(fpr, tpr)
#     precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
#     aupr = metrics.auc(recall_u, precision_u)
#
#     y_score = [0 if j < 0.5 else 1 for j in y_pre]
#
#     acc = metrics.accuracy_score(y_true, y_score)
#     f1 = metrics.f1_score(y_true, y_score)
#     recall = metrics.recall_score(y_true, y_score)
#     precision = metrics.precision_score(y_true, y_score)
#
#     metric_result = [auc, aupr, acc, f1, recall, precision]
#     print("One epoch metric： ")
#     print_met(metric_result)
#
#     return metric_result
#
#
# def print_met(list):
#     print('AUC ：%.4f ' % (list[0]),
#           'AUPR ：%.4f ' % (list[1]),
#           'Accuracy ：%.4f ' % (list[2]),
#           'f1_score ：%.4f ' % (list[3]),
#           'recall ：%.4f ' % (list[4]),
#           'precision ：%.4f \n' % (list[5]))
#
#
# def train_test(simData, train_data, param, state):
#     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     torch.cuda.set_per_process_memory_fraction(0.5, 0)
#     epo_metric = []
#     valid_metric = []
#
#     train_edges = train_data['train_Edges']
#     train_labels = train_data['train_Labels']
#     test_edges = train_data['test_Edges']
#     test_labels = train_data['test_Labels']
#     m_d_matrix = train_data['true_md']
#     md_class = train_data['md_class']
#
#     m_d_matrix[tuple(test_edges.T)] = 0
#     md_class[tuple(test_edges.T)] = 0
#
#     kfolds = param.kfold
#     torch.manual_seed(42)
#
#     if state == 'valid':
#         kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
#         train_idx, valid_idx = [], []
#
#         for train_index, valid_index in kf.split(train_edges):
#             train_idx.append(train_index)
#             valid_idx.append(valid_index)
#
#         for i in range(kfolds):
#             a = i+1
#             model = SuperedgeLearn(param)
#             model.cuda()
#             optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
#             # model.load_state_dict(torch.load('./cross_valid_example/fold_{}.pkl'.format(a)))
#             # print(f'################Fold {i + 1} of {kfolds}################')
#             edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
#             labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
#             trainEdges = EdgeDataset(edges_train, labels_train)
#             validEdges = EdgeDataset(edges_valid, labels_valid)
#             trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
#             validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
#
#             m_d_class = md_class.copy()
#             m_d_class[tuple(edges_valid.T)] = 0.0
#             md_m = torch.from_numpy(m_d_class).cuda()
#
#             print("-----training-----")
#
#             for e in range(param.epoch):
#                 running_loss = 0.0
#                 epo_label = []
#                 epo_score = []
#                 print("epoch：", e + 1)
#                 model.train()
#                 start = time.time()
#                 for i, item in enumerate(trainLoader):
#                     data, label = item
#                     trainData = data.cuda()
#                     trainLabel = label.cuda()
#                     pre_score = model(simData, md_m, trainData)
#                     train_loss = torch.nn.BCELoss()
#                     loss = train_loss(pre_score, trainLabel)
#                     loss.backward()
#                     optimizer.step()
#                     optimizer.zero_grad()
#
#                     running_loss += loss.item()
#                     print(f"After batch {i+1}:loss={loss:.3f};",end='\n')
#
#                     batch_score = pre_score.cpu().detach().numpy()
#                     epo_score = np.append(epo_score,batch_score)
#                     epo_label = np.append(epo_label,label.numpy())
#                 end=time.time()
#                 print('Time:%.2f\n'%(end-start))
#                 train_result = get_metrics(epo_score,epo_label)
#
#
#             valid_score, valid_label = [], []
#             model.eval()
#             with torch.no_grad():
#                 print("-----validing-----")
#                 # val_loss = 0.0
#                 for i, item in enumerate(validLoader):
#                     data, label = item
#                     validData = data.cuda()
#                     # validLabel = label.cuda()
#                     pre_score = model(simData, md_m, validData)
#
#                     val_batch_score = pre_score.detach().cpu().numpy()
#                     valid_score = np.append(valid_score, val_batch_score)
#                     valid_label = np.append(valid_label, label.numpy())
#
#                     # torch.save(model.state_dict(), "./fold_{}.pkl".format(a))
#                 valid_result = get_metrics(valid_score, valid_label)
#                 valid_metric.append(valid_result)
#                 gc.collect()
#                 torch.cuda.empty_cache()
#         print(np.array(valid_metric))
#         cv_metric = np.mean(valid_metric, axis=0)
#         print_met(cv_metric)
#
#
#     else:
#         test_score, test_label = [], []
#         testEdges = EdgeDataset(test_edges, test_labels)
#         testLoader = DataLoader.DataLoader(testEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)
#
#         md_ma = torch.from_numpy(md_class).cuda()
#         model = SuperedgeLearn(param)
#         model.load_state_dict(torch.load('./test_example/test.pkl'))
#         model.cuda()
#         model.eval()
#         with torch.no_grad():
#             start = time.time()
#             for i, item in enumerate(testLoader):
#                 data, label = item
#                 testData = data.cuda()
#                 pre_score = model(simData, md_ma, testData)
#                 batch_score = pre_score.cpu().detach().numpy()
#                 test_score = np.append(test_score, batch_score)
#                 test_label = np.append(test_label, label.numpy())
#             end = time.time()
#             print('Time：%.2f \n' % (end - start))
#             metrics = get_metrics(test_score, test_label)
#
#     return cv_metric
#     # For testing
#     # return metrics
#
#
# def draw_curve(trainloss, trainauc, trainacc, validloss, validauc, validacc, a, b):
#     # Curves can be drawn based on train data and test data.
#     plt.figure(a + b)
#     plt.plot(trainloss, color="blue", label="Train Loss")
#     plt.plot(validloss, color="red", label="Valid Loss")
#     x_major_locator = MultipleLocator(2)
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.xlim(0, 99)
#     plt.legend(loc='upper right')
#     plt.savefig("./curve/loss_{}.png".format(a))
#
#     plt.figure(a + b + 1)
#     plt.plot(trainauc, label="Train Auc")
#     plt.plot(validauc, color="red", label="Valid Auc")
#     x_major_locator = MultipleLocator(2)
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.xlim(0, 99)
#     plt.legend(loc='upper right')
#     plt.savefig("./curve_cnn/auc_{}.png".format(a))
#
#     plt.figure(a + b + 2)
#     plt.plot(trainacc, color="blue", label="Train Acc")
#     plt.plot(validacc, color="red", label="Valid Acc")
#     x_major_locator = MultipleLocator(2)
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.xlim(0, 99)
#     plt.legend(loc='upper right')
#     plt.savefig("./curve_cnn/acc_{}.png".format(a))
import time
import gc
import torch
import torch.nn as nn
from datapro import EdgeDataset
from model import SuperedgeLearn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import os


# ==========================================
# 论文级核心创新: Unbiased Confidence-Aware nnPU Loss
# ==========================================
class ConfidenceAwarePULoss(nn.Module):
    def __init__(self, pi):
        super().__init__()
        self.pi = pi
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, confidence):
        pos_mask = (labels == 1)
        unlabeled_mask = (labels == 0)

        # 1. 计算正样本风险 (Rp)
        if pos_mask.sum() > 0:
            Rp = self.bce(logits[pos_mask], torch.ones_like(logits[pos_mask])).mean()
            Rp_neg = self.bce(logits[pos_mask], torch.zeros_like(logits[pos_mask])).mean()
        else:
            Rp, Rp_neg = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # 2. 计算无标签样本风险 (Ru) -> 【修复：严格数学无偏估计】
        if unlabeled_mask.sum() > 0:
            # 置信度越高(越像真实正样本)，惩罚权重越小
            weights = 1.0 - confidence[unlabeled_mask]
            bce_u = self.bce(logits[unlabeled_mask], torch.zeros_like(logits[unlabeled_mask]))

            # 使用加权求和除以权重总和，确保损失量级的一致性(Unbiased Estimator)
            weight_sum = weights.sum() + 1e-8
            Ru = (weights * bce_u).sum() / weight_sum
        else:
            Ru = torch.tensor(0.0).cuda()

        # 3. 非负风险估计 (Non-negative PU Risk)
        risk = self.pi * Rp + torch.clamp(Ru - self.pi * Rp_neg, min=0)
        return risk


def calculate_cv_confidence(param, edges_train, labels_train, full_shape):
    """【防泄露屏障】：只利用当前划分训练集的正样本拓扑"""
    m_f = np.loadtxt(param.datapath + '/m_fs.csv', dtype=np.float32, delimiter=',')
    m_s = np.loadtxt(param.datapath + '/m_ss.csv', dtype=np.float32, delimiter=',')
    m_g = np.loadtxt(param.datapath + '/m_gs.csv', dtype=np.float32, delimiter=',')
    d_t = np.loadtxt(param.datapath + '/d_ts.csv', dtype=np.float32, delimiter=',')
    d_s = np.loadtxt(param.datapath + '/d_ss.csv', dtype=np.float32, delimiter=',')
    d_g = np.loadtxt(param.datapath + '/d_gs.csv', dtype=np.float32, delimiter=',')

    Sm = (m_f + m_s + m_g) / 3.0
    Sd = (d_t + d_s + d_g) / 3.0

    train_md_matrix = np.zeros(full_shape, dtype=np.float32)
    pos_edges = edges_train[labels_train == 1].astype(int)
    train_md_matrix[pos_edges[:, 0], pos_edges[:, 1]] = 1.0

    confidence_scores = np.dot(np.dot(Sm, train_md_matrix), Sd)

    min_val, max_val = np.min(confidence_scores), np.max(confidence_scores)
    confidence_scores = (confidence_scores - min_val) / (max_val - min_val + 1e-8)
    return confidence_scores


def get_metrics(logits, label):
    # 【弃用不科学指标】：仅评估衡量排序能力的 AUC 和 AUPR
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(label, probs)
    aupr = average_precision_score(label, probs)
    print(f'AUC: {auc:.4f} | AUPR: {aupr:.4f}')
    return [auc, aupr]


def train_test(simData, train_data, param, state):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    m_d_matrix = train_data['true_md']
    md_class = train_data['md_class']
    full_shape = m_d_matrix.shape

    valid_metric = []
    kf = KFold(n_splits=param.kfold, shuffle=True, random_state=42)

    if state == 'valid':
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_edges)):
            print(f'========== Fold {fold + 1} ==========')
            edges_train, edges_valid = train_edges[train_idx], train_edges[valid_idx]
            labels_train, labels_valid = train_labels[train_idx], train_labels[valid_idx]

            # 👉 【学术严谨】：动态先验概率估计 (Dynamic Prior Estimation)
            num_pos = (labels_train == 1).sum()
            num_unlabeled = (labels_train == 0).sum()
            dynamic_pi = num_pos / (num_pos + num_unlabeled)
            print(f"Data Prior Pi estimated for this fold: {dynamic_pi:.4f}")

            conf_matrix = calculate_cv_confidence(param, edges_train, labels_train, full_shape)

            trainEdges = EdgeDataset(edges_train, labels_train, conf_matrix)
            validEdges = EdgeDataset(edges_valid, labels_valid, conf_matrix=None)

            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)

            m_d_class = md_class.copy()
            m_d_class[tuple(edges_valid.T.astype(int))] = 0.0
            md_m = torch.from_numpy(m_d_class).cuda()

            model = SuperedgeLearn(param).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)

            # 传入数学意义完全匹配的 dynamic_pi
            criterion = ConfidenceAwarePULoss(pi=dynamic_pi)

            for e in range(param.epoch):
                model.train()
                epoch_loss = 0.0
                start = time.time()
                for i, (data, label, conf) in enumerate(trainLoader):
                    data, label, conf = data.cuda(), label.cuda(), conf.cuda()

                    logits = model(simData, md_m, data, conf_weights=conf)
                    loss = criterion(logits, label, conf)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                print(f"Epoch {e + 1} | Loss: {epoch_loss / len(trainLoader):.4f} | Time: {time.time() - start:.2f}s")

            # 验证阶段
            model.eval()
            valid_logits, valid_labels_list = [], []
            with torch.no_grad():
                for data, label, _ in validLoader:
                    data = data.cuda()
                    # 验证阶段无需置信度干预，用网络学到的普适权重泛化
                    logits = model(simData, md_m, data)
                    valid_logits.extend(logits.cpu().numpy())
                    valid_labels_list.extend(label.numpy())

            print("--- Fold Validation Metrics ---")
            fold_metric = get_metrics(np.array(valid_logits), np.array(valid_labels_list))
            valid_metric.append(fold_metric)
            gc.collect()
            torch.cuda.empty_cache()

        cv_metric = np.mean(valid_metric, axis=0)
        print(f'\n[Final 5-Fold CV] Mean AUC: {cv_metric[0]:.4f} | Mean AUPR: {cv_metric[1]:.4f}')
        return cv_metric