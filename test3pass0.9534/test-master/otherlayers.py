from math import sqrt

import torch
import torch.nn as nn


class SimAttention(nn.Module):
    def __init__(self, num, feature, viewNum):
        super(SimAttention, self).__init__()
        self.Num = num
        self.FeaSize = feature
        self.viewn = viewNum
        self.dropout = nn.Dropout(0.3)
        self.fc_1 = nn.Linear(self.viewn, 150 * self.viewn, bias=False)
        self.fc_2 = nn.Linear(150 * self.viewn, self.viewn, bias=False)
        self.GAP1 = nn.AvgPool2d((self.Num, self.Num), (1, 1))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, similarity):
        avr_pool = self.GAP1(similarity)
        sim_atten = avr_pool.reshape(-1, avr_pool.size(0))
        sim_atten = self.fc_1(sim_atten)
        sim_atten = self.relu(sim_atten)
        sim_atten = self.fc_2(sim_atten)
        sim_atten = sim_atten.reshape(similarity.size(0), 1, 1)
        all_att = self.softmax(sim_atten)
        sim = all_att * similarity
        return torch.sum(sim, dim=0, keepdim=False)


class OnehotTran(nn.Module):
    def __init__(self, sim_class, md_class, m_num, d_num):
        super(OnehotTran, self).__init__()
        self.m_class = sim_class
        self.d_class = sim_class
        self.md_class = md_class
        self.class_all = self.m_class + self.d_class + self.md_class
        self.M_num = m_num
        self.D_num = d_num

    def forward(self, m_score, d_score, md_score):
        mnew_score = torch.zeros_like(m_score)
        mnew_score = torch.where((m_score > 0.0) & (m_score < 0.35), torch.full_like(m_score, 1.0), mnew_score)
        mnew_score = torch.where((m_score >= 0.35) & (m_score < 0.65), torch.full_like(m_score, 2.0), mnew_score)
        mnew_score = torch.where(m_score >= 0.65, torch.full_like(m_score, 3.0), mnew_score)

        dnew_score = torch.zeros_like(d_score)
        dnew_score = torch.where((d_score > 0.0) & (d_score < 0.35), torch.full_like(d_score, 4.0), dnew_score)
        dnew_score = torch.where((d_score >= 0.35) & (d_score < 0.65), torch.full_like(d_score, 5.0), dnew_score)
        dnew_score = torch.where(d_score >= 0.65, torch.full_like(d_score, 6.0), dnew_score)

        mdnew_score = torch.zeros_like(md_score)
        mdnew_score = torch.where(md_score == -1.0, torch.full_like(md_score, 7.0), mdnew_score)
        mdnew_score = torch.where(md_score == 1.0, torch.full_like(md_score, 8.0), mdnew_score)
        mdnew_score = torch.where(md_score == 2.0, torch.full_like(md_score, 9.0), mdnew_score)

        pre_one = torch.cat((mnew_score, mdnew_score), dim=1)
        pre_two = torch.cat((mdnew_score.t(), dnew_score), dim=1)
        return torch.cat((pre_one, pre_two), dim=0)


class NodeEmbedding(nn.Module):
    def __init__(self, m_num, d_num, feature, dropout):
        super(NodeEmbedding, self).__init__()
        self.m_Num = m_num
        self.d_Num = d_num
        self.node_voca_num = m_num + d_num
        self.fea_Size = feature
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.node_voca_num, self.fea_Size)
        self.relu = nn.ReLU()

    def forward(self, m_sim, d_sim, nei_node_list):
        batch_size = nei_node_list[0].size(0)
        device = m_sim.device
        md_f = torch.zeros(self.m_Num, self.d_Num, device=device)
        prep_m_f = torch.cat((m_sim, md_f), dim=1)
        prep_d_f = torch.cat((md_f.t(), d_sim), dim=1)
        m_d_f = torch.cat((prep_m_f, prep_d_f), dim=0)
        md_node_fea = self.drop(self.relu(self.linear(m_d_f)))

        neinode_emb_list = []
        for nei_node in nei_node_list:
            nei_node = nei_node.reshape(-1)
            nei_node_emb = torch.index_select(md_node_fea, 0, nei_node)
            neinode_emb_list.append(nei_node_emb.reshape(batch_size, -1, self.fea_Size))
        return neinode_emb_list


class EdgeEmbedding(nn.Module):
    def __init__(self, sim_class, md_class, nei_size):
        super(EdgeEmbedding, self).__init__()
        self.m_class = sim_class
        self.d_class = sim_class
        self.md_class = md_class
        self.class_all = self.m_class + self.d_class + self.md_class
        self.neigh_size = nei_size
        bottom = torch.arange(start=0, end=self.class_all, step=1)
        bottom_onehot = torch.nn.functional.one_hot(bottom, self.class_all).float()
        self.register_buffer('bottom_onehot', bottom_onehot)

    def forward(self, nei_rel_list):
        batch_size = nei_rel_list[0].size(0)
        neirel_emb_list = []
        for nei_relation in nei_rel_list:
            nei_relation = nei_relation.reshape(-1)
            nei_rel_emb = torch.index_select(self.bottom_onehot, 0, nei_relation)
            neirel_emb_list.append(nei_rel_emb.reshape(batch_size, -1, self.class_all))
        return neirel_emb_list


class NeiAttention(nn.Module):
    def __init__(self, edgeFea, nodeFea, nei_size):
        super(NeiAttention, self).__init__()
        self.neigh_size = nei_size
        self.norm = 1 / sqrt(nodeFea)
        self.W1 = nn.Linear(edgeFea + nodeFea, nodeFea)
        self.actfun = nn.Softmax(dim=-1)

    def forward(self, x, x_nei_rel, x_nei_node, i):
        now_nei_size = self.neigh_size[i]
        n_neibor = int(x_nei_node.shape[1] / now_nei_size)
        x = x.unsqueeze(dim=2)
        x_nei = torch.cat((x_nei_rel, x_nei_node), dim=-1)
        x_nei_up = self.W1(x_nei)
        x_nei_val = x_nei_up.reshape(x.shape[0], n_neibor, now_nei_size, -1)
        alpha = torch.matmul(x, x_nei_val.permute(0, 1, 3, 2)) * self.norm
        alpha = self.actfun(alpha)
        alpha = alpha.permute(0, 1, 3, 2)
        out = alpha * x_nei_val
        return torch.sum(out, dim=2, keepdim=False)


class NeiAggregator(nn.Module):
    def __init__(self, nodeFea, dropout, actFunc, outBn=False, outAct=True, outDp=True):
        super(NeiAggregator, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(nodeFea)
        self.out = nn.Linear(nodeFea * 2, nodeFea)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x, x_nei):
        x_new = torch.cat((x, x_nei), dim=-1)
        x_new = self.out(x_new)
        if self.outBn:
            if len(x_new.shape) == 3:
                x_new = self.bns(x_new.transpose(1, 2)).transpose(1, 2)
            else:
                x_new = self.bns(x_new)
        if self.outAct:
            x_new = self.actFunc(x_new)
        if self.outDp:
            x_new = self.dropout(x_new)
        return x_new


class Attention(nn.Module):
    def __init__(self, edgeinSize, NodeinSize, outSize):
        super(Attention, self).__init__()
        self.edgeInSize = edgeinSize
        self.NodeInsize = NodeinSize
        self.outSize = outSize
        self.q = nn.Linear(self.edgeInSize, outSize)
        self.k = nn.Linear(self.NodeInsize + self.edgeInSize, outSize)
        self.v = nn.Linear(self.NodeInsize + self.edgeInSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=-1)

    def forward(self, query, input_tensor):
        Q = self.q(query)
        K = self.k(input_tensor)
        V = self.v(input_tensor)
        alpha = torch.bmm(Q, K.transpose(1, 2)) * self.norm
        alpha = self.actfun1(alpha)
        return torch.bmm(alpha, V).squeeze(dim=1)


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=False, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)
        if self.outBn:
            x = self.bns(x)
        if self.outAct:
            x = self.actFunc(x)
        if self.outDp:
            x = self.dropout(x)
        return x
