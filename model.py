# import numpy as np
# from otherlayers import *
# from extractSubGraph import GetSubgraph
#
#
# class SimMatrix(nn.Module):
#     def __init__(self, param):
#         super(SimMatrix, self).__init__()
#         self.mnum = param.m_num
#         self.dnum = param.d_num
#         self.viewn = param.view
#         self.attsim_m = SimAttention(self.mnum, self.mnum, self.viewn)
#         self.attsim_d = SimAttention(self.dnum, self.dnum, self.viewn)
#
#     def forward(self, data):
#         m_funsim = data['mm_f'].cuda()
#         m_seqsim = data['mm_s'].cuda()
#         m_gossim = data['mm_g'].cuda()
#         d_funsim = data['dd_t'].cuda()
#         d_semsim = data['dd_s'].cuda()
#         d_gossim = data['dd_g'].cuda()
#
#         m_sim = torch.stack((m_funsim, m_seqsim, m_gossim), 0)
#         d_sim = torch.stack((d_funsim, d_semsim, d_gossim), 0)
#         m_attsim = self.attsim_m(m_sim)
#         d_attsim = self.attsim_d(d_sim)
#
#         # Set the diagonal to 0.0 for subsequent sampling.
#         m_final_sim = m_attsim.fill_diagonal_(fill_value=0)
#         d_final_sim = d_attsim.fill_diagonal_(fill_value=0)
#
#         return m_final_sim, d_final_sim
#
#
# class SuperedgeLearn(nn.Module):
#     def __init__(self, param):
#         super(SuperedgeLearn, self).__init__()
#
#         self.hop = param.hop
#         self.neigh_size = param.nei_size
#         self.mNum = param.m_num
#         self.dNum = param.d_num
#         self.simClass = param.sim_class
#         self.mdClass = param.md_class
#         self.class_all = self.simClass + self.simClass + self.mdClass
#         self.NodeFea = param.feture_size
#         self.hinddenSize = param.atthidden_fea
#         self.edgeFea = param.edge_feature
#         self.drop = param.Dropout
#
#         self.actfun = nn.LeakyReLU(negative_slope=0.2)
#         self.actfun2 = nn.Sigmoid()
#
#         self.SimGet = SimMatrix(param)
#
#         self.edgeTran = OnehotTran(self.simClass, self.mdClass, self.mNum, self.dNum)
#         self.getSubgraph = GetSubgraph(self.neigh_size, self.hop)
#
#         self.EMBnode = NodeEmbedding(self.mNum, self.dNum, self.NodeFea, self.drop)
#         self.EMBedge = EdgeEmbedding(self.simClass, self.mdClass, self.neigh_size)
#
#         self.NeiAtt = NeiAttention(self.edgeFea, self.NodeFea, self.neigh_size)
#
#         self.Agg = NeiAggregator(self.NodeFea, self.drop, self.actfun)
#
#         self.ConSuperEdge = ConstructSuperEdge(self.edgeFea, self.class_all, self.NodeFea, self.hinddenSize)
#
#         self.fcLinear = MLP(self.hinddenSize + self.edgeFea, 1, self.drop, self.actfun)
#
#     def forward(self, simData, m_d, md_node):
#         # Get the similarity.
#         m_sim, d_sim = self.SimGet(simData)
#         # The original association matrix can be used for edge selection.
#         prep_one = torch.cat((m_sim, m_d), dim=1)
#         prep_two = torch.cat((m_d.t(), d_sim), dim=1)
#         md_all = torch.cat((prep_one, prep_two), dim=0)
#
#         # Redefine the index of the node.
#         m_node = md_node[:, 0]
#         d_node = md_node[:, 1] + self.mNum
#
#         relation_adj = self.edgeTran(m_sim, d_sim, m_d)
#         # Subgraph extraction.
#         m_neinode_list, m_neirel_list, d_neinode_list, d_neirel_list = self.getSubgraph(m_node, d_node, md_all,
#                                                                                         relation_adj)
#
#         # Get embedding representation.
#         m_nodeemb_list = self.EMBnode(m_sim, d_sim, m_neinode_list)
#         d_nodeemb_list = self.EMBnode(m_sim, d_sim, d_neinode_list)
#         m_relemb_list = self.EMBedge(m_neirel_list)
#         d_relemb_list = self.EMBedge(d_neirel_list)
#
#         # Gather long distance information by node-aware attention.
#         for i in range(self.hop - 1, 0, -1):
#             mneigh_update_emb = self.NeiAtt(m_nodeemb_list[i], m_relemb_list[i], m_nodeemb_list[i + 1], i)
#             dneigh_update_emb = self.NeiAtt(d_nodeemb_list[i], d_relemb_list[i], d_nodeemb_list[i + 1], i)
#
#             m_nodeemb_list[i] = self.Agg(m_nodeemb_list[i], mneigh_update_emb)
#             d_nodeemb_list[i] = self.Agg(d_nodeemb_list[i], dneigh_update_emb)
#
#         # Hyperedge-aware attention aggregates direct neighborhood information to learn and identification.
#         md_edge_final = self.ConSuperEdge(m_nodeemb_list, d_nodeemb_list, m_relemb_list, d_relemb_list)
#         md_edge_score = self.fcLinear(md_edge_final)
#         pre_score = self.actfun2(md_edge_score).squeeze(dim=1)
#
#         return pre_score
#
#
# class ConstructSuperEdge(nn.Module):
#     def __init__(self, edgeFea, class_all, nodeFea, hsize):
#         super(ConstructSuperEdge, self).__init__()
#
#         self.class_all = class_all
#         self.nodeFea = nodeFea
#         self.edgeFea = edgeFea
#         self.hidden = hsize
#         self.edgeLinear = nn.Linear(self.nodeFea * 2, self.edgeFea)
#         self.act = nn.ReLU()
#         self.Att = Attention(self.edgeFea, self.nodeFea, self.hidden)
#
#     def forward(self, mnode_list, dnode_list, mrel_list, drel_list):
#         mi_emb = mnode_list[0]
#         dj_emb = dnode_list[0]
#         pre_md_emb = torch.cat((mi_emb, dj_emb), 2)
#
#         edge_emb = self.edgeLinear(pre_md_emb)
#         edge_emb = self.act(edge_emb)
#
#         edge_nei_node = torch.cat((mnode_list[1], dnode_list[1]), 1)
#         edge_nei_rel = torch.cat((mrel_list[0], drel_list[0]), 1)
#         edge_nei = torch.cat((edge_nei_rel, edge_nei_node), 2)
#
#         edge_nei_info = self.Att(edge_emb, edge_nei)
#         vir_edge = torch.cat((edge_emb.squeeze(dim=1), edge_nei_info), dim=-1)
#
#         return vir_edge
import torch
import torch.nn as nn
from otherlayers import *
from extractSubGraph import GetSubgraph


class SimMatrix(nn.Module):
    def __init__(self, param):
        super(SimMatrix, self).__init__()
        self.mnum = param.m_num
        self.dnum = param.d_num
        self.viewn = param.view
        self.attsim_m = SimAttention(self.mnum, self.mnum, self.viewn)
        self.attsim_d = SimAttention(self.dnum, self.dnum, self.viewn)

    def forward(self, data):
        m_sim = torch.stack((data['mm_f'].cuda(), data['mm_s'].cuda(), data['mm_g'].cuda()), 0)
        d_sim = torch.stack((data['dd_t'].cuda(), data['dd_s'].cuda(), data['dd_g'].cuda()), 0)
        m_final_sim = self.attsim_m(m_sim).fill_diagonal_(0)
        d_final_sim = self.attsim_d(d_sim).fill_diagonal_(0)
        return m_final_sim, d_final_sim


class SuperedgeLearn(nn.Module):
    def __init__(self, param):
        super(SuperedgeLearn, self).__init__()
        self.hop = param.hop
        self.neigh_size = param.nei_size
        self.mNum = param.m_num
        self.dNum = param.d_num
        self.simClass = param.sim_class
        self.mdClass = param.md_class
        self.class_all = self.simClass * 2 + self.mdClass
        self.NodeFea = param.feture_size
        self.hinddenSize = param.atthidden_fea
        self.edgeFea = param.edge_feature
        self.drop = param.Dropout

        self.actfun = nn.LeakyReLU(negative_slope=0.2)

        self.SimGet = SimMatrix(param)
        self.edgeTran = OnehotTran(self.simClass, self.mdClass, self.mNum, self.dNum)
        self.getSubgraph = GetSubgraph(self.neigh_size, self.hop)
        self.EMBnode = NodeEmbedding(self.mNum, self.dNum, self.NodeFea, self.drop)
        self.EMBedge = EdgeEmbedding(self.simClass, self.mdClass, self.neigh_size)
        self.NeiAtt = NeiAttention(self.edgeFea, self.NodeFea, self.neigh_size)
        self.Agg = NeiAggregator(self.NodeFea, self.drop, self.actfun)

        self.ConSuperEdge = ConstructSuperEdge(self.edgeFea, self.class_all, self.NodeFea, self.hinddenSize)
        # 移除了 Sigmoid，直接输出 Logits 以确保稳定
        self.scorer = nn.Linear(self.hinddenSize + self.edgeFea, 1)

    def forward(self, simData, m_d, md_node, conf_weights=None):
        m_sim, d_sim = self.SimGet(simData)

        prep_one = torch.cat((m_sim, m_d), dim=1)
        prep_two = torch.cat((m_d.t(), d_sim), dim=1)
        md_all = torch.cat((prep_one, prep_two), dim=0)

        m_node = md_node[:, 0]
        d_node = md_node[:, 1] + self.mNum
        relation_adj = self.edgeTran(m_sim, d_sim, m_d)

        m_neinode_list, m_neirel_list, d_neinode_list, d_neirel_list = self.getSubgraph(
            m_node, d_node, md_all, relation_adj)

        m_nodeemb_list = self.EMBnode(m_sim, d_sim, m_neinode_list)
        d_nodeemb_list = self.EMBnode(m_sim, d_sim, d_neinode_list)
        m_relemb_list = self.EMBedge(m_neirel_list)
        d_relemb_list = self.EMBedge(d_neirel_list)

        for i in range(self.hop - 1, 0, -1):
            mneigh_update_emb = self.NeiAtt(m_nodeemb_list[i], m_relemb_list[i], m_nodeemb_list[i + 1], i)
            dneigh_update_emb = self.NeiAtt(d_nodeemb_list[i], d_relemb_list[i], d_nodeemb_list[i + 1], i)
            m_nodeemb_list[i] = self.Agg(m_nodeemb_list[i], mneigh_update_emb)
            d_nodeemb_list[i] = self.Agg(d_nodeemb_list[i], dneigh_update_emb)

        # 【论文核心点传递】：将 conf_weights 传入超边构建阶段
        md_edge_final = self.ConSuperEdge(m_nodeemb_list, d_nodeemb_list, m_relemb_list, d_relemb_list, conf_weights)

        pre_logits = self.scorer(md_edge_final).squeeze(dim=1)
        return pre_logits


class ConstructSuperEdge(nn.Module):
    def __init__(self, edgeFea, class_all, nodeFea, hsize):
        super(ConstructSuperEdge, self).__init__()
        self.class_all, self.nodeFea, self.edgeFea, self.hidden = class_all, nodeFea, edgeFea, hsize
        self.edgeLinear = nn.Linear(self.nodeFea * 2, self.edgeFea)
        self.act = nn.ReLU()
        self.Att = Attention(self.edgeFea, self.nodeFea, self.hidden)

    # 【论文核心点落地】：引入 conf_weights 调控图结构信息流动
    def forward(self, mnode_list, dnode_list, mrel_list, drel_list, conf_weights=None):
        pre_md_emb = torch.cat((mnode_list[0], dnode_list[0]), 2)
        edge_emb = self.act(self.edgeLinear(pre_md_emb))

        edge_nei_node = torch.cat((mnode_list[1], dnode_list[1]), 1)
        edge_nei_rel = torch.cat((mrel_list[0], drel_list[0]), 1)
        edge_nei = torch.cat((edge_nei_rel, edge_nei_node), 2)

        # 获取超边邻居的信息聚合
        edge_nei_info = self.Att(edge_emb, edge_nei)

        # 👉 【重磅改进】：置信度直接调控超边注意力机制的消息传递权重 (Message Passing Weight)
        if conf_weights is not None:
            conf_weights = conf_weights.view(-1, 1)
            # 置信度低的边，阻断其传播假阴性噪声
            edge_nei_info = edge_nei_info * conf_weights

        return torch.cat((edge_emb.squeeze(dim=1), edge_nei_info), dim=-1)