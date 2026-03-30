import torch
import torch.nn as nn

from extractSubGraph import GetSubgraph
from otherlayers import Attention, EdgeEmbedding, NeiAggregator, NeiAttention, NodeEmbedding, OnehotTran, SimAttention


class SimMatrix(nn.Module):
    def __init__(self, param):
        super(SimMatrix, self).__init__()
        self.mnum = param.m_num
        self.dnum = param.d_num
        self.viewn = param.view
        self.topk = getattr(param, 'sim_topk', 0)
        self.attsim_m = SimAttention(self.mnum, self.mnum, self.viewn)
        self.attsim_d = SimAttention(self.dnum, self.dnum, self.viewn)

    def _sparsify(self, sim):
        sim = sim.clone()
        sim.fill_diagonal_(0.0)

        if self.topk <= 0 or self.topk >= sim.size(1):
            return sim

        k = min(self.topk, sim.size(1) - 1)
        _, topk_idx = torch.topk(sim, k=k, dim=1)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_idx, 1.0)

        sparse_sim = sim * mask
        sparse_sim = torch.maximum(sparse_sim, sparse_sim.transpose(0, 1))
        sparse_sim.fill_diagonal_(0.0)
        return sparse_sim

    def forward(self, data):
        m_sim = torch.stack((data['mm_f'], data['mm_s'], data['mm_g']), dim=0)
        d_sim = torch.stack((data['dd_t'], data['dd_s'], data['dd_g']), dim=0)

        m_final_sim = self.attsim_m(m_sim)
        d_final_sim = self.attsim_d(d_sim)
        return self._sparsify(m_final_sim), self._sparsify(d_final_sim)


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
        self.scorer = nn.Linear(self.hinddenSize + self.edgeFea, 1)

    def forward(self, simData, m_d, md_node):
        m_sim, d_sim = self.SimGet(simData)

        prep_one = torch.cat((m_sim, m_d), dim=1)
        prep_two = torch.cat((m_d.t(), d_sim), dim=1)
        md_all = torch.cat((prep_one, prep_two), dim=0)

        m_node = md_node[:, 0]
        d_node = md_node[:, 1] + self.mNum
        relation_adj = self.edgeTran(m_sim, d_sim, m_d)

        m_neinode_list, m_neirel_list, d_neinode_list, d_neirel_list = self.getSubgraph(
            m_node, d_node, md_all, relation_adj
        )

        m_nodeemb_list = self.EMBnode(m_sim, d_sim, m_neinode_list)
        d_nodeemb_list = self.EMBnode(m_sim, d_sim, d_neinode_list)
        m_relemb_list = self.EMBedge(m_neirel_list)
        d_relemb_list = self.EMBedge(d_neirel_list)

        for hop_index in range(self.hop - 1, 0, -1):
            mneigh_update_emb = self.NeiAtt(m_nodeemb_list[hop_index], m_relemb_list[hop_index], m_nodeemb_list[hop_index + 1], hop_index)
            dneigh_update_emb = self.NeiAtt(d_nodeemb_list[hop_index], d_relemb_list[hop_index], d_nodeemb_list[hop_index + 1], hop_index)
            m_nodeemb_list[hop_index] = self.Agg(m_nodeemb_list[hop_index], mneigh_update_emb)
            d_nodeemb_list[hop_index] = self.Agg(d_nodeemb_list[hop_index], dneigh_update_emb)

        md_edge_final = self.ConSuperEdge(m_nodeemb_list, d_nodeemb_list, m_relemb_list, d_relemb_list)
        return self.scorer(md_edge_final).squeeze(dim=1)


class ConstructSuperEdge(nn.Module):
    def __init__(self, edgeFea, class_all, nodeFea, hsize):
        super(ConstructSuperEdge, self).__init__()
        self.class_all = class_all
        self.nodeFea = nodeFea
        self.edgeFea = edgeFea
        self.hidden = hsize
        self.edgeLinear = nn.Linear(self.nodeFea * 2, self.edgeFea)
        self.act = nn.ReLU()
        self.Att = Attention(self.edgeFea, self.nodeFea, self.hidden)

    def forward(self, mnode_list, dnode_list, mrel_list, drel_list):
        pre_md_emb = torch.cat((mnode_list[0], dnode_list[0]), dim=2)
        edge_emb = self.act(self.edgeLinear(pre_md_emb))

        edge_nei_node = torch.cat((mnode_list[1], dnode_list[1]), dim=1)
        edge_nei_rel = torch.cat((mrel_list[0], drel_list[0]), dim=1)
        edge_nei = torch.cat((edge_nei_rel, edge_nei_node), dim=2)
        edge_nei_info = self.Att(edge_emb, edge_nei)
        return torch.cat((edge_emb.squeeze(dim=1), edge_nei_info), dim=-1)
