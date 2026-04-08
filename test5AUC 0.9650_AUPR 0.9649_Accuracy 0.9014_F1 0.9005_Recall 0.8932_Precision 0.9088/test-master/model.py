from math import sqrt

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


class SemanticViewFusion(nn.Module):
    def __init__(self, feature_dim, dropout):
        super(SemanticViewFusion, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.norm = 1.0 / sqrt(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, center_feat, view_tokens):
        query = self.query(center_feat).unsqueeze(dim=1)
        keys = self.key(view_tokens)
        values = self.value(view_tokens)

        alpha = torch.softmax((query * keys).sum(dim=-1) * self.norm, dim=1).unsqueeze(dim=-1)
        fused = (alpha * values).sum(dim=1)
        disagreement = view_tokens.std(dim=1)
        return self.out(torch.cat((fused, disagreement), dim=-1))


class ExternalSemanticHypergraph(nn.Module):
    def __init__(self, param):
        super(ExternalSemanticHypergraph, self).__init__()
        self.m_num = param.m_num
        self.d_num = param.d_num
        self.topk = getattr(param, 'semantic_topk', 32)
        self.node_fea = param.feture_size
        self.dropout = nn.Dropout(param.Dropout)

        self.m_proj = nn.ModuleDict({
            'mm_f': nn.Linear(self.node_fea, self.node_fea),
            'mm_s': nn.Linear(self.node_fea, self.node_fea),
            'mm_g': nn.Linear(self.node_fea, self.node_fea),
        })
        self.d_proj = nn.ModuleDict({
            'dd_t': nn.Linear(self.node_fea, self.node_fea),
            'dd_s': nn.Linear(self.node_fea, self.node_fea),
            'dd_g': nn.Linear(self.node_fea, self.node_fea),
        })
        self.m_fusion = SemanticViewFusion(self.node_fea, param.Dropout)
        self.d_fusion = SemanticViewFusion(self.node_fea, param.Dropout)
        self.pair_mlp = nn.Sequential(
            nn.Linear(self.node_fea * 6 + 1, param.semantic_hidden),
            nn.ReLU(),
            nn.Dropout(param.Dropout),
            nn.Linear(param.semantic_hidden, param.semantic_hidden),
            nn.ReLU(),
            nn.Dropout(param.Dropout),
        )
        self.cache = {}

    def _topk_cache(self, name, sim_matrix):
        key = (name, sim_matrix.device.type, sim_matrix.device.index)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        sim = sim_matrix.clone()
        sim.fill_diagonal_(0.0)
        k = min(self.topk, sim.size(1) - 1)
        weights, indices = torch.topk(sim, k=k, dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.cache[key] = (indices, weights)
        return indices, weights

    def _aggregate_view(self, name, sim_matrix, node_indices, feature_matrix, proj):
        knn_idx, knn_weight = self._topk_cache(name, sim_matrix)
        batch_idx = knn_idx[node_indices]
        batch_weight = knn_weight[node_indices].unsqueeze(dim=-1)
        neigh_feat = feature_matrix[batch_idx]
        token = (batch_weight * neigh_feat).sum(dim=1)
        return self.dropout(torch.relu(proj(token)))

    def forward(self, sim_data, all_node_feat, m_node, d_node, mi_emb, dj_emb, pair_confidence):
        mi_feat = all_node_feat[:self.m_num]
        di_feat = all_node_feat[self.m_num:]

        m_views = []
        for name, proj in self.m_proj.items():
            m_views.append(self._aggregate_view(name, sim_data[name], m_node, mi_feat, proj))
        d_views = []
        for name, proj in self.d_proj.items():
            d_views.append(self._aggregate_view(name, sim_data[name], d_node, di_feat, proj))

        m_view_tokens = torch.stack(m_views, dim=1)
        d_view_tokens = torch.stack(d_views, dim=1)

        mi_sem = self.m_fusion(mi_emb, m_view_tokens)
        dj_sem = self.d_fusion(dj_emb, d_view_tokens)

        pair_input = torch.cat((
            mi_emb,
            dj_emb,
            mi_sem,
            dj_sem,
            torch.abs(mi_sem - dj_sem),
            mi_sem * dj_sem,
            pair_confidence.unsqueeze(dim=1),
        ), dim=-1)
        return self.pair_mlp(pair_input)


class ConstructBaselineSuperEdge(nn.Module):
    def __init__(self, edgeFea, class_all, nodeFea, hsize):
        super(ConstructBaselineSuperEdge, self).__init__()
        self.class_all = class_all
        self.nodeFea = nodeFea
        self.edgeFea = edgeFea
        self.hidden = hsize
        self.edgeLinear = nn.Linear(self.nodeFea * 2, self.edgeFea)
        self.act = nn.ReLU()
        self.LocalAtt = Attention(self.edgeFea, self.nodeFea, self.hidden)

    def forward(self, mnode_list, dnode_list, mrel_list, drel_list):
        pre_md_emb = torch.cat((mnode_list[0], dnode_list[0]), dim=2)
        edge_emb = self.act(self.edgeLinear(pre_md_emb))
        edge_nei_node = torch.cat((mnode_list[1], dnode_list[1]), dim=1)
        edge_nei_rel = torch.cat((mrel_list[0], drel_list[0]), dim=1)
        edge_nei = torch.cat((edge_nei_rel, edge_nei_node), dim=2)
        local_edge = self.LocalAtt(edge_emb, edge_nei)
        return torch.cat((edge_emb.squeeze(dim=1), local_edge), dim=-1)


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
        self.semanticHidden = param.semantic_hidden
        self.hypergraph_mode = getattr(param, 'hypergraph_mode', 'baseline')
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
        self.StructEdge = ConstructBaselineSuperEdge(
            edgeFea=self.edgeFea,
            class_all=self.class_all,
            nodeFea=self.NodeFea,
            hsize=self.hinddenSize,
        )

        if self.hypergraph_mode == 'bio_semantic':
            self.BioHypergraph = ExternalSemanticHypergraph(param)
            scorer_in = self.hinddenSize + self.edgeFea + self.semanticHidden
        else:
            self.BioHypergraph = None
            scorer_in = self.hinddenSize + self.edgeFea
        self.scorer = nn.Linear(scorer_in, 1)

    def forward(self, simData, m_d, md_node, pair_confidence=None):
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
            mneigh_update_emb = self.NeiAtt(
                m_nodeemb_list[hop_index],
                m_relemb_list[hop_index],
                m_nodeemb_list[hop_index + 1],
                hop_index,
            )
            dneigh_update_emb = self.NeiAtt(
                d_nodeemb_list[hop_index],
                d_relemb_list[hop_index],
                d_nodeemb_list[hop_index + 1],
                hop_index,
            )
            m_nodeemb_list[hop_index] = self.Agg(m_nodeemb_list[hop_index], mneigh_update_emb)
            d_nodeemb_list[hop_index] = self.Agg(d_nodeemb_list[hop_index], dneigh_update_emb)

        struct_token = self.StructEdge(m_nodeemb_list, d_nodeemb_list, m_relemb_list, d_relemb_list)

        if self.BioHypergraph is None:
            final_token = struct_token
        else:
            if pair_confidence is None:
                pair_confidence = struct_token.new_full((struct_token.size(0),), 0.5)
            all_node_feat = self.EMBnode.build_feature_matrix(m_sim, d_sim)
            bio_token = self.BioHypergraph(
                sim_data=simData,
                all_node_feat=all_node_feat,
                m_node=md_node[:, 0],
                d_node=md_node[:, 1],
                mi_emb=m_nodeemb_list[0].squeeze(dim=1),
                dj_emb=d_nodeemb_list[0].squeeze(dim=1),
                pair_confidence=pair_confidence.float().clamp_(0.0, 1.0),
            )
            final_token = torch.cat((struct_token, bio_token), dim=-1)

        return self.scorer(final_token).squeeze(dim=1)
