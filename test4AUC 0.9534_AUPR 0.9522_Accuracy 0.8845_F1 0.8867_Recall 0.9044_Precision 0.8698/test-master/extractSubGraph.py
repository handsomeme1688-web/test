import torch
from torch import nn


def select_neighbors(node_weight, rel_adj, neigh_size):
    available = (node_weight > 0).sum(dim=1)
    min_available = int(available.min().item()) if available.numel() else 0
    if min_available > 0:
        sample_size = min(neigh_size, min_available, node_weight.size(1))
    else:
        sample_size = min(neigh_size, node_weight.size(1))
    sample_size = max(1, sample_size)

    weights, node_sample = torch.topk(node_weight, k=sample_size, dim=1)
    rel_sample = rel_adj.gather(1, node_sample).long() - 1

    valid_mask = weights > 0
    if not torch.all(valid_mask):
        fallback_nodes = node_sample[:, :1].expand_as(node_sample)
        fallback_rels = rel_sample[:, :1].expand_as(rel_sample)
        node_sample = torch.where(valid_mask, node_sample, fallback_nodes)
        rel_sample = torch.where(valid_mask, rel_sample, fallback_rels)

    rel_sample = rel_sample.clamp_min(0)
    return node_sample, rel_sample


class GetSubgraph(nn.Module):
    def __init__(self, nei_size, hop):
        super(GetSubgraph, self).__init__()
        self.neigh_size = nei_size
        self.hop = hop

    def forward(self, m_node, d_node, node_adj, rel_adj):
        md_node = torch.cat((m_node, d_node), dim=0)
        dm_node = torch.cat((d_node, m_node), dim=0)

        node_mask = node_adj.ne(0).float()
        node_mask[md_node, dm_node] = 0.0
        node_weight = node_adj.abs() * node_mask

        mnei_emb_list = [m_node.unsqueeze(dim=1)]
        dnei_emb_list = [d_node.unsqueeze(dim=1)]
        mrel_emb_list, drel_emb_list = [], []

        for hop_index in range(self.hop):
            now_nei_size = self.neigh_size[hop_index]
            nei_node_adj, rel_node_adj = select_neighbors(node_weight, rel_adj, now_nei_size)

            m_subnode, m_subrel = getNeiRel(mnei_emb_list[-1], nei_node_adj, rel_node_adj)
            d_subnode, d_subrel = getNeiRel(dnei_emb_list[-1], nei_node_adj, rel_node_adj)

            mnei_emb_list.append(m_subnode)
            dnei_emb_list.append(d_subnode)
            mrel_emb_list.append(m_subrel)
            drel_emb_list.append(d_subrel)

        return mnei_emb_list, mrel_emb_list, dnei_emb_list, drel_emb_list


def getNeiRel(nodes_index, nei_node, nei_rel):
    node_index = nodes_index.reshape(-1)
    node_neigh = torch.index_select(nei_node, 0, node_index)
    rel_neigh = torch.index_select(nei_rel, 0, node_index)

    node_subnei = node_neigh.reshape(nodes_index.size(0), -1)
    rel_subnei = rel_neigh.reshape(nodes_index.size(0), -1)
    return node_subnei, rel_subnei
