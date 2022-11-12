import random
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, GINConv, GATConv
import torch.nn.functional as F
from src.utils import get_activation


class GNN_Encoder(nn.Module):
    def __init__(self, args, num_forward_layer, no_bn):
        super(GNN_Encoder, self).__init__()
        self.args = args
        self.num_forward_layer = num_forward_layer
        self.use_bn = not no_bn
        self.dropout = args.dropout

        self.forward_pass = [GCNConv(args.num_features, args.hidden)]
        if self.use_bn:
            self.bns = [nn.BatchNorm1d(args.hidden)]
        for _ in range(1, num_forward_layer):
            self.forward_pass.append(GCNConv(args.hidden, args.hidden))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(args.hidden))
        self.forward_pass = nn.ModuleList(self.forward_pass)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)

        self.activation = get_activation(args.activation)
        self.output_dim = args.hidden


    def forward(self, x, edge_index):
        for i in range(self.num_forward_layer):
            x = self.forward_pass[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x



class SPGCL(nn.Module):
    def __init__(self, args):
        super(SPGCL, self).__init__()
        self.args = args
        self.encoder = GNN_Encoder(args, args.encoder_layer_num, args.no_bn)

        if self.args.num_proj_layer == 1:
            self.fc_pipe = torch.nn.Sequential(
                torch.nn.Linear(args.hidden, args.hidden)
            )
        elif self.args.num_proj_layer == 2:
            self.fc_pipe = torch.nn.Sequential(
                torch.nn.Linear(args.hidden, args.hidden),
                get_activation(args.proj_activation),
                torch.nn.Linear(args.hidden, args.hidden)
            )
        elif self.args.num_proj_layer == 3:
            self.fc_pipe = torch.nn.Sequential(
                torch.nn.Linear(args.hidden, args.hidden),
                get_activation(args.proj_activation),
                torch.nn.Linear(args.hidden, args.hidden),
                get_activation(args.proj_activation),
                torch.nn.Linear(args.hidden, args.hidden),
            )


    def projection(self, z: torch.Tensor):
        return self.fc_pipe(z)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embed = self.encoder(x, edge_index)
        return embed


    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        embed = self.encoder(x, edge_index)
        return embed.detach()


    def obtain_proj_embedding(self, data):
        with torch.no_grad():
            orig_node_embed_full = self.encoder(data.x, data.edge_index)
            if self.args.pre_proj:
                node_embed_full = self.projection(F.relu(orig_node_embed_full))
            else:
                node_embed_full = orig_node_embed_full
            norm_node_embed_full = F.normalize(node_embed_full, p=2, dim=-1)
        return norm_node_embed_full.detach()


    def model_train(self, data, optimiser):
        optimiser.zero_grad()
        self.MAX_SIZE = self.args.max_size
        orig_node_embed_full = self.encoder(data.x, data.edge_index)

        if self.args.seed_sampling == 'random':
            sample_idx = np.array(random.sample(list(range(data.x.shape[0])), min(self.MAX_SIZE, data.x.shape[0])))
        else: # tree
            sample_idx_ = torch.concat([data.subgraph_cache[i] for i in
                                        random.sample(list(range(data.x.shape[0])), self.args.seed_num)]).tolist()
            # prevent neighbor explosion
            sample_idx = np.array(random.sample(sample_idx_, min(self.MAX_SIZE, len(sample_idx_)) ))

        if self.args.square_subg:
            node_embed_sample = self.projection(F.relu(orig_node_embed_full[sample_idx]))
            norm_node_embed = F.normalize(node_embed_sample, p=2, dim=-1)
            sim_matrix_sample = torch.mm(norm_node_embed, norm_node_embed.t())
        else:
            node_embed_full = self.projection(F.relu(orig_node_embed_full))
            norm_node_embed_full = F.normalize(node_embed_full, p=2, dim=-1)
            norm_node_embed = norm_node_embed_full[sample_idx]
            sim_matrix_sample = torch.mm(norm_node_embed, norm_node_embed_full.t())

        topk_col_idx = torch.topk(sim_matrix_sample, self.args.topk, dim=1)[1].cpu().numpy()
        filter_index = []
        for i in range(topk_col_idx.shape[1]):
            a = np.array(list(range(sim_matrix_sample.shape[0]))).reshape(-1,1)
            b = topk_col_idx[:,i].reshape(-1,1)
            c = np.hstack([a,b])
            filter_index.append(c)
        filter_index = torch.tensor(np.concatenate(filter_index, axis=0)).to(data.x.device)

        if self.args.neg_selection == 'topk':
            topk_col_idx = torch.topk(-sim_matrix_sample, self.args.neg_topk, dim=1)[1].cpu().numpy()
        if self.args.neg_selection == 'random':
            topk_col_idx = torch.randint(0, len(sample_idx), (len(sample_idx), self.args.neg_topk)).numpy()
        filter_index_neg = []
        for i in range(topk_col_idx.shape[1]):
            a = np.array(list(range(sim_matrix_sample.shape[0]))).reshape(-1,1)
            b = topk_col_idx[:,i].reshape(-1,1)
            c = np.hstack([a,b])
            filter_index_neg.append(c)
        filter_index_neg = torch.tensor(np.concatenate(filter_index_neg, axis=0)).to(data.x.device)

        pos_score_per_node = torch.zeros(sample_idx.shape[0]).to(sim_matrix_sample.device)
        pos_score_per_node = pos_score_per_node.scatter_add_(0, filter_index[:, 0], sim_matrix_sample[filter_index[:, 0], filter_index[:, 1]])
        per_node_count = torch.zeros(sample_idx.shape[0]).float().to(sim_matrix_sample.device)
        per_node_count = per_node_count.scatter_add_(0, filter_index[:, 0], torch.ones_like(filter_index[:, 0]).float().to(sim_matrix_sample.device))
        pos_part = (-2 * pos_score_per_node/per_node_count).mean() #
        neg_score_per_node = torch.zeros(sample_idx.shape[0]).to(sim_matrix_sample.device)
        neg_score_per_node = neg_score_per_node.scatter_add_(0, filter_index_neg[:, 0], sim_matrix_sample[filter_index_neg[:, 0], filter_index_neg[:, 1]]**2)
        neg_part = (neg_score_per_node).mean()

        loss = pos_part + neg_part

        loss.backward()
        optimiser.step()
        return loss.item()