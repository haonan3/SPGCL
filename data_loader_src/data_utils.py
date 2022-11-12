import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data

from torch_sparse import SparseTensor
# from google_drive_downloader import GoogleDriveDownloader as gdd


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
DATAPATH = parent_path + '/dataset/non_homophilous_benchmark_data/'

dataset_drive_url = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',  # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',  # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'  # Wiki 1.9M
}


splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}


def nc_dataset_to_torch_geo(edge_index, features, label, train_idx, valid_idx, test_idx):
    tg_data = Data()
    tg_data.x = features
    tg_data.edge_index = edge_index
    tg_data.edge_attr = None
    tg_data.y = label

    tg_data.train_idx = train_idx
    tg_data.valid_idx = valid_idx
    tg_data.test_idx = test_idx
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True
    tg_data.train_mask = train_mask
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[valid_idx] = True
    tg_data.val_mask = val_mask
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True
    tg_data.test_mask = test_mask

    tg_data.node_ids = torch.arange(features.shape[0],)
    return tg_data


def idx_remapping(idx_tensor, idx_map_dict):
    mapped_idx_list = []
    for i in idx_tensor.tolist():
        mapped_idx_list.append(idx_map_dict[i])
    return torch.tensor(mapped_idx_list).long()


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:

        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out



def load_splits(dataset_name, sub_dataset, dataset, train_prop, valid_prop, benchmark_split):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'actor']:
        # splits_lst = []
        split_dict = {}
        split_dict['train'] = dataset.train_idx
        split_dict['valid'] = dataset.valid_idx
        split_dict['test'] = dataset.test_idx
    else:
        name = dataset_name
        if sub_dataset:
            name += f'-{sub_dataset}'

        if benchmark_split and os.path.exists(f'{DATAPATH}splits/{name}-splits.npy'):
            splits_lst = np.load(f'{DATAPATH}splits/{name}-splits.npy', allow_pickle=True)
            for i in range(len(splits_lst)):
                for key in splits_lst[i]:
                    if not torch.is_tensor(splits_lst[i][key]):
                        splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])

        elif benchmark_split and dataset_name in splits_drive_url.keys():
            gdd.download_file_from_google_drive(
                file_id=splits_drive_url[dataset_name], \
                dest_path=f'{DATAPATH}splits/{name}-splits.npy', showsize=True)

            splits_lst = np.load(f'{DATAPATH}splits/{name}-splits.npy', allow_pickle=True)
            for i in range(len(splits_lst)):
                for key in splits_lst[i]:
                    if not torch.is_tensor(splits_lst[i][key]):
                        splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        else:
            splits_lst = [dataset.get_idx_split(train_prop=train_prop, valid_prop=valid_prop) for _ in range(10)]

        num_splits = len(splits_lst)
        splits_lst_reorganized = defaultdict(list)
        for part_name in ['train', 'valid', 'test']:
            for i in range(num_splits):
                splits_lst_reorganized[part_name].append(splits_lst[i][part_name])
        split_dict = splits_lst_reorganized
    return split_dict




