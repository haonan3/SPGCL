import os
import torch
from data_loader_src.data_utils import load_splits, idx_remapping
from data_loader_src.dataset import load_nc_dataset
from torch_geometric.utils import to_undirected, remove_isolated_nodes, \
    contains_isolated_nodes, add_remaining_self_loops
import torch.nn.functional as F

from src.utils import homophily_degree

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
dataf = parent_path + '/dataset/'


class dataset_wrapper(object):
    def __init__(self, edge_index, features, label, train_idx, valid_idx, test_idx):
        self.edge_index = edge_index
        self.x = features
        self.y = label
        self.self_loop_edge_index = add_remaining_self_loops(self.edge_index)[0]
        self.train_idx_list = train_idx
        self.valid_idx_list = valid_idx
        self.test_idx_list = test_idx

    def to(self, device):
        self.self_loop_edge_index = self.self_loop_edge_index.to(device)
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


def univ_load_data(args, dataset_name, sub_dataset, train_prop=.5, valid_prop=.1, norm_feat=0, benchmark_split=True):
    dataset = load_nc_dataset(dataset_name, sub_dataset)
    orig_edge_homo_degree, orig_node_homo_degree = homophily_degree(dataset.graph['edge_index'], dataset.label)
    print('Orig Edge Homo:{}'.format(orig_edge_homo_degree))
    print('Orig Node Homo:{}'.format(orig_node_homo_degree))

    split_idx_lst = load_splits(dataset_name, sub_dataset, dataset, train_prop, valid_prop, benchmark_split)
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    print('Edge Num:{}'.format(dataset.graph['edge_index'].shape[1] / 2))
    graph_data = dataset.graph['edge_index']
    features = dataset.graph['node_feat']

    if norm_feat:
        features = F.normalize(features, p=1, dim=-1)

    train_idx = split_idx_lst['train']
    valid_idx = split_idx_lst['valid']
    test_idx = split_idx_lst['test']
    label = dataset.label

    if isinstance(train_idx, list):
        args.num_diff_init = len(train_idx)
    else:
        args.num_diff_init = 10

    if contains_isolated_nodes(graph_data):
        print("Contrain isolated nodes. Removing..")
        graph_data, edge_attr, mask = remove_isolated_nodes(graph_data, num_nodes=features.shape[0])
        features = features[mask]
        label = label[mask]
        idx_map_dict = {}
        for idx, i in enumerate(mask.tolist()):
            if i:
                idx_map_dict[idx] = len(idx_map_dict)

        if dataset_name == 'WikiCS':
            for i in range(20):
                train_idx[i] = idx_remapping(train_idx[i][mask[train_idx[i]]], idx_map_dict)
                valid_idx[i] = idx_remapping(valid_idx[i][mask[valid_idx[i]]], idx_map_dict)
            test_idx[0] = idx_remapping(test_idx[0][mask[test_idx[0]]], idx_map_dict)
        elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            train_idx = idx_remapping(train_idx[mask[train_idx]], idx_map_dict)
            valid_idx = idx_remapping(valid_idx[mask[valid_idx]], idx_map_dict)
            test_idx = idx_remapping(test_idx[mask[test_idx]], idx_map_dict)
        else:
            for i in range(10):
                train_idx[i] = idx_remapping(train_idx[i][mask[train_idx[i]]], idx_map_dict)
                valid_idx[i] = idx_remapping(valid_idx[i][mask[valid_idx[i]]], idx_map_dict)
                test_idx[i] = idx_remapping(test_idx[i][mask[test_idx[i]]], idx_map_dict)


    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        print("# Node:{}, # Train:{}, # Valid:{}, # Test:{}".format(features.shape[0],
                                                            train_idx.shape[0]/features.shape[0],
                                                            valid_idx.shape[0]/features.shape[0],
                                                            test_idx.shape[0]/features.shape[0]))
    else:
        print("# Node:{}, # Train:{}, # Valid:{}, # Test:{}".format(features.shape[0],
                                                                    train_idx[0].shape[0] / features.shape[0],
                                                                    valid_idx[0].shape[0] / features.shape[0],
                                                                    test_idx[0].shape[0] / features.shape[0]))

    assert label.shape[0] == features.shape[0]
    assert label.shape[0] == graph_data.max().item() + 1

    return graph_data, features, label, train_idx, valid_idx, test_idx