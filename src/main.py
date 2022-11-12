import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import k_hop_subgraph, contains_isolated_nodes, coalesce, to_undirected, is_undirected
from tqdm import tqdm
import networkx as nx
from data_loader_src.data_utils import rand_train_test_idx
from data_loader_src.universal_dataloader import univ_load_data, dataset_wrapper
from src.logreg import LogReg
from src.Non_Homophily_GCL import SPGCL
from src.utils import setup_logger, show_gpu, load_params
from torch.utils.tensorboard import SummaryWriter

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_embedding(args, embeds, data, test_epochs=1500, linear_lr=0.0005):
    ft_in = embeds.shape[1]
    xent = nn.CrossEntropyLoss()
    acc_list, acc_last_list = [], []
    auc_roc_list, auc_roc_last_list = [], []
    best_acc_list, best_roc_list = [], []
    for run_id in range(args.num_diff_init):
        if args.benchmark_split:
            if args.dataset not in ['Cora', 'CiteSeer', 'PubMed']:
                data.train_idx = data.train_idx_list[run_id].to(data.x.device)
                data.valid_idx = data.valid_idx_list[run_id].to(data.x.device)
                data.test_idx = data.test_idx_list[0].to(data.x.device) if args.dataset == 'WikiCS' \
                    else data.test_idx_list[run_id].to(data.x.device)
            else:
                data.train_idx = data.train_idx_list.to(data.x.device)
                data.valid_idx = data.valid_idx_list.to(data.x.device)
                data.test_idx = data.test_idx_list.to(data.x.device)
        else:
            train_idx, valid_idx, test_idx = rand_train_test_idx(data.y, train_prop=args.train_rate,
                                                                 valid_prop=args.val_rate, ignore_negative=True)
            data.train_idx = train_idx.to(data.x.device)
            data.valid_idx = valid_idx.to(data.x.device)
            data.test_idx = test_idx.to(data.x.device)

        train_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        train_mask[data.train_idx] = True
        data.train_mask = train_mask.to(data.x.device)
        val_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        val_mask[data.valid_idx] = True
        data.val_mask = val_mask.to(data.x.device)
        test_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        test_mask[data.test_idx] = True
        data.test_mask = test_mask.to(data.x.device)

        train_embs = embeds[data.train_mask]
        val_embs = embeds[data.val_mask]
        test_embs = embeds[data.test_mask]
        assert len(data.y.shape) in [1, 2]
        train_lbls = torch.argmax(data.y[data.train_mask], dim=1) if len(data.y.shape) == 2 else data.y[data.train_mask]
        val_lbls = torch.argmax(data.y[data.val_mask], dim=1) if len(data.y.shape) == 2 else data.y[data.val_mask]
        test_lbls = torch.argmax(data.y[data.test_mask], dim=1) if len(data.y.shape) == 2 else data.y[data.test_mask]

        log = LogReg(ft_in, args.hidden, args.num_classes).to(args.device)
        opt = torch.optim.Adam(log.parameters(), lr=linear_lr, weight_decay=1e-5)

        best_val_acc, best_val_roc_auc, best_test_acc, best_test_roc_auc = 0, 0, 0, 0
        _best_test_acc, _best_test_roc_auc = 0, 0
        for log_epoch in tqdm(range(test_epochs)):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
            if (log_epoch + 1) % args.test_interval == 0:
                with torch.no_grad():
                    log.eval()
                    # for valid set
                    val_logits = log(val_embs)
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = torch.sum(val_preds == val_lbls).float() / val_lbls.shape[0]
                    val_roc_auc = roc_auc_score(F.one_hot(val_lbls, num_classes=args.num_classes).cpu(),
                                            torch.softmax(val_logits, dim=1).detach().cpu())
                    # for test set
                    test_logits = log(test_embs)
                    test_preds = torch.argmax(test_logits, dim=1)
                    test_acc = torch.sum(test_preds == test_lbls).float() / test_lbls.shape[0]
                    test_roc_auc = roc_auc_score(F.one_hot(test_lbls, num_classes=args.num_classes).cpu(),
                                                 torch.softmax(test_logits, dim=1).detach().cpu())

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                    if val_roc_auc > best_val_roc_auc:
                        best_val_roc_auc = val_roc_auc
                        best_test_roc_auc = test_roc_auc
                    if test_acc > _best_test_acc:
                        _best_test_acc = test_acc
                    if test_roc_auc > _best_test_roc_auc:
                        _best_test_roc_auc = test_roc_auc
                    test_acc_last = test_acc
                    test_roc_auc_last = test_roc_auc

        acc_last_list.append(test_acc_last)
        auc_roc_last_list.append(test_roc_auc_last)
        acc_list.append(best_test_acc)
        auc_roc_list.append(best_test_roc_auc)

        best_acc_list.append(_best_test_acc)
        best_roc_list.append(_best_test_roc_auc)

    accs_last = torch.stack(acc_last_list)
    AUCROCs_last = np.array(auc_roc_last_list)

    accs = torch.stack(acc_list)
    AUCROCs = np.array(auc_roc_list)

    best_accs = torch.stack(best_acc_list)
    best_AUCROCs = np.array(best_roc_list)

    return accs_last.mean().item(), accs_last.std().item(), AUCROCs_last.mean(),  AUCROCs_last.std(), \
           accs.mean().item(), accs.std().item(), AUCROCs.mean(), AUCROCs.std(), \
           best_accs.mean().item(), best_accs.std().item(), best_AUCROCs.mean(), best_AUCROCs.std()


def ssl_train(args, epoch, data, model, optimiser, cnt_wait, best):
    model.train()
    model.current_epoch = epoch
    loss = model.model_train(data, optimiser)

    if cnt_wait > args.patience and args.early_stop:
        print('Early stopping!')
        loss = None
    else:
        if loss < best:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1
        print('Epoch{}, Train Loss:{}, CNT:{}'.format(epoch, loss, cnt_wait))

    return loss, cnt_wait, best


def main(args):
    edge_index, features, label, train_idx, valid_idx, test_idx = univ_load_data(args, args.dataset, args.sub_dataset,
                                                        train_prop=args.train_rate, valid_prop=args.val_rate,
                                                        norm_feat=args.norm_feat, benchmark_split=args.benchmark_split)
    data = dataset_wrapper(edge_index, features, label, train_idx, valid_idx, test_idx)
    num_classes = len(data.y.unique())
    args.num_classes = num_classes if (args.dataset != 'fb100') else 2
    args.C = args.num_classes if (args.dataset != 'fb100') else 2
    args.num_features = data.x.shape[1]
    args.num_nodes = data.x.shape[0]

    t1 = time.time()
    print("initing model..")
    assert not contains_isolated_nodes(data.edge_index)
    subg_file_path = parent_path + '/saved_models/{}_hop_{}_subg.pkl'.format(args.dataset, args.subg_num_hops)
    print(subg_file_path)

    if os.path.exists(subg_file_path):
        print('Loading saved subg...')
        with open(subg_file_path, 'rb') as file:
            subgraph_cache = pickle.load(file)
        mask_matrix = []
        subgraph_list_len = len(subgraph_cache)
        for node_idx in tqdm(range(subgraph_list_len), total=subgraph_list_len):
            result = subgraph_cache[node_idx]
            l = (torch.zeros_like(result) + node_idx).unsqueeze(0)
            r = result.unsqueeze(0)
            mask_matrix.append(torch.cat([l,r], dim=0))
        mask_matrix = torch.cat(mask_matrix, dim=1)
        if not is_undirected(mask_matrix):
            print(args.dataset + 'need check!!!!')
            mask_matrix = to_undirected(mask_matrix)
        mask_matrix = coalesce(mask_matrix)
    else:
        print("construct k-hop-subg")
        subgraph_cache = []
        mask_matrix = []
        assert data.edge_index.max() + 1 == data.x.shape[0]
        for node_idx in tqdm(set(data.edge_index[0].tolist()), total=len(set(data.edge_index[0].tolist()))):
            result = k_hop_subgraph(node_idx=node_idx, num_hops=args.subg_num_hops, edge_index=data.edge_index)[0]
            if len(result) == 0: # if no neighbor, skip
                continue
            subgraph_cache.append(result)
            l = (torch.zeros_like(result) + node_idx).unsqueeze(0)
            r = result.unsqueeze(0)
            mask_matrix.append(torch.cat([l,r], dim=0))
        mask_matrix = torch.cat(mask_matrix, dim=1)
        with open(subg_file_path, 'wb') as file:
            pickle.dump(subgraph_cache, file)

    data.subgraph_cache = subgraph_cache
    data.mask_matrix = mask_matrix
    t2 = time.time()
    print("Pre-compute subgraph: {}s".format(t2-t1))

    # 2.create model
    model = SPGCL(args)

    model.logger = logger
    model.MAX_USED_MEMO = 0

    if args.optimizer == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimiser = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("optimizer error.")

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.to(args.device)
        data = data.to(args.device)

    cnt_wait = 0
    best = 1e9

    for epoch in range(args.epochs):

        epoch_loss, cnt_wait, best = ssl_train(args, epoch, data, model, optimiser, cnt_wait, best)

        if epoch_loss is None:
            break

    model.eval()
    with torch.no_grad():
        embeds = model.embed(data)
    accs_last_mean, accs_last_std,  AUCROC_last_mean, AUCROC_last_std, \
    accs_mean, accs_std,  AUCROC_mean, AUCROC_std, \
    b_accs_mean, b_accs_std, b_AUCROC_mean, b_AUCROC_std = test_embedding(args, embeds, data, test_epochs=args.linear_epochs)
    info_str = "[Test] Acc Mean:{} Acc Std:{} AUCROC Mean:{} AUCROC Std:{},\n" \
   "[Last] Acc Mean:{} Acc Std:{} AUCROC Mean:{} AUCROC Std:{}.\n" \
               "[Best] BAcc Mean:{} BAcc Std:{} BAUCROC Mean:{} BAUCROC Std:{}.".format(accs_mean, accs_std, AUCROC_mean, AUCROC_std,
                                    accs_last_mean, accs_last_std,  AUCROC_last_mean, AUCROC_last_std,
                                                                                        b_accs_mean, b_accs_std,
                                                                                        b_AUCROC_mean, b_AUCROC_std)
    print(info_str)
    if 'txt' in args.log_type:
        logger.info(info_str)
    return model


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subg_num_hops', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='actor',
                        help='heter-graph: : chameleon, squirrel, actor twitch-e, genius, twitch-gamer,'
                             '+ homo-graph: ; Cora, CiteSeer, PubMed; WikiCS; Computers, Photo; cs, physics')
    parser.add_argument('--sub_dataset', type=str, default=None)
    # model args
    parser.add_argument('--norm_feat', type=int, default=1)
    parser.add_argument('--no_bn', type=int, default=0, help='do not use batchnorm')
    parser.add_argument('--encoder_layer_num', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed_num', type=int, default=128)
    parser.add_argument('--max_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    parser.add_argument('--activation', type=str, default='prelu', help='prelu, relu, rrelu')
    parser.add_argument('--proj_activation', type=str, default='prelu', help='prelu, relu, rrelu')
    parser.add_argument('--square_subg', type=int, default=1)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--neg_topk', type=int, default=100)
    parser.add_argument('--num_proj_layer', type=int, default=2)
    parser.add_argument('--benchmark_split', type=int, default=1)
    parser.add_argument('--neg_selection', type=str, default='random', help='topk, random')
    parser.add_argument('--seed_sampling', type=str, default='tree', help='random, tree')
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--linear_lr', type=float, default=0.0005)
    parser.add_argument('--linear_epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--early_stop', type=int, default=1)
    parser.add_argument('--only_preprocess', type=int, default=0)
    parser.add_argument('--log_type', type=str, default='')
    parser.add_argument('--log_note', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='ICLR_logs')
    parser.add_argument('--on_the_fly', type=int, default=0, choices=[0,1])
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--track_gpu_memo', type=int, default=0)
    parser.add_argument('--load_params', type=int, default=1)
    # patch args for the params re-loading
    parser.add_argument('--reset_max_size', type=int, default=None)
    parser.add_argument('--reset_epochs', type=int, default=None)
    parser.add_argument('--reset_topK', type=int, default=None)
    parser.add_argument('--reset_lr', type=float, default=None)
    parser.add_argument('--reset_seed_num', type=int, default=None)
    parser.add_argument('--reset_weight_decay', type=float, default=None)
    parser.add_argument('--reset_subg_num_hops', type=int, default=None)
    parser.add_argument('--reset_hidden', type=int, default=None)

    args = parser.parse_args()
    # 1.process args
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset in ['Computers', 'Photo', 'cs', 'physics']:
        args.benchmark_split = 0
    args.train_rate = 0.5 if args.benchmark_split else 0.1
    args.val_rate = 0.25 if args.benchmark_split else 0.1
    if args.load_params:
        args = load_params(args)
    return args


if __name__ == '__main__':
    current_time = str(time.time()).replace('.', '')
    args = args_parser()
    set_seed(args.seed)

    args.early_stop = 0

    if args.reset_hidden is not None:
        args.hidden = args.reset_hidden
    if args.reset_topK is not None:
        args.topk = args.reset_topK
    if args.reset_lr is not None:
        args.lr = args.reset_lr
    if args.reset_max_size is not None:
        args.max_size = args.reset_max_size
    if args.reset_epochs is not None:
        args.epochs = args.reset_epochs
    if args.reset_seed_num is not None:
        args.seed_num = args.reset_seed_num
    if args.reset_weight_decay is not None:
        args.weight_decay = args.reset_weight_decay
    if args.reset_subg_num_hops is not None:
        args.subg_num_hops = args.reset_subg_num_hops

    # 2.init logger
    logger, homo_logger = None, None
    title = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
       args.log_note, args.dataset,  args.norm_feat, args.no_bn,
       args.activation, args.proj_activation, args.dropout, args.encoder_layer_num,
       args.num_proj_layer, args.hidden, args.optimizer, args.lr,
       args.weight_decay, args.linear_lr, args.subg_num_hops, args.max_size, args.epochs,
       args.topk, args.neg_topk, args.linear_epochs, args.seed_sampling, args.seed_num)

    save_folder = args.save_folder
    if 'txt' in args.log_type:
        logger = setup_logger('logger', '{}/results/{}/{}.txt'.format(parent_path, save_folder, title))
    if 'tb' in args.log_type:
        writer = SummaryWriter(comment=title)

    model = main(args)