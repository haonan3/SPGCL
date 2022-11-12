import argparse
import os
import pickle
import torch
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitch-gamer')
    parser.add_argument('--subg_num_hops', type=int, default=3)
    args = parser.parse_args()
    return args






if __name__ == '__main__':
    args = args_parser()
    print(parent_path)

    subg_save_dir = parent_path + '/saved_models/{}/'.format(args.dataset)
    file_list = os.listdir(subg_save_dir)
    data_dict = {}
    for file_name in file_list:
        if 'pkl' in file_name:
            file_id = int(file_name.split('.')[0].split('_')[-1])
            with open(subg_save_dir+file_name, 'rb') as f:
                data_dict[file_id] = pickle.load(f)
    num_file = len(data_dict)
    data_cache = None
    for i in tqdm(range(num_file)):
        data_cache = data_dict[i] if data_cache is None else data_cache + data_dict[i]
    save_path = parent_path + '/saved_models/{}_hop_{}_subg.pkl'.format(args.dataset, args.subg_num_hops)
    print('Start Saving...')
    with open(save_path, 'wb') as file:
        pickle.dump(data_cache, file)