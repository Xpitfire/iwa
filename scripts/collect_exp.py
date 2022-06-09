import sys
import os
from typing import List
import numpy as np
import argparse
from argparse import ArgumentParser
from approaches.ensemble_trainer import EnsembleTrainer


HHAR_SA = ['0_src-6_tgt', '1_src-6_tgt', '2_src-7_tgt', '3_src-8_tgt', '4_src-5_tgt']
HAR = ['2_src-11_tgt', '6_src-23_tgt', '7_src-13_tgt', '9_src-18_tgt', '12_src-16_tgt']
EEG = ['0_src-11_tgt', '7_src-18_tgt', '9_src-14_tgt', '12_src-5_tgt', '16_src-1_tgt']
WISDM = ['6_src-19_tgt', '7_src-18_tgt', '18_src-23_tgt', '20_src-30_tgt', '35_src-31_tgt']
MOONS = ['moons_src-moons_tgt']
AMAZON_REVIEWS = ['books-dvd',
        'books-electronics',
        'books-kitchen',
        'dvd-books',
        'dvd-electronics',
        'dvd-kitchen',
        'electronics-books',
        'electronics-dvd',
        'electronics-kitchen',
        'kitchen-books',
        'kitchen-dvd',
        'kitchen-electronics']
MINI_DOMAIN_NET = ['painting-real-sketch-clipart-infograph-quickdraw',
           'painting-quickdraw-sketch-clipart-infograph-real',
           'painting-quickdraw-real-clipart-infograph-sketch',
           'painting-quickdraw-real-sketch-infograph-clipart',
           'painting-quickdraw-real-sketch-clipart-infograph',
           'quickdraw-real-sketch-clipart-infograph-painting']


METHOD_ADATIME = ['DDC', 'DANN', 'Deep_Coral', 'DSAN', 'MMDA', 'AdvSKM', 'HoMM', 'DIRT', 'CDAN', 'CoDATS']
METHOD_BP = ['dann', 'cmd', 'mmd']


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser()
    parser.add_argument('--base_dir',
                        help='base directory path to results',
                        type=str)
    parser.add_argument('--method',
                        help='domain adaptation method [ADATIME, BP]',
                        type=str)
    parser.add_argument('--dataset',
                        help='domain adaptation dataset names [HHAR_SA, EEG, HAR, WISDM, MOONS, AMAZON_REVIEWS, MINI_DOMAIN_NET]',
                        type=str)
    parser.add_argument('--out_dir',
                        help='target directory for final parsed results',
                        type=str)
    parser.add_argument('--rcond',
                        help='rcond parameter used in Aggregation for np.linalg.pinv()',
                        type=float,
                        default=1e-2)
    parser.add_argument('--skip_exists',
                        action='store_true')
    parser.add_argument('--seed_list',
                        help='determine seed list to be used',
                        type=str,
                        default='')
    return parser.parse_args()


def check_dirs(base_dir):
    """Creates the basic temporary folder structure for the projects"""
    assert os.path.exists(base_dir)


def load_seed_list(file_name='seed_list.txt'):
    """Loads seeds from a file"""
    with open(file_name, 'r') as f:
        seeds = f.read().split('\n')
    seeds = [int(s) for s in seeds if s is not None and s != '']
    return seeds


def rec_collect_folders(base_dir: str):
    dirs = []
    for d in os.walk(base_dir):
        dirs.append(d)
    return dirs


def extract_results(dirs, da_method, dataset):
    cls_preds = {}
    iwv_preds = {}
    for d in dirs:
        if da_method in d[0] and f'cls_pred_dataset_{dataset}.npz' in d[2]:
            try:
                seed = d[0].split('seed=')[1].split('/')[0] # version 1
            except:
                print(d)
                seed = d[0].split('seed')[1].split('-')[0] # alternative version
            for f in d[2]:
                if 'cls_pred_' in f:
                    if seed not in cls_preds:
                        cls_preds[seed] = np.load(os.path.join(d[0], f), allow_pickle=True)['arr_0'].item()
                    else:
                        tmp = np.load(os.path.join(d[0], f), allow_pickle=True)['arr_0'].item()
                        for k, v in tmp.items():
                            cls_preds[seed][k] = v
                elif 'iwv_pred_' in f:
                    if seed not in iwv_preds:
                        iwv_preds[seed] = np.load(os.path.join(d[0], f), allow_pickle=True)['arr_0'].item()
                    else:
                        tmp = np.load(os.path.join(d[0], f), allow_pickle=True)['arr_0'].item()
                        k = list(tmp.keys())[0]
                        iwv_preds[seed][k] = tmp[k]
    if len(list(cls_preds.keys())) == 0:
        return None, None, None
    if len(list(cls_preds.keys())) > 1:
        id = list(cls_preds.keys())[0]
        keys = list(cls_preds[id].keys())
        tmp_cls_preds = cls_preds[id]
        del cls_preds[id]
        for k in keys:
            for vv in cls_preds.values():
                v_ = vv[k]
                k_ = list(v_.keys())[0]
                tmp_cls_preds[k][k_] = v_[k_]
        id = list(iwv_preds.keys())[0]
        keys = list(iwv_preds[id].keys())
        seeds = set((id))
        tmp_iwv_preds = iwv_preds[id]
        del iwv_preds[id]
        for k in keys:
            for vv in iwv_preds.values():
                v_ = vv[k]
                k_ = list(v_.keys())[0]
                seeds.add(k_)
                tmp_iwv_preds[k][k_] = v_[k_]
        return tmp_cls_preds, tmp_iwv_preds, seeds
    else:
        id = list(cls_preds.keys())[0]
        cls_preds = cls_preds[id]
        iwv_preds = iwv_preds[list(iwv_preds.keys())[0]]
        return cls_preds, iwv_preds, (id)


def run():
    """Main entrance point for extracting results.
    """
    options = parse_args()

    method = options.method
    if method == 'ADATIME':
        da_methods = METHOD_ADATIME
    elif method == 'BP':
        da_methods = METHOD_BP
    else:
        raise ValueError(f'Unknown method: {method}')    
    
    ds = options.dataset
    if ds == 'HHAR_SA':
        ds_names = HHAR_SA
        extractor = 'extractor/adatime_hhar_sa_results_loader.py'
    elif ds == 'HAR':
        ds_names = HAR
        extractor = 'extractor/adatime_har_results_loader.py'
    elif ds == 'EEG':
        ds_names = EEG
        extractor = 'extractor/adatime_eeg_results_loader.py'
    elif ds == 'WISDM':
        ds_names = WISDM
        extractor = 'extractor/adatime_wisdm_results_loader.py'
    elif ds == 'MOONS':
        ds_names = MOONS
        extractor = 'extractor/twinmoons_results_loader.py'
    elif ds == 'AMAZON_REVIEWS':
        ds_names = AMAZON_REVIEWS
        extractor = 'extractor/amazon_results_loader.py'
    elif ds == 'MINI_DOMAIN_NET':
        ds_names = MINI_DOMAIN_NET
        extractor = 'extractor/minidomainnet_results_loader.py'
    else:
        raise ValueError(f'Unknown dataset: {ds}')
    
    skip_collection = False
    if options.skip_exists and os.path.exists(os.path.join(options.out_dir, ds)):
        skip_collection = True

    seed_list = options.seed_list.split(',')
    if not skip_collection:
        check_dirs(options.base_dir)
        # collect all directories
        dirs = rec_collect_folders(options.base_dir)
        for method in da_methods:
            target_dir = os.path.join(options.out_dir, ds)
            for ds_name in ds_names:
                # extract the defined results
                cls_preds, iwv_preds, seeds = extract_results(dirs, method, ds_name)
                if seeds is not None:
                    seed_list = seeds
                if cls_preds is None: 
                    print(f'WARNING: No results found for {method} > {ds_name}')
                    continue
                # save all extracted results
                out_dir = os.path.join(target_dir, method)
                os.makedirs(out_dir, exist_ok=True)
                pred_file = os.path.join(out_dir, f'cls_pred_dataset_{ds_name}.npz')
                np.savez(pred_file, cls_preds)
                pred_file = os.path.join(out_dir, f'iwv_pred_dataset_{ds_name}.npz')
                np.savez(pred_file, iwv_preds)
    
    # post-process the results
    for method in da_methods:
        out_dir = os.path.join(options.out_dir, ds, method)
        ensemble_trainer = EnsembleTrainer(out_dir, method, extractor, seed_list, options.rcond) #! use seeds from last entries
        ensemble_trainer.run()


if __name__ == "__main__":
    run()
