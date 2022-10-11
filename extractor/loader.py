import numpy as np
from typing import Any, Dict, List
from pathlib import Path


class Loader:
    da_methods: List[str] = ['cmd', 'mmd', 'dann'] # domain adaptation methods
    file_prefixes: List[str] = ['cls_pred_dataset_', 'iwv_pred_dataset_']
    cls_train_results_prefix: str = 'cls_train_results_'

    def __init__(self, base_dir: str, da_method: str, prefix: str, suffix: str = '',
                 file_suffix: str = '.npz'):

        # Directory handling
        self.base_dir = base_dir
        self.prefix = prefix
        self.suffix = suffix

        # File handling
        self.file_suffix = file_suffix
        self.method_results_dir = Path(base_dir)

        # processed results
        self.processed_results_dir = Path(base_dir) / 'processed_results'

    def get_class_source_target_preds_labels(self, domain: str, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
        # levels in cls_pred
        # -> keys: lambdas (with domain as prefix f'{domain}-{lambda})
        # -> keys: seeds (as str, i.e. '1234')
        # -> [list with 1 entry] # if multiple runs with one seed this has multiple entries
        # -> keys: ['s_preds', 't_preds', 's_lbls', 't_lbls', 's_da_preds', 't_da_preds']
        # -> [list with 25 entries] # batches < stack these
        # -> arrays with different shapes (128,5)
        # Remark#1: X_da_preds are the activations in the representation layer (this is the layer before the classifier) for cmd and mmd (for domain X (source/target))
        # Remark#2: X_da_preds is the domain label for dann (for domain X (source/target))

        def get_lambdas_from_keys(keys: List[str]) -> List[str]:
            lambdas = []
            for k in keys:
                k_split = k.split('-')
                if k_split[-2] == '1e':
                    lambdas.append(k_split[-2] + '-' + k_split[-1])
                else:
                    lambdas.append(k_split[-1])
            return lambdas

        fileprefix = 'cls_pred_dataset_'
        cls_predictions = np.load(self.method_results_dir /
                                  f'{fileprefix}{domain}{self.file_suffix}', allow_pickle=True)
        cls_predictions = cls_predictions['arr_0'].item()

        lambdas = get_lambdas_from_keys(list(cls_predictions.keys()))

        lamb_s_t_pred_labels_dict = {}

        for lamb in lambdas:
            key = f'{domain}-{lamb}'
            s_t_preds_labels = cls_predictions[key][str(seed)][0]

            # stack and concatenate the list of arrays
            for k, v in s_t_preds_labels.items():
                if 'preds' in k:
                    s_t_preds_labels[k] = np.vstack(v)
                elif 'lbls' in k:
                    if not isinstance(v, list) and len(v.shape) == 1:
                        s_t_preds_labels[k] = v
                    else:
                        s_t_preds_labels[k] = np.concatenate(v)
                else:
                    raise ValueError(f'Unknown key: {k}')

            lamb_s_t_pred_labels_dict[lamb] = s_t_preds_labels
        # return Dict[lambda]['s_preds', 't_preds', 's_lbls', 't_lbls', 's_da_preds', 't_da_preds'] = np.array.shape(n_data, n_class)
        return lamb_s_t_pred_labels_dict

    def get_iwv_predictions(self, domain: str, seed: int) -> Dict[str, np.ndarray]:
        # file
        # -> keys: dataset (one key)
        # -> keys: seeds (as str, i.e. '1234')
        # -> [list with 1 entry]
        # -> keys: ['s_preds', 't_preds', 's_lbls', 't_lbls']
        # -> [list with 25 entries] #? what are the entries?
        # s_lbls and t_lbls arrays with shape (128,)
        # s_preds and t_preds arrays with shape (128, 256) ??
        fileprefix = 'iwv_pred_dataset_'
        iwv_predictions = np.load(self.method_results_dir /
                                  f'{fileprefix}{domain}{self.file_suffix}', allow_pickle=True)

        iwv_preds = iwv_predictions['arr_0'].item()[domain][str(seed)][0]

        # stack and concatenate the list of arrays
        for k, v in iwv_preds.items():
            if 'preds' in k:
                iwv_preds[k] = np.vstack(v)
            elif 'lbls' in k:
                iwv_preds[k] = np.concatenate(v)
            else:
                raise ValueError(f'Unknown key: {k}')

        return iwv_preds

    def get_cls_trainlogs(domain: str) -> Any:
        #! probably not so important
        # levels in cls_results
        # -> keys: lambdas (with domain as prefix f'{domain}-{lambda})
        # -> keys: seeds (as str, i.e. '1234')
        # -> [list with 1 entry]
        # -> keys: ['train_losses', 'eval_losses', 'train_accs', 'eval_accs', 'train_values', 'eval_values']
        raise NotImplementedError()
