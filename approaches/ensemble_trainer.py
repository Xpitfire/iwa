from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from approaches.aggregation import Aggregator
from approaches.dev_iwv import DeepEmbeddedValidation, ImportanceWeightedValidation
from approaches.model_selector_base import TargetMajorityVotePrediction
from approaches.multi_regression import SourceLinearRegression, TargetConfidenceLinearRegression, TargetMajorityVoteLinearRegression

from misc.helpers import acc, load_function


class EnsembleTrainer():

    def __init__(self, base_dir: str, da_method: str, extractor: str,
                 seed_list: List[int], rcond: float, suffix: str = '', 
                 manual_filter_lambdas: List[str] = []):

        self.ExtractorClass = load_function(extractor, 'ResultsLoader')
        self.extractor = self.ExtractorClass(base_dir, da_method, suffix)

        self.seeds = seed_list
        self.da_method = da_method
        self.rcond = rcond
        self.manual_filter_lambdas = manual_filter_lambdas

        self.aggregator_dict = {}
        #* Output of the algorithm
        self.src_acc_dict = {}
        self.tgt_acc_dict = {}  # Structure: seed | domain | lambda / method -> np.ndarray
        # stores the weights for the models in the ensemble
        self.ensemble_weights_dict = {}  # Structure: seed | domain | lambda / method -> np.ndarray

    def run(self):
        # seeds_ = tqdm(self.seeds)
        for seed in self.seeds: #seeds_:
            # seeds_.set_description(f'Seed {seed}')
            self.src_acc_dict[seed] = {}
            self.tgt_acc_dict[seed] = {}
            self.aggregator_dict[seed] = {}
            self.ensemble_weights_dict[seed] = {}

            # domains_ = tqdm(self.ExtractorClass.domains)
            for domain in self.ExtractorClass.domains: #domains_:
                # domains_.set_description(f'Domain {domain}')
                print(f'Seed: {seed} --- Domain: {domain}')
                cls_dict = self.extractor.get_class_source_target_preds_labels(
                    domain=domain, seed=seed)
                iwv_dict = self.extractor.get_iwv_predictions(domain=domain, seed=seed)

                # dictionary for predictions of model selection methods
                model_sel_methods_preds = {}  # key: model_selector.key_name, value: Tuple[source_preds, target_preds]
                ensemble_weights = {}  # key: model_selector.key_name, value: np.ndarray[ensemble weight]

                #! Aggregation method
                aggregator = Aggregator(rcond=self.rcond, manual_filter_lambdas=self.manual_filter_lambdas)
                aggregator.domains_name = domain
                aggregator.da_method = self.da_method
                model_sel_methods_preds[aggregator.key_name()] = aggregator.predict(cls_dict, iwv_dict)
                ensemble_weights[aggregator.key_name()] = aggregator.ensemble_weights
                # save aggregation results:
                self.aggregator_dict[seed][domain] = aggregator

                #! Baseline 1: Regression with source only data
                sourcereg = SourceLinearRegression(self.manual_filter_lambdas)
                sourcereg.domains_name = domain
                sourcereg.da_method = self.da_method

                model_sel_methods_preds[sourcereg.key_name()] = sourcereg.predict(cls_dict, iwv_dict)
                ensemble_weights[sourcereg.key_name()] = sourcereg.ensemble_weights

                #! Baseline 2: Regression with pseudo target labels based on target majority vote
                targetmajreg = TargetMajorityVoteLinearRegression(self.manual_filter_lambdas)
                targetmajreg.domains_name = domain
                targetmajreg.da_method = self.da_method

                model_sel_methods_preds[targetmajreg.key_name()] = targetmajreg.predict(cls_dict, iwv_dict)
                ensemble_weights[targetmajreg.key_name()] = targetmajreg.ensemble_weights

                #! Baseline 3: Regression with pseudo target labels based on confidence scores
                targetconfreg = TargetConfidenceLinearRegression(self.manual_filter_lambdas)
                targetconfreg.domains_name = domain
                targetconfreg.da_method = self.da_method

                model_sel_methods_preds[targetconfreg.key_name()] = targetconfreg.predict(cls_dict, iwv_dict)
                ensemble_weights[targetconfreg.key_name()] = targetconfreg.ensemble_weights

                #! Baseline 4: Deep Embedded Validation
                dev = DeepEmbeddedValidation(self.manual_filter_lambdas)
                dev.domains_name = domain
                dev.da_method = self.da_method

                model_sel_methods_preds[dev.key_name()] = dev.predict(cls_dict, iwv_dict)
                ensemble_weights[dev.key_name()] = dev.ensemble_weights

                #! Baseline 5: Deep Embedded Validation
                iwv = ImportanceWeightedValidation(self.manual_filter_lambdas)
                iwv.domains_name = domain
                iwv.da_method = self.da_method

                model_sel_methods_preds[iwv.key_name()] = iwv.predict(cls_dict, iwv_dict)
                ensemble_weights[iwv.key_name()] = iwv.ensemble_weights

                #! Baseline 6: Majority vote prediction
                targetmajvote = TargetMajorityVotePrediction(self.manual_filter_lambdas)
                targetmajvote.domains_name = domain
                targetmajvote.da_method = self.da_method

                model_sel_methods_preds[targetmajvote.key_name()] = targetmajvote.predict(cls_dict, iwv_dict)
                ensemble_weights[targetmajvote.key_name()] = targetmajvote.ensemble_weights

                src_acc, tgt_acc = compute_accuracies(
                    cls_dict, model_sel_methods_preds, aggregator._src_test_idxs, aggregator._tgt_test_idxs, self.manual_filter_lambdas)
                self.src_acc_dict[seed][domain] = src_acc
                self.tgt_acc_dict[seed][domain] = tgt_acc
                self.ensemble_weights_dict[seed][domain] = ensemble_weights

        # save aggregation results
        save_dir = self.extractor.processed_results_dir
        save_dir.mkdir(exist_ok=True)
        acc_file = str(save_dir / f'accuracies.npz')  
        np.savez(acc_file, src_acc=self.src_acc_dict, tgt_acc=self.tgt_acc_dict,
                 ensemble_weights=self.ensemble_weights_dict)
        agg_file = str(save_dir / f'aggregators.npz')
        np.savez(agg_file, aggregators=self.aggregator_dict)


def compute_accuracies(cls_dict: Dict[str, Dict[str, np.ndarray]],
                       model_sel_methods_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       src_test_idxs: np.ndarray,
                       tgt_test_idxs: np.ndarray, manual_filter_lambdas: List[str]=[]):
    """Returns an array with the accuracies per lambda and parameter selection methods for source and target domain.
    """
    src_acc = {}
    tgt_acc = {}

    # different lambda models
    #! lambda models accuracy on all source predictions
    for lamb, res_dict in cls_dict.items():
        if lamb in manual_filter_lambdas:
            continue

        # source accuracy
        s_lbls = res_dict['s_lbls']
        s_preds = res_dict['s_preds'].argmax(axis=1)
        # accuracy only on source test split
        s_acc = acc(s_preds[src_test_idxs], s_lbls[src_test_idxs])
        src_acc[lamb] = s_acc
        # target accuracy
        t_lbls = res_dict['t_lbls']
        t_preds = res_dict['t_preds'].argmax(axis=1)
        # accuracy only on target test split
        t_acc = acc(t_preds[tgt_test_idxs], t_lbls[tgt_test_idxs])
        tgt_acc[lamb] = t_acc

    # model selection methods
    #! model selection methods accuracy only evaluated on test split of source predictions
    for model_sel_method_key, preds in model_sel_methods_preds.items():
        s_preds, s_lbls, t_preds, t_lbls = preds
        s_acc = acc(s_preds, s_lbls)
        src_acc[model_sel_method_key] = s_acc
        t_acc = acc(t_preds, t_lbls)
        tgt_acc[model_sel_method_key] = t_acc

    return src_acc, tgt_acc
