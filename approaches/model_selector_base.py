from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import misc.aux_numpy as aux_np
import copy

from misc.helpers import get_train_test_split_idxs, get_weights_numpy


class ModelSelector(ABC):
    """Base class for all model selection methods.
    Examples: agg, bp, dev, source_regression, etc.
    """

    def __init__(self, manual_filter_lambdas: List[str] = [], train_val_split: float = 0.5, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.manual_filter_lambdas = manual_filter_lambdas
        self.train_val_split = train_val_split

        self.n_source = 0  # num of (train) source data points
        self.n_target = 0  # num of (train) target data points
        self.n_source_test = 0  # num of (test) source data points
        self.n_target_test = 0  # num of (test) target data points

        self.n_classes = 0  # num classification classes
        self.lambdas = []  # lambda parameter values
        self.all_lambdas = []  # lambdas used for aggregation method

        self.domains_name = ''
        self.da_method = ''

        self._src_train_idxs = None
        self._src_test_idxs = None
        self._tgt_train_idxs = None
        self._tgt_test_idxs = None

        # * Inputs to the algorithm
        # in these tensors all rows are normalized to one
        # n_source is reassigned after creation of the train-test split
        # 'training data' for model selection method
        self.importance_weights = None # (n_source, )
        self.source_pred_probabilities = None # (n_lambdas, n_source, n_classes)
        self.source_label_one_hot = None # (n_source, n_classes)
        self.target_pred_probabilities = None # (n_lambdas, n_source, n_classes)
        self.target_labels = None # (n_target, )

        # 'test data' for model selection method
        self.importance_weights_test = None
        self.source_pred_probabilities_test = None # (n_lambdas, n_source_test, n_classes)
        self.source_label_one_hot_test = None # (n_source_test, n_classes)
        self.target_pred_probabilities_test = None # (n_lambdas, n_target_test, n_classes)


        # * Output of model selector
        self.source_predictions_test = None # (n_source_test, )
        self.source_labels_test = None # (n_source_test, )
        self.target_predictions_test = None # (n_target_test, )
        self.target_labels_test = None # (n_target_test, )
        self.ensemble_weights = None # (n_lambdas, )

    @property
    def n_lambdas(self):  # num of lambda values
        return len(self.lambdas)

    def _create_train_val_split(self, train_val_split):
        # split source data
        self._src_train_idxs, self._src_test_idxs = get_train_test_split_idxs(self.rng, self.n_source, train_val_split)
        src_train_importance_weights = self.importance_weights[self._src_train_idxs]
        src_test_importance_weights = self.importance_weights[self._src_test_idxs]
        src_train_pred_probs = self.source_pred_probabilities[:, self._src_train_idxs, :]
        src_test_pred_probs = self.source_pred_probabilities[:, self._src_test_idxs, :]
        src_train_labels_one_hot = self.source_label_one_hot[self._src_train_idxs, :]
        src_test_labels_one_hot = self.source_label_one_hot[self._src_test_idxs, :]
        
        # split target data (for aggregation)
        self._tgt_train_idxs, self._tgt_test_idxs = get_train_test_split_idxs(self.rng, self.n_target, train_val_split)
        tgt_train_pred_probs = self.target_pred_probabilities[:, self._tgt_train_idxs, :]
        tgt_test_pred_probs = self.target_pred_probabilities[:, self._tgt_test_idxs, :]
        tgt_train_labels = self.target_labels[self._tgt_test_idxs]
        tgt_test_labels = self.target_labels[self._tgt_test_idxs]

        train_data = (src_train_importance_weights, src_train_pred_probs, src_train_labels_one_hot, tgt_train_pred_probs, tgt_train_labels)
        test_data = (src_test_importance_weights, src_test_pred_probs, src_test_labels_one_hot, tgt_test_pred_probs, tgt_test_labels)

        return train_data, test_data

    def _set_source_target_activations_per_lambda(
            self, lambda_source_act: Dict[str, Dict[str, np.ndarray]]):
        """Add the source activations (before softmax)"""
        source_preds = []
        target_preds = []
        source_labels = None
        target_labels = None
        for lamb in lambda_source_act.keys():
            s_val_preds = lambda_source_act[lamb]['s_preds']
            t_val_preds = lambda_source_act[lamb]['t_preds']
            s_val_lbls = lambda_source_act[lamb]['s_lbls']
            t_val_lbls = lambda_source_act[lamb]['t_lbls']

            if np.any(np.isnan(s_val_preds)):
                print(f'Skipping lambda {lamb} due to NaN predictions..')
                continue

            self.lambdas.append(lamb)
            if self.n_target <= 0:
                self.n_target = t_val_preds.shape[0]
            if self.n_source <= 0:
                self.n_source = s_val_preds.shape[0]
            if self.n_classes <= 0:
                self.n_classes = s_val_preds.shape[1]
            source_preds.append(aux_np.softmax(s_val_preds))
            target_preds.append(aux_np.softmax(t_val_preds))
            if source_labels is None:
                source_labels = s_val_lbls
            if target_labels is None:
                target_labels = t_val_lbls

        self.source_pred_probabilities = np.stack(source_preds, axis=0)
        self.target_pred_probabilities = np.stack(target_preds, axis=0)
        self.source_label_one_hot = aux_np.onehot(source_labels, self.n_classes)
        self.target_labels = target_labels
        self.all_lambdas = copy.deepcopy(self.lambdas)

    def _set_importance_weights_from_source(
            self, domain_classifier_act: Dict[str, Dict[str, np.ndarray]]):
        """Add the importance weights for each source data point
        keys = ['s_preds', 't_preds', 's_lbls', 't_lbls']"""
        # shape (n_data_points, 2)
        source_preds_act = domain_classifier_act['s_preds']
        n_s = source_preds_act.shape[0]
        n_t = domain_classifier_act['t_preds'].shape[0]

        self.importance_weights = get_weights_numpy(source_preds_act, n_s, n_t)

    def _set_model_predictions(self, cls_dict: Dict[str, Dict[str, np.ndarray]],
                               iwv_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
        
        self._set_source_target_activations_per_lambda(cls_dict)
        self._set_importance_weights_from_source(iwv_dict)

        if self.manual_filter_lambdas:
            self._manual_filter_models(self.manual_filter_lambdas)

        # * Make train val split for source
        train_data, test_data = self._create_train_val_split(self.train_val_split)
        # train data
        (self.importance_weights,
         self.source_pred_probabilities,
         self.source_label_one_hot, 
         self.target_pred_probabilities, 
         self.target_labels) = train_data
        # test data
        (self.importance_weights_test,
         self.source_pred_probabilities_test,
         self.source_label_one_hot_test, 
         self.target_pred_probabilities_test, 
         self.target_labels_test) = test_data

        # update data sample counts
        self.n_source = self.source_pred_probabilities.shape[1]
        self.n_source_test = self.source_pred_probabilities_test.shape[1]
        self.n_target = self.target_pred_probabilities.shape[1]
        self.n_target_test = self.target_pred_probabilities_test.shape[1]

        # set source labels for test data
        self.source_labels_test = self.source_label_one_hot_test.argmax(axis=1)

    def _manual_filter_models(self, lambdas_to_remove: List[str]):
        keep_lambdas = list(set(self.lambdas) - set(lambdas_to_remove))

        keep_idxs = [keep_lambdas.index(l) for l in keep_lambdas]

        f_softmax_s = self.source_pred_probabilities[keep_idxs]
        f_softmax_t = self.target_pred_probabilities[keep_idxs]

        self.target_pred_probabilities = f_softmax_t
        self.source_pred_probabilities = f_softmax_s
        self.all_lambdas = copy.deepcopy(self.lambdas)
        self.lambdas = [self.all_lambdas[idx] for idx in keep_idxs]

    def _compute_ensemble_predictions(self):
        # make predictions on source
        src_probs = (self.source_pred_probabilities_test * self.ensemble_weights[:, None, None]).sum(0)
        self.source_predictions_test = np.argmax(src_probs, axis=1)
        # make predictions on target
        tgt_probs = (self.target_pred_probabilities_test * self.ensemble_weights[:, None, None]).sum(0)
        self.target_predictions_test = np.argmax(tgt_probs, axis=1)

    @abstractmethod
    def predict(self, cls_dict: Dict[str, Dict[str, np.ndarray]],
                iwv_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Receives a dictionary with all predictions and of all models and selects/aggregates them.

        Args:
            cls_dict (Dict[str, Dict[str, np.ndarray]]): The predictions on the dataset. Dict[lambda]['s_preds', 't_preds', 's_lbls', 't_lbls', 's_da_preds', 't_da_preds']: np.array.shape(n_data, n_class)
            iwv_dict (Dict[str, Dict[str, np.ndarray]]): The domain classifier results. Dict['s_preds', 't_preds', 's_lbls', 't_lbls']: np.array.shape(n_data, n_class), for preds / np.array.shape(n_data,), for lbls

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: source predictions and labels for test split, target predictions for all target data
        """
        pass

    @abstractmethod
    def key_name(self) -> str:
        "Returns the Acronym of the method."
        pass


class LinearRegressionSelector(ModelSelector):

    def __init__(self, manual_filter_lambdas: List[str] = []):
        super().__init__(manual_filter_lambdas)

    @abstractmethod
    def _get_labels_and_data_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _linear_regression(self, X_pred: np.ndarray, y_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # linear regression
        X = np.linalg.pinv(X_pred)
        self.ensemble_weights = np.dot(X, y_labels)  # theta (n_lambdas, ) weight the single models
        self._compute_ensemble_predictions()
        return self.source_predictions_test, self.source_labels_test, self.target_predictions_test, self.target_labels_test

    def predict(self, cls_dict: Dict[str, Dict[str, np.ndarray]], iwv_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        self._set_model_predictions(cls_dict, iwv_dict)
        X_pred, y_labels = self._get_labels_and_data_matrix()
        return self._linear_regression(X_pred, y_labels)


class TargetMajorityVotePrediction(ModelSelector):

    def __init__(self, manual_filter_lambdas: List[str] = []):
        super().__init__(manual_filter_lambdas)

    def _majority_vote_on_predictions(self, pred_probbabilities) -> np.ndarray:
        """Returns the majority vote predictions, given the predictions of multiple models."""
        # pred_probabilites has shape (n_models, n_samples, n_classes)
        preds = np.argmax(pred_probbabilities, axis=2)
        majority_preds = np.zeros(pred_probbabilities.shape[1], dtype=np.int)
        for i in range(pred_probbabilities.shape[1]): # range(n_samples)
            # find majority among models     
            unique, counts = np.unique(preds[:,i], return_counts=True)
            # voted class idx
            majority_preds[i] = unique[np.argmax(counts)] # argmax returns the first index in case of equal occurrences
        return majority_preds

    def _compute_majority_vote_predictions(self):
        tgt_probs = self.target_pred_probabilities_test
        src_probs = self.source_pred_probabilities_test
        self.source_predictions_test = self._majority_vote_on_predictions(src_probs)
        self.target_predictions_test = self._majority_vote_on_predictions(tgt_probs)
        
    def predict(self, cls_dict: Dict[str, Dict[str, np.ndarray]], iwv_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        self._set_model_predictions(cls_dict, iwv_dict)
        # no ensemble weights as we make a majority vote for each prediction
        self.ensemble_weights = np.zeros((self.n_lambdas))
        self._compute_majority_vote_predictions()
        return self.source_predictions_test, self.source_labels_test, self.target_predictions_test, self.target_labels_test


    def key_name(self) -> str:
        return 'target_majority_vote'