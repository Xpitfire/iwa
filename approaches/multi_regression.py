
import numpy as np
import torch
import misc.aux_numpy as aux_np

from approaches.model_selector_base import LinearRegressionSelector, ModelSelector
from typing import Dict, Tuple
import copy

class SourceLinearRegression(LinearRegressionSelector):

    def _get_labels_and_data_matrix(self)-> Tuple[np.ndarray, np.ndarray]:
        # data matrix, shape = (num source samples, num_lambdas)
        X_pred = self.source_pred_probabilities.reshape(self.n_lambdas, -1).transpose()

        # labels, shape = (num_source samples)
        y_labels = self.source_label_one_hot.reshape(-1)

        return X_pred, y_labels

    def key_name(self) -> str:
        return 'source_reg'

class TargetMajorityVoteLinearRegression(LinearRegressionSelector):

    def _get_labels_and_data_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        
        # data matrix
        X_pred = self.target_pred_probabilities.reshape(self.n_lambdas, -1).transpose()
        # construct pseudo target labels based on majority voting
        target_preds = np.argmax(self.target_pred_probabilities, axis=2)
        target_majority_preds = np.zeros(self.n_target, dtype=np.int64) # number of target samples
        for i in range(self.n_target): 
            # find majority among models     
            unique, counts = np.unique(target_preds[:,i], return_counts=True)
            # voted class idx
            target_majority_preds[i] = unique[np.argmax(counts)] # argmax returns the first index in case of equal occurrences

        # create onehot vector
        target_labels_one_hot = aux_np.onehot(target_majority_preds, self.n_classes)
        y_labels = target_labels_one_hot.reshape(-1)

        return X_pred, y_labels

    def key_name(self) -> str:
        return 'target_majority_reg'

class TargetConfidenceLinearRegression(LinearRegressionSelector):

    def _get_labels_and_data_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # data matrix
        X_pred = self.target_pred_probabilities.reshape(self.n_lambdas, -1).transpose()

        # labels
        # construct pseudo target labels based on confidence
        # take argmax of all averaged softmax probabilities
        avg_target_preds_probs = self.target_pred_probabilities.mean(axis=0)
        target_preds = np.argmax(avg_target_preds_probs, axis=1)

        target_labels_one_hot = aux_np.onehot(target_preds, self.n_classes)
        y_labels = target_labels_one_hot.reshape(-1)

        return X_pred, y_labels

    def key_name(self) -> str:
        return 'target_confidence_reg'