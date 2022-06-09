from typing import Dict, List, Tuple

import numpy as np
from approaches.model_selector_base import ModelSelector


class ImportanceWeightedValidation(ModelSelector):

    def __init__(self, manual_filter_lambdas: List[str]):
        super().__init__(manual_filter_lambdas=manual_filter_lambdas)
        self.best_lambda = None

    def _get_dev_iw_risk(self):
        """Implementation of Algorithm 1 'GetRisk' and Algorithm 2 'Deep Embedded Validation' 
        from the paper [1]
        [1] You, Kaichao, Ximei Wang, Mingsheng Long, and Michael I Jordan. “Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation,” n.d., 10.

        """
        # * compute the dev risk (i.e. 1 - acc) for each model

        # select best model with dev on ensemble-train data (subset of overal validation data)
        # select predictions of this best model (according to dev) on ensemble-test data and return them
        src_lbls = self.source_label_one_hot.argmax(axis=1)

        # arrays to store the risks of each model
        dev_risks = np.zeros((self.n_lambdas, ))
        iw_risks = np.zeros((self.n_lambdas, ))

        for l in range(self.n_lambdas):
            # compute weighted loss for all lambdas (matrix operations)
            src_preds = self.source_pred_probabilities[l].argmax(axis=1)
            W = self.importance_weights.squeeze()  # (1, n_source)
            err = 1 - (src_preds == src_lbls)  # contains 0 for correct classification and 1 for wrong classification
            L = W * err

            # estimate coefficient for control variate
            m = np.stack([L, W])  # (2, n_source)
            cov = np.cov(m)[0][1] #! this takes the diagonal element of the covariance matrix  -> should be the covariance 
            var = np.var(W, ddof=1)
            eta = -cov / var
            
            # compute DEV risk
            dev_risk = np.mean(L) + eta*np.mean(W) - eta
            dev_risks[l] = dev_risk

            # compute importance weighted risk (w/o control variate)
            iw_risk = np.mean(L)
            iw_risks[l] = iw_risk

        return dev_risks, iw_risks

    def _select_model(self):
        _, iw_risks = self._get_dev_iw_risk()

        # * select the best model (model with minimal risk)
        model_idx = iw_risks.argmin()

        # output as ensemble weights (here one-hot vector)
        sel_model = np.zeros((self.n_lambdas, ))
        sel_model[model_idx] = 1.
        return sel_model

    def predict(self, cls_dict: Dict[str, Dict[str, np.ndarray]], iwv_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._set_model_predictions(cls_dict, iwv_dict)
        
        self.ensemble_weights = self._select_model()

        self._compute_ensemble_predictions()
        return self.source_predictions_test, self.source_labels_test, self.target_predictions_test, self.target_labels_test

    def key_name(self) -> str:
        return 'iwv'


class DeepEmbeddedValidation(ImportanceWeightedValidation):

    def _select_model(self):
        dev_risks, _ = self._get_dev_iw_risk()

        # * select the best model (model with minimal risk)
        model_idx = dev_risks.argmin()

        # output as ensemble weights (here one-hot vector)
        sel_model = np.zeros((self.n_lambdas, ))
        sel_model[model_idx] = 1.
        return sel_model

    def key_name(self) -> str:
        return 'dev'