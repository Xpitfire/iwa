import numpy as np
import torch
from typing import Dict, List, Tuple
from approaches.model_selector_base import ModelSelector
from misc.helpers import get_weights_numpy
import misc.aux_numpy as aux_np
from operator import itemgetter
import copy


class Aggregator(ModelSelector):
    """This class implements the Aggregation algorithm
    for multi-class classification. Seeds are not handled. 
    If multiple seeds available, create multiple instances of this class.

    Args:
        device (_type_, optional): The cuda device. Defaults to None.
        eps (float, optional): The similarity parameter. 
        Models, whose cosine similarity of their target predictions is smaller than this parameter are considered equal. 
        Defaults to 0.005.
        num_singular_values (int, optional): The number of singular values to keep for inversion of matrix G. 
                    If specified uses np auxiliary function for inversion. Otherwise np.linalg.pinv().
    """

    def __init__(self, rcond=1e-2, filter_similar_models=False, eps=0.02, manual_filter_lambdas: List[str] = [],
                 num_singular_values: int = None):
        super().__init__(manual_filter_lambdas=manual_filter_lambdas)
        # hyperparameter epsilon similarity
        self.filter_similar_models = filter_similar_models
        self.eps = eps
        # hyperparameter rcond for np.linalg.pinv()
        self.rcond = rcond
        self.num_singular_values = num_singular_values

        # * Temporary values of Aggregation algorithm
        self.matrix_G_similarity = None
        self.matrix_G = None
        self.matrix_G_inverse = None
        self.matrix_G_condition_numbers = None
        self.matrix_F = None
        # singular values, the cutoff threshold for rcond,
        # and an boolean array indicating larger singular values.
        self.matrix_G_pinv_analysis: Tuple = None

        # * Output of Aggregation algorithm
        self.aggregation_weights = None

    @property
    def n_lambdas(self):  # num of lambda values
        return len(self.lambdas)

    def _filter_similar_models(self):
        """Filters similar models (trained with different lambda hyperparameters) for the inverse computation.
        Similarity is computed based on cosine similarity. Filtering out similiar models makes inversion of the G matrix more stable.
        All models with cosine similarity smaller than 'eps' are considered equal. """
        l = self.n_lambdas
        # filter by too similar
        sim = self.matrix_G_similarity[0]
        select_idx = []
        i = 0
        # find indices which to keep
        while i < l:
            j = i + 1
            cnt = i
            max_ = i
            while j < l - 1:
                if np.abs(sim[i] - sim[j]) < self.eps:
                    cnt += 1
                if cnt < l and sim[cnt] > sim[max_]:
                    max_ = j
                j += 1
            cnt += 1
            select_idx.append(max_)
            i = cnt

        keep_idx = select_idx

        # assign values
        f_lambdas_ = itemgetter(*keep_idx)(self.lambdas)
        f_softmax_s = self.source_pred_probabilities[keep_idx]
        f_softmax_t = self.target_pred_probabilities[keep_idx]

        # make single number to list
        if isinstance(f_lambdas_, float):
            keep_idx = [self.lambdas.index(f_lambdas_)]
            f_lambdas_ = [f_lambdas_]
            print(
                'WARNING: Aggregation filter received only single lambda value after filtering!'
            )
        # fallback
        elif f_lambdas_ is None or f_lambdas_ == 0:
            keep_idx = [i for i in range(l)]
            f_lambdas_ = self.lambdas
            print(
                'WARNING: Aggregation filter failed. Fallback to original lambdas!'
            )

        self.target_pred_probabilities = f_softmax_t
        self.source_pred_probabilities = f_softmax_s
        self.all_lambdas = copy.deepcopy(self.lambdas)
        self.lambdas = f_lambdas_

    def _compute_similarity_matrix(self):
        """Computes the G matrix with target predictions on all models.
        """
        l = self.n_lambdas
        m = self.n_target
        c = self.n_classes

        # filter by too similar
        self.matrix_G_similarity = np.zeros((l, l))
        for p in range(l):
            for q in range(l):
                # prediction similarity between models
                # Full-batch dot product between prediction probabilities
                # motivated by cosine-similarity
                self.matrix_G_similarity[p, q] = (self.target_pred_probabilities[p, :] *
                    self.target_pred_probabilities[q, :]).sum().sum() / m

    def _compute_aggregation_weights(self):
        """Compute the aggregation weights."""
        l = self.n_lambdas
        m = self.n_target
        n = self.n_source
        c = self.n_classes

        self.matrix_G = np.zeros((l, l))
        for p in range(l):
            for q in range(l):

                self.matrix_G[p,q] = (self.target_pred_probabilities[p, :] *
                    self.target_pred_probabilities[q, :]).sum(0).sum(0) / m
                
        self.matrix_G_condition_numbers = np.array([
            np.linalg.cond(self.matrix_G, p=x)
            for x in ['fro', 1, 2, np.inf]
        ])

        self.matrix_F = np.zeros((l, c))
        for k in range(l):
            F_ = np.zeros((n, c))
            for i in range(n):
                F_[i] = (self.importance_weights[i] *
                         np.matmul(self.source_pred_probabilities[k, i], self.source_label_one_hot[i].T)) / n
            self.matrix_F[k] = np.sum(F_, axis=0)

        # compute pinv analysis
        self.matrix_G_pinv_analysis = aux_np.get_pinv_analysis(self.matrix_G, self.rcond)
        if self.num_singular_values:
            self.matrix_G_inverse = aux_np.pinv_with_singular_values(
                self.matrix_G, self.num_singular_values)
        else:
            self.matrix_G_inverse = np.linalg.pinv(
                self.matrix_G, rcond=self.rcond)
        self.aggregation_weights = np.matmul(self.matrix_G_inverse,
                                             self.matrix_F)
        # assign result
        self.ensemble_weights = self.aggregation_weights[:, 0]
        

    def predict(self, cls_dict: Dict[str, Dict[str, np.ndarray]],
                iwv_dict: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        self._set_model_predictions(cls_dict, iwv_dict)
        self._compute_similarity_matrix()
        self._compute_aggregation_weights()
        self._compute_ensemble_predictions()
        return self.source_predictions_test, self.source_labels_test, self.target_predictions_test, self.target_labels_test

    def key_name(self):
        return 'agg'
