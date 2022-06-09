from typing import Any, Dict, List, Tuple, Type
import pandas as pd
import numpy as np
from pathlib import Path

from approaches.aggregation import Aggregator
from approaches.ensemble_trainer import compute_accuracies
from extractor.loader import Loader
from misc.helpers import acc


def _get_idx_lists_from_seed_domain_model_key_dict(accs_dict):
    seeds = list(accs_dict.keys())
    domains = []
    lambdas = []
    for s, domain_dict in accs_dict.items():
        if not domains:
            domains = list(domain_dict.keys())
        for d, lambda_dict in domain_dict.items():
            if not lambdas:
                lambdas = list(lambda_dict.keys())
    return seeds, domains, lambdas


def _create_results_table(accs_dict: dict, add_lambda_mean_median: bool = False) -> pd.DataFrame:
    """Create a pandas dataframe from results dict."""
    seeds = list(accs_dict.keys())
    domains = []
    lambdas = []

    accs_series_list = []
    for s, domain_dict in accs_dict.items():
        if not domains:
            domains = list(domain_dict.keys())
        for d, lambda_dict in domain_dict.items():
            if not lambdas:
                lambdas = list(lambda_dict.keys())
            accs_series_list.append(pd.Series(lambda_dict))
    index = pd.MultiIndex.from_product([seeds, domains], names=['seed', 'domains'])
    results_df = pd.DataFrame(accs_series_list, index=index)
    if add_lambda_mean_median:
        results_df = _add_lambda_mean_median_column(results_df)
    return results_df


def _add_lambda_mean_median_column(results_df: pd.DataFrame, parameter_sel_indices: List[str] = ['agg', 'multi_reg', 'bp', 'dev', 'iwv']) -> pd.DataFrame:
    # get dataframe with lambda results only (without parameter selection methods):
    # create a list with the columns to remove
    available_param_sel_methods = list(set(results_df.columns.values) & set(parameter_sel_indices))

    lambda_df = results_df.drop(labels=available_param_sel_methods, axis=1)

    results_df = results_df.copy()
    results_df['lam_mean'] = lambda_df.mean(axis=1)
    results_df['lam_median'] = lambda_df.median(axis=1)
    return results_df


def _combine_src_target_table(src_df: pd.DataFrame, tgt_df: pd.DataFrame) -> pd.DataFrame:
    src_tgt_acc_df = pd.concat({'source': src_df.copy(), 'target': tgt_df.copy()}, names=['domain'])
    src_tgt_acc_df = src_tgt_acc_df.swaplevel(0, 2).sort_index(level='seed').sort_index(level='domains')
    return src_tgt_acc_df


def create_results_table(src_acc: Dict, tgt_acc: Dict) -> pd.DataFrame:
    src_acc_df = _create_results_table(src_acc)
    tgt_acc_df = _create_results_table(tgt_acc)
    src_tgt_acc_df = _combine_src_target_table(src_acc_df, tgt_acc_df)
    return src_tgt_acc_df


def load_results_table(base_dir: str, processed_results_folder: str = 'processed_results') -> pd.DataFrame:
    """Loads the final results table containing accuracies for every seed, domain, lambdas and parameter selection methods.
    Used for presentin in a jupyter noteboook."""
    base_dir = Path(base_dir)
    results_dir = base_dir / processed_results_folder
    acc_file = results_dir / 'accuracies.npz'
    src_tgt_acc_dict = np.load(acc_file, allow_pickle=True)
    list(src_tgt_acc_dict.keys())
    src_acc = src_tgt_acc_dict['src_acc'].item()
    tgt_acc = src_tgt_acc_dict['tgt_acc'].item()
    src_tgt_acc_df = create_results_table(src_acc, tgt_acc)
    return src_tgt_acc_df


def load_results_table_for_dataset(dataset_results_dir):
    """Loads all results for a full dataset."""
    dsr = Path(dataset_results_dir)
    # collect all result dataframes per method in this dict
    method_results_dict = {}  # str: pd.Dataframe
    for da_method_path in dsr.iterdir():
        da_method = da_method_path.stem
        src_tgt_acc_df = load_results_table(da_method_path)
        method_results_dict[da_method] = src_tgt_acc_df
    method_results_df = pd.concat(method_results_dict, names=['Method'])
    return method_results_df


def prepare_dataset_results_df_for_paper(dataset_results_df, columns_in_order=['0', 'iwv', 'dev', 'source_reg', 'target_majority_reg', 'target_confidence_reg', 'target_majority_vote', 'agg', 'tb']):
    mean_df = dataset_results_df.groupby(level=['Method', 'domain']).mean()
    mean_df = mean_df.xs('target', level='domain')

    std_df = dataset_results_df.groupby(level=['Method', 'domain']).std()
    std_df = std_df.xs('target', level='domain')

    # get lambdas
    def _is_float(element) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False
    lambdas = [l for l in mean_df.columns if _is_float(l)]
    temp_df = mean_df.copy()

    # add target best column to mean table
    mean_df['tb'] = mean_df[lambdas].max(axis=1)
    # add target best column to std table
    tb_idxs = mean_df[lambdas].idxmax(axis=1)
    tb_std_column = pd.Series(dtype=np.float64)
    for idx, col in tb_idxs.iteritems():
        tb_std_column[idx] = std_df.loc[idx, col]
    std_df['tb'] = tb_std_column

    # add source only column -> simply select column with lambda = 0
    # remove all lambdas, but 0
    lambdas_but_0 = lambdas[1:]
    mean_df = mean_df.drop(columns=lambdas_but_0)[columns_in_order]
    std_df = std_df.drop(columns=lambdas_but_0)[columns_in_order]
    return mean_df, std_df


def create_ensemble_weights_table(ew_dict: Dict[str, Any], accs_dict: Dict[str, Any]) -> pd.DataFrame:
    """Create a table with the ensemble weights for each domain, seed and ensemble methods for every lambda.

    Args:
        ew_dict (Dict[str, Any]): The ensemble weights dictionary.
        accs_dict (Dict[str, Any]): Source or target accuracy dictionary. Used to get lambdas only
    Returns:
        pd.DataFrame: A table containing the ensemble weights for each lambda.
    """
    seeds, domains, lambdas = _get_idx_lists_from_seed_domain_model_key_dict(accs_dict)
    seeds, domains, ews_keys = _get_idx_lists_from_seed_domain_model_key_dict(ew_dict)
    # this keeps ordering of lambdas (set operation shuffles the list)
    for ensemble_method in ews_keys:
        lambdas.remove(ensemble_method)

    ensemble_methods = []
    ew_series_list = []
    for s, domain_dict in ew_dict.items():
        for d, ensemble_method_weights_dict in domain_dict.items():
            if not ensemble_methods:
                ensemble_methods = list(ensemble_method_weights_dict.keys())
            for ensemble_method, ensemble_weights in ensemble_method_weights_dict.items():
                ew_series_list.append(pd.Series(data=ensemble_weights, index=lambdas))
    index = pd.MultiIndex.from_product([seeds, domains, ensemble_methods], names=[
                                       'seed', 'domains', 'ensemble_methods'])
    results_df = pd.DataFrame(ew_series_list, index=index)
    return results_df


def load_ensemble_weights_table(base_dir: str, processed_results_folder: str = 'processed_results') -> pd.DataFrame:
    base_dir = Path(base_dir)
    results_dir = base_dir / processed_results_folder
    acc_file = results_dir / 'accuracies.npz'
    src_tgt_acc_dict = np.load(acc_file, allow_pickle=True)
    list(src_tgt_acc_dict.keys())
    src_acc = src_tgt_acc_dict['src_acc'].item()
    tgt_acc = src_tgt_acc_dict['tgt_acc'].item()
    ensemble_weights = src_tgt_acc_dict['ensemble_weights'].item()
    ensemble_weights_df = create_ensemble_weights_table(ensemble_weights, src_acc)
    return ensemble_weights_df


def aggregation_rcond_sweep_per_dataset(results_dir: str, seeds: List[int], results_loader_class: Type[Loader],
                                        rconds: List[float] = [1.0, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]) -> pd.DataFrame:
    """Compute aggregation method (accuracy) for all methods in the results dir for every domain adaptation task for every seed.
    Will produce a table with the following row index: Method, Task, seed, domain (=source/target)"""
    res_dir = Path(results_dir)
    da_methods = [p.stem for p in res_dir.iterdir()]  # list all available domain adaptation methods
    da_method_dfs = {}
    for da_method in da_methods:
        print(f'Aggregation for {da_method}')
        task_dfs = {}
        for domain_idx, domain in enumerate(results_loader_class.domains):
            base_dir = res_dir / da_method
            base_dir = str(base_dir)
            acc_df_rcond, aggs_rcond = aggregation_rcond_sweep(
                base_dir, da_method, seeds, results_loader_class, rconds=rconds, domains_index=domain_idx)
            task_dfs[domain] = acc_df_rcond
        da_method_df = pd.concat(task_dfs, names=['Task'])
        da_method_dfs[da_method] = da_method_df
    res_df = pd.concat(da_method_dfs, names=['Method'])
    return res_df


def aggregation_rcond_sweep(base_dir: str, da_method: str, seeds: List[int], results_loader_class: Type[Loader],
                            rconds: List[float] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                            domains_index: int = 0) -> Tuple[pd.DataFrame, Dict[str, Aggregator]]:
    """Perform aggregation with different rcond parameters for pseudo inverse computation.

    Returns:
        Tuple[pd.Dataframe, Dict[str, Aggregator]]: Results table and Aggregators for every seed.
    """
    # create results loader
    res_loader = results_loader_class(base_dir, da_method)
    domain = res_loader.domains[domains_index]  # select the domains of the dataset DomainNet has multiple options
    rcond_res_per_seed = []
    aggs_per_seed = {}
    for seed in seeds:
        # use loader
        cls_dict = res_loader.get_class_source_target_preds_labels(
            domain=domain, seed=seed)
        iwv_dict = res_loader.get_iwv_predictions(domain=domain, seed=seed)
        rconds_res_dict = {}
        for rc in rconds:
            model_sel_methods_preds = {}  # key: model_selector.key_name, value: Tuple[source_preds, target_preds]
            agg = Aggregator(rcond=rc, eps=0.005, filter_similar_models=False)
            agg.domains_name = domain
            agg.da_method = da_method
            model_sel_methods_preds[agg.key_name()] = agg.predict(cls_dict, iwv_dict)
            aggs_per_seed[seed] = agg  # store aggregator
            agg_key = agg.key_name()
            src_acc, tgt_acc = compute_accuracies(
                cls_dict, model_sel_methods_preds, agg._src_test_idxs, agg._tgt_test_idxs)
            # extract agg source target value
            res_df = pd.DataFrame([pd.Series(src_acc), pd.Series(tgt_acc)], index=['source', 'target'])
            # make a table with rconds - agg
            rconds_res_dict[rc] = res_df[agg_key]
        rcond_res_per_seed.append(pd.DataFrame(rconds_res_dict))
    # create results table
    index = pd.MultiIndex.from_product([seeds, ['source', 'target']], names=['seed', 'domain'])
    display_df = pd.concat(rcond_res_per_seed)
    display_df = display_df.set_index(index)
    return display_df, aggs_per_seed


def aggregation_num_svd_sweep(base_dir: str, da_method: str, seeds: List[int], results_loader_class: Type[Loader],
                              num_svds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
                              domains_index: int = 0) -> Tuple[pd.DataFrame, Dict[str, Aggregator]]:
    """Perform aggregation with different number of singular value parameters for pseudo inverse computation.

    Returns:
        Tuple[pd.Dataframe, Dict[str, Aggregator]]: Results table and Aggregators for every seed.
    """
    # create results loader
    res_loader = results_loader_class(base_dir, da_method)
    domain = res_loader.domains[domains_index]  # select the domains of the dataset DomainNet has multiple options
    n_svd_res_per_seed = []
    aggs_per_seed = {}
    for seed in seeds:
        # use loader
        cls_dict = res_loader.get_class_source_target_preds_labels(
            domain=domain, seed=seed)
        iwv_dict = res_loader.get_iwv_predictions(domain=domain, seed=seed)
        rconds_res_dict = {}
        for n_sv in num_svds:
            model_sel_methods_preds = {}  # key: model_selector.key_name, value: Tuple[source_preds, target_preds]
            agg = Aggregator(num_singular_values=n_sv, eps=0.005, filter_similar_models=False)
            agg.domains_name = domain
            agg.da_method = da_method
            model_sel_methods_preds[agg.key_name()] = agg.predict(cls_dict, iwv_dict)
            aggs_per_seed[seed] = agg  # store aggregator
            agg_key = agg.key_name()
            src_acc, tgt_acc = compute_accuracies(
                cls_dict, model_sel_methods_preds, agg._src_test_idxs, agg._tgt_test_idxs)
            # extract agg source target value
            res_df = pd.DataFrame([pd.Series(src_acc), pd.Series(tgt_acc)], index=['source', 'target'])
            # make a table with rconds - agg
            rconds_res_dict[n_sv] = res_df[agg_key]
        n_svd_res_per_seed.append(pd.DataFrame(rconds_res_dict))
    # create results table
    index = pd.MultiIndex.from_product([seeds, ['source', 'target']], names=['seed', 'domain'])
    display_df = pd.concat(n_svd_res_per_seed)
    display_df = display_df.set_index(index)
    return display_df, aggs_per_seed


def get_aggregator(base_dir: str, da_method: str, seed: int, rcond: float, results_loader_class: Type[Loader],
                   domains_index: int = 0, manual_filter_lambdas: List[str] = []):
    """Use this to analyze the results of a single aggregation (one seed, one domain adaptation method)"""
    # create results loader
    res_loader = results_loader_class(base_dir, da_method)
    domain = res_loader.domains[domains_index]
    # use extractor
    cls_dict = res_loader.get_class_source_target_preds_labels(
        domain=domain, seed=seed)
    iwv_dict = res_loader.get_iwv_predictions(domain=domain, seed=seed)
    agg = Aggregator(rcond=rcond, eps=0.005, filter_similar_models=False,
                     manual_filter_lambdas=manual_filter_lambdas)

    model_sel_methods_preds = {}  # key: model_selector.key_name, value: Tuple[source_preds, target_preds]

    model_sel_methods_preds[agg.key_name()] = agg.predict(cls_dict, iwv_dict)
    agg_key = agg.key_name()
    src_acc, tgt_acc = compute_accuracies(
        cls_dict, model_sel_methods_preds, agg._src_test_idxs, agg._tgt_test_idxs)
    agg_results_df = pd.DataFrame([pd.Series(src_acc), pd.Series(tgt_acc)], index=['source', 'target'])
    return agg, agg_results_df
