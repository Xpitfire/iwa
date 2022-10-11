import torch
import torch.nn.functional as F

import os
import wandb
import pandas as pd
import numpy as np
from dataloader.dataloader import data_generator as ts_data_generator
from dataloader.mdn_dataloader import mdn_data_generator
from dataloader.ar_dataloader import ar_data_generator
from dataloader.tm_dataloader import tm_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from hydra.utils import get_original_cwd, to_absolute_path

from configs.sweep_params import sweep_alg_hparams
from misc.utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from misc.utils import calc_dev_risk as single_calc_dev_risk, calculate_risk as single_calculate_risk
from misc.utils import batch_calc_dev_risk, batch_calculate_risk
import warnings

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from misc.utils import AverageMeter

torch.backends.cudnn.benchmark = True  # to fasten TCN


class cross_domain_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.num_sweeps = args.num_sweeps
        self.seed = args.seed
        self.iwv_method = args.iwv_method
        self.config = args

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_name

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(get_original_cwd(), args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.dataset_configs.device = args.device
        self.dataset_configs.debug = args.debug
        self.dataset_configs.used_backbone = args.backbone
        self.dataset_configs.used_da_method = args.da_method

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}
        self.default_hparams['seed'] = self.seed
        if args.da_method == 'DIRT' and args.backbone == 'Pretrained2D':
            self.default_hparams['batch_size'] = 4

    def train(self):
        run_name = f"{self.run_description}"
        self.hparams = self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files:

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.

        self.metrics = {'accuracy': [], 'f1_score': [], 'src_risk': [], 'trg_risk': [], 'dev_risk': []}

        for i in scenarios:

            cls_predictions = {}
            iwv_predictions = {}

            src_id = i[0]
            trg_id = i[1]
            ds_name = f"{src_id}_src-{trg_id}_tgt"

            for run_id in range(self.num_runs):  # specify number of consecutive runs
                seed = self.seed + run_id

                # fixing random seed
                fix_randomness(seed)

                seed = str(seed)

                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                backbone_fe = get_backbone_class(self.backbone)

                # loop over lambdas
                for lamb in self.hparams["lambdas"]:

                    # Logging
                    self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                    src_id, trg_id, seed, lamb=lamb)

                    key = f"{ds_name}-{lamb}"

                    algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device, lamb)
                    algorithm.to(self.device)

                    # Average meters
                    loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                    # training..
                    if not self.config.debug:
                        for epoch in range(1, self.hparams["num_epochs"] + 1):
                            joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                            len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                            algorithm.train()

                            for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                                src_x, src_y, trg_x = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                    trg_x.float().to(self.device)

                                if self.da_method == "DANN" or self.da_method == "CoDATS":
                                    losses = algorithm.update(src_x, src_y, trg_x, step, epoch, len_dataloader)
                                else:
                                    losses = algorithm.update(src_x, src_y, trg_x)

                                for k, v in losses.items():
                                    loss_avg_meters[k].update(v, src_x.size(0))

                            # logging
                            wandb_log = {'epoch': epoch}
                            self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                            for k, v in loss_avg_meters.items():
                                self.logger.debug(f'{k}\t: {v.avg:2.4f}')
                                wandb_log[f'train/cls-{k}'] = v.avg
                            if self.config.use_wandb:
                                wandb.log(wandb_log)
                            self.logger.debug(f'-------------------------------------')

                    save_checkpoint(self.home_path, algorithm, scenarios, self.dataset_configs,
                                    self.scenario_log_dir, self.hparams)

                    src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels = self.evaluate(algorithm)
                    self.calc_results_per_run(algorithm, src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels)

                    # init dicts
                    if key not in cls_predictions:
                        cls_predictions[key] = {}
                    if seed not in cls_predictions[key]:
                        cls_predictions[key][seed] = []

                    cls_predictions[key][seed].append({
                        's_preds': src_pred_labels, 
                        't_preds': trg_pred_labels, 
                        's_lbls': src_true_labels,
                        't_lbls': trg_true_labels
                    })

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method + '-iwv', self.exp_log_dir,
                                                                src_id, trg_id, seed, lamb='none')

                # get IWV algorithm
                if self.iwv_method == "IWV_DANN":
                    algorithm_class = get_algorithm_class('IWV_DANN')
                elif self.iwv_method == 'IWV_Domain_Classifier_With_Source':
                    algorithm_class = get_algorithm_class('IWV_Domain_Classifier_With_Source')
                elif self.iwv_method == 'IWV_Domain_Classifier':
                    algorithm_class = get_algorithm_class('IWV_Domain_Classifier')
                else:
                    raise ValueError(f"Unknown IWV method: {self.iwv_method}")

                backbone_fe = get_backbone_class(self.backbone)

                # IWV domain classifier
                algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device, lamb)
                algorithm.to(self.device)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                if not self.config.debug:
                    for epoch in range(1, self.hparams["iwv_epochs"] + 1):
                        joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                        len_dataloader = min(len(self.src_train_dl), len(self.trg_train_dl))
                        algorithm.train()

                        for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                            src_x, trg_x = src_x.float().to(self.device), trg_x.float().to(self.device)
                            p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
                            alpha = 2. / (1. + np.exp(-10 * p)) - 1
                            losses = algorithm.update(src_x, src_y, trg_x, alpha)

                            for k, v in losses.items():
                                loss_avg_meters[k].update(v, src_x.size(0))

                        # logging
                        wandb_log = {'epoch': epoch}
                        self.logger.debug(f'[Epoch : {epoch}/{self.hparams["iwv_epochs"]}]')
                        for k, v in loss_avg_meters.items():
                            self.logger.debug(f'{k}\t: {v.avg:2.4f}')
                            wandb_log[f'train/iwv-{k}'] = v.avg
                        if self.config.use_wandb:
                            wandb.log(wandb_log)
                        self.logger.debug(f'-------------------------------------')

                save_checkpoint(self.home_path, algorithm, scenarios, self.dataset_configs,
                                    self.scenario_log_dir, self.hparams)

                src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels = self.evaluate(algorithm, iwv_domain_clf=True)

                if ds_name not in iwv_predictions:
                    iwv_predictions[ds_name] = {}
                if seed not in iwv_predictions[ds_name]:
                    iwv_predictions[ds_name][seed] = []
                
                iwv_predictions[ds_name][seed].append({
                    's_preds': src_pred_labels, 
                    't_preds': trg_pred_labels, 
                    's_lbls': src_true_labels,
                    't_lbls': trg_true_labels
                })

            # create results directory
            log_dir = os.path.join(self.exp_log_dir, "adatime_agg_" + self.da_method)
            os.makedirs(log_dir, exist_ok=True)
            # save the predictions for cls
            pred_file = os.path.join(log_dir, f'cls_pred_dataset_{ds_name}.npz')
            np.savez(pred_file, cls_predictions)
            # save the predictions for iwv
            pred_file = os.path.join(log_dir, f'iwv_pred_dataset_{ds_name}.npz')
            np.savez(pred_file, iwv_predictions)

        # logging metrics
        self.calc_overall_results()
        average_metrics = {metric: np.mean(value) for (metric, value) in self.metrics.items()}
        if self.config.use_wandb:
            wandb.log(average_metrics)
            wandb.log({'hparams': wandb.Table(
                dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
                allow_mixed_types=True)})
            wandb.log({'results': wandb.Table(dataframe=self.df_results, allow_mixed_types=True)})
            wandb.log({'avg_results': wandb.Table(dataframe=self.df_avg_results, allow_mixed_types=True)})
            wandb.log({'std_results': wandb.Table(dataframe=self.df_std_results, allow_mixed_types=True)})


    @torch.no_grad()
    def evaluate(self, algorithm, iwv_domain_clf=False):
        feature_extractor = algorithm.feature_extractor.to(self.device)
        classifier = algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        src_pred_labels = []
        src_true_labels = []
        trg_pred_labels = []
        trg_true_labels = []

        for data, labels in self.src_test_dl:
            data = data.float().to(self.device)
            if iwv_domain_clf:
                labels = torch.zeros(len(data)).long().to(self.device)
            else:
                labels = labels.view((-1)).long().to(self.device)

            # forward pass
            features = feature_extractor(data)
            predictions = classifier(features)

            src_pred_labels.append(predictions.cpu().numpy())
            src_true_labels.append(labels.long().cpu().numpy())

        for data, labels in self.trg_test_dl:
            data = data.float().to(self.device)
            if iwv_domain_clf:
                labels = torch.ones(len(data)).long().to(self.device)
            else:
                labels = labels.view((-1)).long().to(self.device)

            # forward pass
            features = feature_extractor(data)
            predictions = classifier(features)

            # compute loss
            loss = F.cross_entropy(predictions, labels)
            total_loss_.append(loss.item())

            trg_pred_labels.append(predictions.cpu().numpy())
            trg_true_labels.append(labels.long().cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss
        return src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        if self.dataset == 'MINI_DOMAIN_NET':
            data_generator = mdn_data_generator
        elif self.dataset == 'AMAZON_REVIEWS':
            data_generator = ar_data_generator
        elif self.dataset == 'TRANSFORMED_MOONS':
            data_generator = tm_data_generator
        elif self.dataset == 'EEG' or self.dataset == 'WISDM' or self.dataset == 'HAR' or self.dataset == 'HHAR_SA':
            data_generator = ts_data_generator
        
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self, algorithm, src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels):
        '''
        Calculates the acc, f1 and risk values for each cross-domain scenario
        '''
        src_pred_labels = np.concatenate(src_pred_labels, axis=0).argmax(axis=-1).reshape(-1)
        src_true_labels = np.concatenate(src_true_labels, axis=0).reshape(-1)

        self.src_acc, self.src_f1 = _calc_metrics(src_pred_labels, src_true_labels, self.scenario_log_dir,
                                                  self.home_path,
                                                  self.dataset_configs.class_names)
        
        trg_pred_labels = np.concatenate(trg_pred_labels, axis=0).argmax(axis=-1).reshape(-1)
        trg_true_labels = np.concatenate(trg_true_labels, axis=0).reshape(-1)

        self.trg_acc, self.trg_f1 = _calc_metrics(trg_pred_labels, trg_true_labels, self.scenario_log_dir,
                                                  self.home_path,
                                                  self.dataset_configs.class_names)
        
        if self.dataset == 'MINI_DOMAIN_NET' or \
            self.dataset == 'AMAZON_REVIEWS' or \
                self.dataset == 'TRANSFORMED_MOONS':
            calculate_risk = batch_calculate_risk
            calc_dev_risk = batch_calc_dev_risk
        elif self.dataset == 'EEG' or self.dataset == 'WISDM' or self.dataset == 'HAR' or self.dataset == 'HHAR_SA':
            calculate_risk = single_calculate_risk
            calc_dev_risk = single_calc_dev_risk
        
        self.src_risk = calculate_risk(algorithm, self.src_test_dl, self.device)
        self.trg_risk = calculate_risk(algorithm, self.trg_test_dl, self.device)
        self.dev_risk = calc_dev_risk(algorithm, self.src_train_dl, self.trg_train_dl, self.src_test_dl,
                                      self.dataset_configs, self.device)

        run_metrics = {
            'src_acc': self.src_acc,
            'src_f1': self.src_f1,
            'trg_acc': self.trg_acc,
            'trg_f1': self.trg_f1,
            'src_risk': self.src_risk,
            'trg_risk': self.trg_risk,
            'dev_risk': self.dev_risk
        }

        df = pd.DataFrame(columns=["src_acc", "src_f1", "trg_acc", "trg_f1", "src_risk", "trg_risk", "dev_risk"])
        df.loc[0] = [self.src_acc, self.src_f1, self.trg_acc, self.trg_f1, self.src_risk, self.trg_risk, self.dev_risk]

        for (key, val) in run_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(val)

        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df
        
        if self.config.use_wandb:
            wandb.log(run_metrics)

    def calc_overall_results(self):
        exp = self.exp_log_dir
        
        # for exp in experiments:
        results = pd.DataFrame(
                columns=["scenario", "lambda", "src_acc", "src_f1", "trg_acc", "trg_f1", "src_risk", "trg_risk", "dev_risk"])

        single_exp = os.listdir(exp)
        single_exp = [i for i in single_exp if "_tgt-" in i and 'none' not in i]
        single_exp.sort()

        src_ids = [single_exp[i].split("-")[2] for i in range(len(single_exp))]
        scenarios_ids = np.unique(["_".join(i.split("-")[2:4]) for i in single_exp])
        num_runs = len(src_ids) // len(scenarios_ids)

        for scenario in single_exp:
            scenario_dir = os.path.join(exp, scenario)
            scores = pd.read_excel(os.path.join(scenario_dir, 'scores.xlsx'))
            results = results.append(scores)
            name = '_'.join(scenario.split('-')[:-1])
            idx = 0
            for s in scenarios_ids:
                if s in name:
                    break
                idx += 1
            results.iloc[len(results) - 1, 0] = scenarios_ids[idx]
            results.iloc[len(results) - 1, 1] = scenario.split('-')[-1]

        avg_results = results.groupby(np.arange(len(results)) // num_runs).mean()
        std_results = results.groupby(np.arange(len(results)) // num_runs).std()

        avg_results.insert(0, "scenario", list(scenarios_ids), True)
        avg_results.loc[len(avg_results)] = avg_results.mean()
        avg_results = avg_results.fillna('Mean')
        std_results.insert(0, "scenario", list(scenarios_ids), True)

        report_save_path_all = os.path.join(exp, f"all_results.xlsx")
        report_save_path_avg = os.path.join(exp, f"average_results.xlsx")
        report_save_path_std = os.path.join(exp, f"std_results.xlsx")

        results.to_excel(report_save_path_all)
        avg_results.to_excel(report_save_path_avg)
        std_results.to_excel(report_save_path_std)
        
        self.df_results = results
        self.df_avg_results = avg_results
        self.df_std_results = std_results
