import torch
import copy
import os
import time
import uuid
import numpy as np
from tqdm import tqdm
from lighter.collectible import BaseCollectible
from lighter.writer import BaseWriter
from misc.helpers import load_function, map_reduce, set_seed, EarlyStopping
from torch.cuda.amp import autocast, GradScaler
import warnings

# rollout epochs + batches -> interpolate from 0 to 1
def compute_alpha(config, loader, i, epoch, gamma=10):
    # disable gradient flow entirely
    if not hasattr(config.trainer, 'alpha_type') or config.trainer.alpha_type is None: return 0
    # distinguish between different alpha types
    if config.trainer.alpha_type == 'reversal_anneal':
        # anneal alpha as in the original DANN paper
        p = float(i + epoch * len(loader)) / config.trainer.epochs / len(loader)
        alpha = 2. / (1. + np.exp(-gamma * p)) - 1
    elif config.trainer.alpha_type == 'reversal':
        # do not anneal alpha and return 1:1 reversal scale
        alpha = 1
    elif config.trainer.alpha_type == 'discriminator':
        # use as a regular discriminator, this undos the gradient reversal in the gradient reversal layer
        alpha = -1
    else:
        raise NotImplementedError('Unknown alpha type!')
    return alpha


class Trainer(object):
    r"""Main training loop for the current approach.
    """
    def __init__(self, config, model, dataloaders, experiment_id, writer, collectible, start_epoch=0, verbose=False):
        self.config = config
        self.model = model
        self.verbose = verbose
        if len(dataloaders) == 2:
            self.train_loader, self.eval_loader = dataloaders
        elif len(dataloaders) == 3:
            self.train_loader, self.eval_loader, self.test_loader = dataloaders
        self.save_interval = config.trainer.save_interval
        # handle experiment related variables
        self.experiment_id = experiment_id
        self.checkpoint_dir = os.path.join(config.trainer.checkpoint_dir, 
                                           self.experiment_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = start_epoch
        print(f'Started experiment: {self.experiment_id}')
        self.writer = writer
        self.collectible = collectible
        # load training related metric, optimizer and criterion functions
        self.optimizer, self.scheduler = load_function(config.trainer.optimizer_file, config.trainer.optimizer)(config, self.model)
        self.metric = load_function(config.trainer.metric_file, config.trainer.metric)
        self.criterion = load_function(config.trainer.criterion_file, config.trainer.criterion)
        # initialize early stopping
        self.early_stopping = EarlyStopping(self, patience=config.trainer.early_stopping_patience, 
                                            min_epochs=config.trainer.min_epochs, 
                                            verbose=verbose)
        # Creates a GradScaler once at the beginning of training.
        self.scaler = GradScaler()

    def save_checkpoint(self, epoch, info='', overwrite=False):
        """Saves a model checkpoint"""
        conf = copy.deepcopy(self.config)
        conf.device = None # required because device is not serializable
        state = {
            'info': info,
            'epoch': epoch,
            'experiment_id': self.experiment_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': conf
        }
        try:
            os.mkdir(self.checkpoint_dir)
        except FileExistsError:
            pass
        if overwrite:
            filename = os.path.join(self.checkpoint_dir, f'{info}.pth')
        else:
            filename = os.path.join(self.checkpoint_dir, f'{info}checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        if self.verbose: print("Saving checkpoint: {} ...".format(filename))

    def resume_checkpoint(self, resume_path):
        """Resumes training from an existing model checkpoint"""
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    @torch.enable_grad()
    def _train(self, epoch):
        """Training step through the batch."""
        losses = []
        accs = []
        values = []
        self.model.train()
        for i, (xs, ys, xt, yt) in enumerate(self.train_loader):
            self.config.source_only = False
            # create alpha value for DANN according to paper https://arxiv.org/abs/1505.07818
            alpha = compute_alpha(self.config, self.train_loader, i, epoch)
            # prepare input
            if self.config.trainer.criterion == "source_only":
                self.config.source_only = True
                input = xs.to(self.config.device)
            elif self.config.approach.lambda_ == 0:
                input = torch.cat([xs.to(self.config.device), xs.to(self.config.device)], dim=0)
            else:
                input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            s_target = ys.to(self.config.device)
            t_target = yt.to(self.config.device)
            # reset gradients
            self.optimizer.zero_grad()
            # Runs the forward pass with autocasting for half precision.
            with autocast():
                # forward through the model
                output = self.model(input, alpha)
                # compute loss
                loss, loss_items = self.criterion(self.config, output, s_target, t_target)
            if self.config.trainer.use_mixed_precission:
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if self.config.trainer.apply_gradient_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_gradient_norm)
            # evaluate metrics
            acc = self.metric(self.config, output, s_target, t_target)
            # collect the stats
            accs.append(acc)
            losses.append(loss_items)
            vals = {'alpha': alpha, 'lambda': self.config.approach.lambda_}
            values.append(vals)
            if self.config.trainer.use_mixed_precission:
                # perform optimization step
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                self.scaler.update()
            else:
                self.optimizer.step()
            # update stats for tensorboard
            self.collectible.update(category='train', **loss_items)
            self.collectible.update(category='train', **acc)
            self.collectible.update(category='train', **vals)
        return {'losses': losses, 'accs': accs, 'values': values}

    @torch.no_grad()
    def _eval(self, epoch):
        """Evaluation step through the batch"""
        losses = []
        accs = []
        values = []
        self.model.eval()
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            self.config.source_only = False
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            s_target = ys.to(self.config.device)
            t_target = yt.to(self.config.device)
            # forward through the model
            output = self.model(input)
            # compute loss
            loss, loss_items = self.criterion(self.config, output, s_target, t_target)
            # evaluate metrics
            acc = self.metric(self.config, output, s_target, t_target)
            # collect the stats
            accs.append(acc)
            losses.append(loss_items)
            vals = {'alpha': 1.0, 'lambda': self.config.approach.lambda_}
            values.append(vals)
            # update stats for tensorboard
            self.collectible.update(category='eval', **loss_items)
            self.collectible.update(category='eval', **acc)
            self.collectible.update(category='eval', **vals)
        return {'loss': loss, 'losses': losses, 'accs': accs, 'values': values}

    def init_best_model(self):
        """Get the best model"""
        if self.early_stopping.patience > 0 and self.early_stopping.best_model is not None:
            del self.model
            self.model = copy.deepcopy(self.early_stopping.best_model)
        return self.model

    @torch.no_grad()
    def val_cls_preds(self, include_da_preds=True):
        """Make predictions on evaluation dataset."""
        self.model.eval()
        s_preds = []
        t_preds = []
        s_da_preds = []
        t_da_preds = []
        s_lbls = []
        t_lbls = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            b = xs.shape[0]
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            # forward through the model
            classifier_preds, da_preds = self.model(input)
            # predictions
            s_pred = classifier_preds[:b].cpu().numpy()
            t_pred = classifier_preds[b:].cpu().numpy()
            s_preds.append(s_pred)
            t_preds.append(t_pred)
            s_lbls.append(ys.cpu().numpy())
            t_lbls.append(yt.cpu().numpy())
            if include_da_preds:
                da_source_pred = da_preds[:b].cpu().numpy()
                da_target_pred = da_preds[b:].cpu().numpy()            
                s_da_preds.append(da_source_pred)
                t_da_preds.append(da_target_pred)
        res = {
            's_preds': s_preds, 
            't_preds': t_preds, 
            's_lbls': s_lbls,
            't_lbls': t_lbls
        }
        if include_da_preds:
            res['s_da_preds'] = s_da_preds
            res['t_da_preds'] = t_da_preds
        return res

    @torch.no_grad()
    def val_iwv_preds(self):
        self.model.eval()
        s_preds = []
        t_preds = []
        s_lbls = []
        t_lbls = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            b = xs.shape[0]
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            # forward through the model 
            da_preds, _ = self.model(input) #! da_preds.shape = (batch_size, 2), _ (see output of domain classifier forward())
            # predictions
            s_pred = da_preds[:b].cpu().numpy()
            t_pred = da_preds[b:].cpu().numpy()
            s_preds.append(s_pred)
            t_preds.append(t_pred)
            s_lbls.append(torch.zeros_like(ys).cpu().numpy())
            t_lbls.append(torch.ones_like(yt).cpu().numpy())
        res = {
            's_preds': s_preds, 
            't_preds': t_preds, 
            's_lbls': s_lbls,
            't_lbls': t_lbls
        }
        return res

    def run(self):
        """Main run loop over multiple epochs"""
        train_losses = []
        eval_losses = []
        train_accs = []
        eval_accs = []
        train_values = []
        eval_values = []
        self.start_epoch += 1
        # initialize progress bar
        with tqdm(range(self.start_epoch, self.config.trainer.epochs + 1)) as pbar:
            # run over n epochs
            for epoch in pbar:
                # check eval time
                start_time = time.time_ns()
                # perform an evaluation step
                eval_res = self._eval(epoch)
                eval_losses.append(eval_res['losses'])
                eval_accs.append(eval_res['accs'])
                eval_values.append(eval_res['values'])
                eval_time = (time.time_ns()-start_time)/(10**9)

                # check train time
                start_time = time.time_ns()
                # perform an training step
                train_res = self._train(epoch)
                train_losses.append(train_res['losses'])
                train_accs.append(train_res['accs'])
                train_values.append(train_res['values'])
                train_time = (time.time_ns()-start_time)/(10**9)

                # update the progress bar info
                pbar.set_postfix(train_loss=map_reduce(train_losses[-1], 'loss'), 
                                 eval_loss=map_reduce(eval_losses[-1], 'loss'),
                                 train_acc=map_reduce(train_accs[-1], 'acc'), 
                                 eval_acc=map_reduce(eval_accs[-1], 'acc'),
                                 eval_time=eval_time,
                                 train_time=train_time,
                                 refresh=False)

                # if a scheduler is used, perform scheduler step
                if self.scheduler:
                    # check if we are still improving on the source loss otherwise decrease LR
                    self.scheduler.step()

                # create checkpoints periodically
                if self.save_interval != 0 and epoch % self.save_interval == 0:
                    self.save_checkpoint(epoch)

                # summaries the collected stats
                collection = self.collectible.redux()
                # write to tensorboard
                self.writer.write(category='train', **collection)
                self.writer.write(category='eval', **collection)

                # update progress bar step
                pbar.update()
                # update tensorboard counter
                self.writer.step()
                # reset collected stats
                self.collectible.reset()

                # early_stopping needs the validation loss to check if it has decreased, 
                # and if it has, it will make a checkpoint of the current model
                self.early_stopping(epoch, map_reduce(eval_losses[-1], 'loss'))
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

        print(f"Train loss: {map_reduce(train_losses[-1], 'loss')} Eval loss: {map_reduce(eval_losses[-1], 'loss')}")
        print(f"Train accuracy: {map_reduce(train_accs[-1], 'acc')} Eval accuracy: {map_reduce(eval_accs[-1], 'acc')}")
        return {'train_losses': train_losses, 'eval_losses': eval_losses, 'train_accs': train_accs, 'eval_accs': eval_accs, 'train_values': train_values, 'eval_values': eval_values}


def experiments(config):
    """Main experiment entrance point for each `<approach>.py`"""
    warnings.simplefilter("ignore")

    seeds = config.seeds
    device = config.device
    lambda_list = config.approach.lambda_list

    # create the dataset and get the dataloader
    config.backbone = config.agg_backbone
    create_domain_adaptation_data = load_function(config.dataloader.module, config.dataloader.funcname)
    for train_loader, eval_loader in create_domain_adaptation_data(config):
        # dataset options
        dataloaders = (train_loader, eval_loader)
        ds_name = f"{train_loader.dataset.source_domain_name}-{train_loader.dataset.target_domain_name}"

        cls_results = {}
        iwv_results = {}
        cls_predictions = {}
        iwv_predictions = {}

        # run an experiment for multiple seeds
        for i, seed in enumerate(seeds):
            print(f'Running experiment with seed: {seed} Run: {i+1}/{len(seeds)}')
            set_seed(seed)
            config.seed = seed
            experiment_name = config.trainer.experiment_name
            # create method experiment config
            cfg = config.method #* use trainer from backbone config
            cfg.debug = config.debug
            cfg.device = device
            cfg.seed = seed
            seed = str(seed)
            cfg.checkpoint = config.checkpoint
            cfg.trainer.epochs = config.trainer.epochs_cls
            cfg.trainer.save_interval = config.trainer.save_interval

            # used for collecting data statistics
            collectible = BaseCollectible()
            experiment_id = f"{config.trainer.experiment_name}-backbone-method_{str(seed)}run-{ds_name}"
            # used to write to tensorboard
            tensorboard_dir = os.path.join(config.trainer.tensorboard_dir, experiment_id)
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = BaseWriter(log_dir=tensorboard_dir)

            # measure elapsed time per experiment
            start = time.time()
            id = 0
            # run over multiple lambdas
            for k, lamb in enumerate(lambda_list):
                key = f"{ds_name}-{lamb}"
                id += 1
                runid = f'{id}run'
                print(f'Current lambda {k+1}/{len(lambda_list)}: src to trg', key, 'Experiment id-pair', runid)

                # load a model architecture
                Net = load_function(cfg.model.module, cfg.model.classname)
                net = Net(cfg).to(device)
                cfg.trainer.experiment_name = f"{experiment_name}"
                cfg.approach.lambda_ = lamb
                # create a trainer instance and execute the approach
                trainer = Trainer(cfg, net, dataloaders, experiment_id=f'{config.trainer.experiment_name}-backbone-method_{runid}-{key}', writer=writer, collectible=collectible)
                if cfg.checkpoint:
                    trainer.resume_checkpoint(cfg.checkpoint)
                cls_res = trainer.run() # res: losses, accuracies, other values recorded during traing (e.g. alpha)
                trainer.init_best_model()
                cls_preds = trainer.val_cls_preds(include_da_preds=True) # preds: predictions on evaluation dataset

                # init dicts
                if key not in cls_results:
                    cls_results[key] = {}
                if seed not in cls_results[key]:
                    cls_results[key][seed] = []

                if key not in cls_predictions:
                    cls_predictions[key] = {}
                if seed not in cls_predictions[key]:
                    cls_predictions[key][seed] = []

                cls_results[key][seed].append(cls_res)
                cls_predictions[key][seed].append(cls_preds)

            # importance weighted validation classifier 
            config.model = config.iwv_model #* use trainer from agg_config
            if hasattr(config, 'iwv_backbone') and config.iwv_backbone:
                config.backbone = config.iwv_backbone
            IWVNet = load_function(config.model.module, config.model.classname)
            iwv_net = IWVNet(config).to(device)

            # used for collecting data statistics
            collectible = BaseCollectible()
            experiment_id = f"{config.trainer.experiment_name}-iwv-method_{str(seed)}run-{ds_name}"
            # used to write to tensorboard
            tensorboard_dir = os.path.join(config.trainer.tensorboard_dir, experiment_id)
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = BaseWriter(log_dir=tensorboard_dir)

            # set main criterion and metric
            config.trainer.criterion = config.trainer.iwv_criterion
            config.trainer.metric = config.trainer.iwv_metric
            # create trainer object
            iwv_trainer = Trainer(config, iwv_net, dataloaders, experiment_id=f'{config.trainer.experiment_name}-iwv-method_{str(seed)}run-{ds_name}', writer=writer, collectible=collectible)
            if 'use_random_matrix' not in config.model or not config.model.use_random_matrix:
                iwv_res = iwv_trainer.run()
            else:
                iwv_trainer.save_checkpoint(0)
                iwv_res = {'train_losses': [0], 'eval_losses': [0], 'train_accs': [0], 'eval_accs': [0], 'train_values': [0], 'eval_values': [0]}
            # predict iwv validation
            iwv_trainer.init_best_model()
            iwv_preds = iwv_trainer.val_iwv_preds()

            if ds_name not in iwv_predictions:
                    iwv_predictions[ds_name] = {}
            if seed not in iwv_predictions[ds_name]:
                iwv_predictions[ds_name][seed] = []
            
            if ds_name not in iwv_results:
                    iwv_results[ds_name] = {}
            if seed not in iwv_results[ds_name]:
                iwv_results[ds_name][seed] = []

            iwv_results[ds_name][seed].append(iwv_res)
            iwv_predictions[ds_name][seed].append(iwv_preds)

            # check elapsed time
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Task completed - time elapsed for task {} with seed {}> {:0>2}:{:0>2}:{:05.2f}".format(ds_name, seed, int(hours),int(minutes),seconds))

        # save the predictions for cls
        pred_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'cls_pred_dataset_{ds_name}.npz')
        np.savez(pred_file, cls_predictions)
        # save the predictions for iwv
        pred_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'iwv_pred_dataset_{ds_name}.npz')
        np.savez(pred_file, iwv_predictions)
        # save final / total cls_results
        cls_res_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'cls_results_dataset_{ds_name}.npz')
        np.savez(cls_res_file, cls_results)
        # save final / total iwv_results
        iwv_res_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'iwv_results_dataset_{ds_name}.npz')
        np.savez(iwv_res_file, iwv_results)

        # reset memory to avoid compounding dataset allocations
        if config.dataloader.reset_and_reload_memory:
            train_loader.dataset.reset_memory()
            eval_loader.dataset.reset_memory()
