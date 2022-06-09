import zipfile
import os
import hydra
import wandb
import json
import torch
import omegaconf
from pathlib import Path
from argparse import ArgumentParser
from shutil import copyfile
from misc.helpers import load_function, load_seed_list
import torch
from hydra.utils import get_original_cwd
torch.multiprocessing.set_sharing_strategy('file_system')


def snapshot_code(path, fname, ignore=["data", "tmp", "runs"]):
    """Makes a snapshot of the code as a zip file to a target `fname` destination"""
    zipf = zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        # ignore temporary folders
        if any(i in root for i in ignore):
            continue
        for file in files:
            # only consider .py and .json files
            if file.endswith(".py") or file.endswith(".json"):
                zipf.write(os.path.join(root, file))
    zipf.close()


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser()
    parser.add_argument('--config', help='path to config file', type=str, required=True)
    parser.add_argument('--checkpoint', help='path of the checkpoint file to resume training', type=str)
    return parser.parse_args()


def run_experiments(config):
    """Loads the approach code and executes the experiments function"""
    approach = load_function(config.approach.module, 'experiments')
    approach(config)


def check_or_create_dirs():
    """Creates the basic temporary folder structure for the projects"""
    os.makedirs('tmp', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('data', exist_ok=True)


def make_wandb_tags(config):
    tags = []
    for key in ["experiment_name", "seed"]:
        if key in config:
            val = config[key]
            if key == "seed":
                val = "seed_" + str(val)
            tags.append(val)
    return tags


def setup_wandb(args):
    print("Setting up logging to Weights & Biases.")
    config_dict = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # hydra changes working directories
    log_dir = str(Path.joinpath(Path(os.getcwd()), args.run_params.log_dir))
    # make "wandb" path, otherwise WSL might block writing to dir
    wandb_path = Path.joinpath(Path(log_dir), "wandb")
    wandb_path.mkdir(exist_ok=True, parents=True)
    wandb.login()
    # tracks everything that TensorBoard tracks
    # writes to same dir as TensorBoard
    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb_run = wandb.init(project="aggregation_method", name=log_dir,
                           dir=log_dir, save_code=False, config=config_dict)
    wandb_run.tags = wandb_run.tags + tuple(make_wandb_tags(config_dict))
    print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
    return wandb_run


def rec_assign_values(args, config, key, value, chain=[]):
    if key not in config:
        path_key = key if len(chain) == 1 else ".".join(chain)
        print(f'WARNING OVERRIDE FAILED: > {path_key} < not in config')
        return
    chain.append(key)
    if isinstance(value, omegaconf.dictconfig.DictConfig):
        for k, v in args[key].items():
            rec_assign_values(args[key], config[key], k, v, chain)
    else:
        path_key = key if len(chain) == 1 else ".".join(chain)
        print(f'>>> OVERRIDE CONFIG: {path_key} >>> from [{config[key]}] to [{value}]')
        config[key] = value


@hydra.main(config_path="settings", config_name="args")
def main(args):
    """Main entrance point for training"""
    setup_wandb(args)
    check_or_create_dirs()
    # copy current config file to new experiment folder
    src = os.path.join(get_original_cwd())
    dst = 'configs'
    #! Opening JSON file and copy over all necessary configs to correct for the hydra path changes
    with open(os.path.join(src, args.config)) as config_file:        
        # returns JSON object as a dictionary
        data = json.load(config_file)
    parsed_import_config = data['method']['config'][len('import::'):]
    with open(os.path.join(src, parsed_import_config)) as config_file:
        parse_data = json.load(config_file)
    if not os.path.exists(dst):
        os.makedirs(dst)
    copyfile(os.path.join(src, args.config), os.path.join(os.getcwd(), args.config))
    copyfile(os.path.join(src, parsed_import_config), os.path.join(os.getcwd(), parsed_import_config))
    # # create JSON config
    config = omegaconf.OmegaConf.create(data)
    config.method = omegaconf.OmegaConf.create(parse_data)
    config.trainer.experiment_name = args.experiment_name
    dst = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name)
    if not os.path.exists(dst):
        os.makedirs(dst)
    # copy code backup to experiments dir
    snapshot_code(get_original_cwd(), os.path.join(dst, config.trainer.code_backup_filename))
    # set seed if set by args
    config.seeds = [args.seed]
    config.device = "cuda:0"
    # override by hydra config the default config:
    for key, value in args.items():
        if key in ['experiment_name', 'config', 'run_params']: continue # ignore these keys
        rec_assign_values(args, config, key, value)
    print(f"Using device: {config.device}")
    # load seeding list
    if "seed_list" in config and config.seed_list is not None:
        config.seeds = load_seed_list('config.seed_list')
        print("Using config seeds", config.seeds)
    if 'checkpoint' not in config:
        config.checkpoint = False
    # save corrent experiment config
    omegaconf.OmegaConf.save(config=config, f='configs/run_config_dump.yaml')
    # run main experiment
    run_experiments(config)
    # necessary for Hydra multiruns
    wandb.finish()
    wandb.tensorboard.unpatch()


if __name__ == "__main__":
    main()
