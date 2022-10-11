import os
import hydra
import wandb
import omegaconf
import warnings
from pathlib import Path
from trainer import cross_domain_trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def make_wandb_tags(config):
    tags = []
    for key in ["experiment_name", "seed"]:
        if key in config:
            val = config[key]
            if key == "seed":
                val = "seed_" + str(val)
            tags.append(val)
    return tags


def setup_wandb(config):
    print("Setting up logging to Weights & Biases.")
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # hydra changes working directories
    log_dir = str(Path.joinpath(Path(os.getcwd()), config.run_params.log_dir))
    # make "wandb" path, otherwise WSL might block writing to dir
    wandb_path = Path.joinpath(Path(log_dir), "wandb")
    wandb_path.mkdir(exist_ok=True, parents=True)
    wandb.login()
    # tracks everything that TensorBoard tracks
    # writes to same dir as TensorBoard
    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb_run = wandb.init(project="aggregation_method-AdaTime", name=log_dir,
                           dir=log_dir, save_code=False, config=config_dict)
    wandb_run.tags = wandb_run.tags + tuple(make_wandb_tags(config_dict))
    print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
    return wandb_run


@hydra.main(config_path="configs", config_name="config")
def main(config):
    if config.use_wandb:
        wandb_run = setup_wandb(config)
    trainer = cross_domain_trainer(config)
    trainer.train()
    # necessary for Hydra multiruns
    wandb.finish()
    wandb.tensorboard.unpatch()


if __name__ == "__main__":
    main()
