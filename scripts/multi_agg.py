import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
from argparse import ArgumentParser
from approaches.ensemble_trainer import EnsembleTrainer


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser()
    parser.add_argument('--device',
                        help='set device for matrix computations',
                        type=str, default='')
    parser.add_argument('--base_dir',
                        help='base directory path to results',
                        type=str)
    parser.add_argument('--da_method',
                        help='domain adaptation method [cmd, mmd, dann,...]',
                        type=str)
    parser.add_argument('--rcond',
                        help='rcond parameter used in Aggregation for np.linalg.pinv()',
                        type=float,
                        default=1e-2)
    parser.add_argument('--suffix',
                        help='directory path suffix',
                        type=str,
                        default='')
    parser.add_argument(
        '--extractor',
        help='module.file_name of dataset for result extraction',
        type=str)
    parser.add_argument(
        '--seed_list',
        help='Path to seed file or list of seeds separated by ,',
        type=str)
    return parser.parse_args()


def check_dirs(base_dir):
    """Creates the basic temporary folder structure for the projects"""
    assert os.path.exists(base_dir)


def load_seed_list(file_name='seed_list.txt'):
    """Loads seeds from a file"""
    with open(file_name, 'r') as f:
        seeds = f.read().split('\n')
    seeds = [int(s) for s in seeds if s is not None and s != '']
    return seeds


def run():
    """Main entrance point for extracting results.
    """
    options = parse_args()
    check_dirs(options.base_dir)
    base_dir = Path(options.base_dir)

    # load seeding list
    if hasattr(options, 'seed_list') and options.seed_list:
        seeds = options.seed_list.split(',')
        print("Using args seeds", seeds)
    else:
        seeds = load_seed_list()
        print("Using file loaded seeds", seeds)

    da_method = options.da_method
    extractor = options.extractor
    seed_list = seeds
    rcond = options.rcond
    suffix = options.suffix

    ensemble_trainer = EnsembleTrainer(base_dir, da_method, extractor, seed_list, rcond, suffix)
    ensemble_trainer.run()


if __name__ == "__main__":
    run()
