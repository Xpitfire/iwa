from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    gaussian_prefix: str = 'gaussian-cs_agg'
    domains = ['gauss_src-gauss_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str = ''):
        super().__init__(base_dir, da_method, ResultsLoader.gaussian_prefix, suffix)

