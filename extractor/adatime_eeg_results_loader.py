from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    adatime_hhar_sa_prefix: str = 'adatime_agg'
    domains: List[str] = ['0_src-11_tgt', '7_src-18_tgt', '9_src-14_tgt', '12_src-5_tgt', '16_src-1_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str=''):
        super().__init__(base_dir, da_method, ResultsLoader.adatime_hhar_sa_prefix, suffix)
