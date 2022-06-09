from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    adatime_hhar_sa_prefix: str = 'adatime_agg'
    domains: List[str] = ['6_src-19_tgt', '7_src-18_tgt', '18_src-23_tgt', '20_src-30_tgt', '35_src-31_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str=''):
        super().__init__(base_dir, da_method, ResultsLoader.adatime_hhar_sa_prefix, suffix)
