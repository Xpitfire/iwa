from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    adatime_hhar_sa_prefix: str = 'adatime_agg'
    domains: List[str] = ['0_src-6_tgt', '1_src-6_tgt', '2_src-7_tgt', '3_src-8_tgt', '4_src-5_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str=''):
        super().__init__(base_dir, da_method, ResultsLoader.adatime_hhar_sa_prefix, suffix)
