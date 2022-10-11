from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    minidomainnet_prefix: str = 'minidomainnet-cs_agg'
    domains: List[str] = ['real_src-sketch_tgt',
           'real_src-painting_tgt',
           'real_src-quickdraw_tgt',
           'real_src-infograph_tgt',
           'real_src-clipart_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str = ''):
        super().__init__(base_dir, da_method, ResultsLoader.minidomainnet_prefix, suffix)

