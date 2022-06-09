from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    minidomainnet_prefix: str = 'minidomainnet-cs_agg'
    domains: List[str] = ['painting-real-sketch-clipart-infograph-quickdraw',
           'painting-quickdraw-sketch-clipart-infograph-real',
           'painting-quickdraw-real-clipart-infograph-sketch',
           'painting-quickdraw-real-sketch-infograph-clipart',
           'painting-quickdraw-real-sketch-clipart-infograph',
           'quickdraw-real-sketch-clipart-infograph-painting']

    def __init__(self, base_dir: str, da_method: str, suffix: str = ''):
        super().__init__(base_dir, da_method, ResultsLoader.minidomainnet_prefix, suffix)

