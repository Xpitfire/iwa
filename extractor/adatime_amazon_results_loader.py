from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    amazon_prefix: str = 'amazon-cs_agg'
    domains = ['books_src-dvd_tgt',
        'books_src-electronics_tgt',
        'books_src-kitchen_tgt',
        'dvd_src-books_tgt',
        'dvd_src-electronics_tgt',
        'dvd_src-kitchen_tgt',
        'electronics_src-books_tgt',
        'electronics_src-dvd_tgt',
        'electronics_src-kitchen_tgt',
        'kitchen_src-books_tgt',
        'kitchen_src-dvd_tgt',
        'kitchen_src-electronics_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str=''):
        super().__init__(base_dir, da_method, ResultsLoader.amazon_prefix, suffix)

