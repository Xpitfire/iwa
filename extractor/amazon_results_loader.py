from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from extractor.loader import Loader


class ResultsLoader(Loader):
    amazon_prefix: str = 'amazon-cs_agg'
    domains = ['books-dvd',
        'books-electronics',
        'books-kitchen',
        'dvd-books',
        'dvd-electronics',
        'dvd-kitchen',
        'electronics-books',
        'electronics-dvd',
        'electronics-kitchen',
        'kitchen-books',
        'kitchen-dvd',
        'kitchen-electronics']

    def __init__(self, base_dir: str, da_method: str, suffix: str=''):
        super().__init__(base_dir, da_method, ResultsLoader.amazon_prefix, suffix)

