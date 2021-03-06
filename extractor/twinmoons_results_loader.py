from extractor.loader import Loader


class ResultsLoader(Loader):
    twinmoons_prefix: str = 'twinmoons-cs_agg'
    domains = ['moons_src-moons_tgt']

    def __init__(self, base_dir: str, da_method: str, suffix: str = ''):
        super().__init__(base_dir, da_method, ResultsLoader.twinmoons_prefix, suffix)

