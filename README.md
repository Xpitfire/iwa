# Aggregation Method for Domain Adaptation

## Installing

1. Clone repository
```bash
git clone git@https://anonymous.4open.science/r/iwa-4C76
cd agg
```

2. Create a python 3 conda environment
```bash
conda env create -f environment.yml
```

3. Install package
```bash
pip install -e .
```

## Compute Results

1. Train domain adaptation method with aggregation method by calling the `agg` configs:
```bash
CUDA_VISIBLE_DEVICES=<device-id> PYTHONPATH=. python scripts/train.py -m config=configs/<your-agg-config>.json experiment_name=<name> seed=<seeds>
```
```bash
# running a CMD experiment with AGG
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train.py -m config=configs/config.twinmoons_agg_cmd.json experiment_name=twinmoons_agg_cmd seed=1,2,3,4,5
```


## References

* [Moment Matching for Multi-Source Domain Adaptation](http://ai.bu.edu/M3SDA/)
* [Amazon product data](https://jmcauley.ucsd.edu/data/amazon/)
* [Unsupervised Domain Adaptation by Backpropagation](https://github.com/fungtion/DANN)
* [Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation](https://github.com/thuml/Deep-Embedded-Validation)
* [The balancing principle for parameter choice in distance-regularized domain adaptation](https://github.com/Xpitfire/bpda)
* [AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data](https://github.com/emadeldeen24/AdaTime)

