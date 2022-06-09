# AdaTime: A Systematic Evaluation of Domain Adaptation Algorithms on Time Series Data

**AdaTime** is a PyTorch suite to systematically and fairly evaluate different domain adaptation methods on time series data.
 
## Requirmenets:
- Python3
- Pytorch==1.7
- Numpy==1.20.1
- scikit-learn==0.24.1
- Pandas==1.2.4
- skorch==0.10.0 (For DEV risk calculations)
- openpyxl==3.0.7 (for classification reports)
- Wandb=0.12.7 (for sweeps)

## Datasets

### Available Datasets
We used four public datasets in this study. We also provide the **preprocessed** versions as follows:
- [EGG](https://researchdata.ntu.edu.sg/privateurl.xhtml?token=9f854e11-4384-44d4-bad8-9d2894c76f07)
- [HAR](https://researchdata.ntu.edu.sg/privateurl.xhtml?token=ddaf52b4-37ef-4578-aaed-d9d4c8a942c0)
- [HHAR](https://researchdata.ntu.edu.sg/privateurl.xhtml?token=e44f10b6-e160-4e63-8fcf-8060aadbd3e5)
- [WISDM](https://researchdata.ntu.edu.sg/privateurl.xhtml?token=55e459de-c9d7-470f-8453-ad086c304f9d)

The datasets can be downloaded [here](https://drive.google.com/drive/folders/1P1wHoX1O8w2drG1EumkFsav4v5FSe_xL?usp=sharing).

### Adding New Dataset

#### Structure of data
To add new dataset (*e.g.,* NewData), it should be placed in a folder named: NewData in the datasets directory.

Since "NewData" has several domains, each domain should be split into train/test splits with naming style as
"train_*x*.pt" and "test_*x*.pt".

The structure of data files should in dictionary form as follows:
`train.pt = {"samples": data, "labels: labels}`, and similarly for `test.pt`.

#### Configurations
Next, you have to add a class with the name NewData in the `configs/data_model_configs.py` file. 
You can find similar classes for existing datasets as guidelines. 
Also, you have to specify the cross-domain scenarios in `self.scenarios` variable.

Last, you have to add another class with the name NewData in the `configs/hparams.py` file to specify
the training parameters.


## Domain Adaptation Algorithms
### Existing Algorithms
- [Deep Coral](https://arxiv.org/abs/1607.01719)
- [MMDA](https://arxiv.org/abs/1901.00282)
- [DANN](https://arxiv.org/abs/1505.07818)
- [CDAN](https://arxiv.org/abs/1705.10667)
- [DIRT-T](https://arxiv.org/abs/1802.08735)
- [DSAN](https://ieeexplore.ieee.org/document/9085896)
- [HoMM](https://arxiv.org/pdf/1912.11976.pdf)
- [DDC](https://arxiv.org/abs/1412.3474)
- [CoDATS](https://arxiv.org/pdf/2005.10996.pdf)
- [AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf)


### Adding New Algorithm
To add a new algorithm, place it in `algorithms/algorithms.py` file.


## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `experiment_description`.
- Each experiment could have different trials, each is specified by `run_description`.
- For example, if we want to experiment different UDA methods with CNN backbone, we can assign
`experiment_description=CNN_backnones run_description=DANN` and `experiment_description=CNN_backnones run_description=DDC` and so on.

### Training a model

To train a model:

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH= .python main.py  -m \
                                                    experiment_name=multirun \
                                                    run_description=da \
                                                    da_method=CDAN \
                                                    dataset=HHAR_SA \
                                                    seed=1,2,3
```

Upon the run, you will find the running progress in the specified project page in wandb.

`Note:` If you got cuda out of memory error during testing, this is probably due to DEV risk calculations


### Upper and Lower bounds
To obtain the source-only or the target-only results, you can run `same_domain_trainer.py` file.

## Results
- Each run will have all the cross-domain scenarios results in the format `src_to_trg_run_x`, where `x`
is the run_id (you can have multiple runs by assigning `num_runs` arg). 
- Under each directory, you will find the classification report, a log file, checkpoint, 
and the different risks scores.
- By the end of the all the runs, you will find the overall average and std results in the run directory.
