#!/bin/sh

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=DDC
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=CMD 
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=DDC 
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=CMD 
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=DDC 
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=TRANSFORMED_MOONS backbone=MLP da_method=CMD 
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DDC
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=CMD
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=DDC
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=HAR da_method=CMD
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=DDC
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=EEG da_method=CMD
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=DANN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=Deep_Coral
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=CoDATS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=DDC
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=HoMM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=DIRT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=DSAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=AdvSKM
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=MMDA
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=CDAN
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python main.py -m seed=1,2,3 experiment_name=multirun run_description=da dataset=WISDM da_method=CMD
