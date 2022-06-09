#!/bin/sh


# AdaTime suite
echo "HHAR_SA"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir approach/adatime/output/multirun \
    --method ADATIME \
    --dataset HHAR_SA \
    --out_dir results/ADATIME \
    --rcond 1e-1

echo "EEG"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir approach/adatime/output/multirun \
    --method ADATIME \
    --dataset EEG \
    --out_dir results/ADATIME \
    --rcond 1e-1

echo "WISDM"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir approach/adatime/output/multirun \
    --method ADATIME \
    --dataset WISDM \
    --out_dir results/ADATIME \
    --rcond 1e-1

echo "HAR"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir approach/adatime/output/multirun \
    --method ADATIME \
    --dataset HAR \
    --out_dir results/ADATIME \
    --rcond 1e-1


# MiniDomainNet Reviews Dataset
echo "MINI_DOMAIN_NET"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir outputs/minidomainnet/ \
    --method BP \
    --dataset MINI_DOMAIN_NET \
    --out_dir results/BP \
    --seed_list 213564,11223,844585 \
    --skip_exists \
    --rcond 1e-1


# Amazon Reviews Dataset
echo "AMAZON_REVIEWS"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir outputs/amazon/ \
    --method BP \
    --dataset AMAZON_REVIEWS \
    --out_dir results/BP \
    --rcond 1e-1


# Moons Dataset
echo "MOONS"
PYTHONPATH=. python scripts/collect_exp.py \
    --base_dir outputs/moons/ \
    --method BP \
    --dataset MOONS \
    --out_dir results/BP \
    --rcond 1e-3
