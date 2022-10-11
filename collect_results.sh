#!/bin/sh

echo "AMAZON_REVIEWS"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/AMAZON_REVIEWS/ \
    --method ADATIME \
    --dataset AMAZON_REVIEWS \
    --out_dir results_iclr/AMAZON_REVIEWS/processed \
    --rcond 1e-1

echo "MOONS"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/MOONS/ \
    --method ADATIME \
    --dataset MOONS \
    --out_dir results_iclr/MOONS/processed \
    --rcond 1e-1

echo "MINI_DOMAIN_NET"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/MINI_DOMAIN_NET/ \
    --method ADATIME \
    --dataset MINI_DOMAIN_NET \
    --out_dir results_iclr/MINI_DOMAIN_NET/processed \
    --rcond 1e-1

echo "HAR"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/HAR/ \
    --method ADATIME \
    --dataset HAR \
    --out_dir results_iclr/HAR/processed \
    --rcond 1e-1

echo "EEG"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/EEG/ \
    --method ADATIME \
    --dataset EEG \
    --out_dir results_iclr/EEG/processed \
    --rcond 1e-1

echo "HHAR_SA"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/HHAR_SA/ \
    --method ADATIME \
    --dataset HHAR_SA \
    --out_dir results_iclr/HHAR_SA/processed \
    --rcond 1e-1

echo "WISDM"
PYTHONPATH=. python scripts/collect_new_exp.py \
    --base_dir results_iclr/WISDM/ \
    --method ADATIME \
    --dataset WISDM \
    --out_dir results_iclr/WISDM/processed \
    --rcond 1e-1

