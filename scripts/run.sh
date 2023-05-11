#!/bin/bash

python main.py configs/datasets/cifar10.yaml dataset.lda_alpha=1000
python main.py configs/datasets/cifar10.yaml dataset.lda_alpha=1.0
python main.py configs/datasets/cifar10.yaml dataset.lda_alpha=0.1

python main.py configs/datasets/cifar100.yaml
python main.py configs/datasets/cifar100_multitask.yaml

python main.py configs/datasets/shakespeare.yaml
python main.py configs/datasets/speechcommands.yaml

