#!/bin/bash

# Nota that since `normal_fl.py` triggers Stage-III of FedorAS, it requires information about
# clients-to-tiers assignements. This is generated everytime a new FedorAS experiment is started (i.e. at the
# beginning of Stage-I). If you run the command below, it assumed that you aready have run such experiment (i.e.
# a cifar100 experiment directory has been created and `clusters_info.pkl` resides in it).

# The output directory (including TensorBoard) of this experiment will be <exp_dir>/end2end/Tier0/<date&time_you_run_this>

# Here we use the same number of rounds, clients, alpha, clients_per_round and batchsize as it was used in Reddi et al (ICLR'21)
python normal_fl.py configs/datasets/cifar100.yaml \
                    finetune.num_rounds_end2end=4000 \
                    finetune.strategy.clients_per_round=10 \
                    finetuneclient.type.client_cfg.batch_size=20 \
                    finetuneclient.type.client_cfg.optim.type.lr=0.1 \
                    --decision "1,2,8,1,5,3,5,3,8,0,6,4,5,8,1,1" \
                    --end2end \
                    --exp_dir experiments/<path_to_cifar100_500clients_lda0.1_experiment>/ # path to a previous c100 experiment which should contain a pickle w/ info about client-to-tiers
