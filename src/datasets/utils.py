# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import pickle
from pathlib import Path
import collections.abc as cabc

from src.datasets.wrappers import CIFAR10, CIFAR100, Shakespeare, SpeechCommands


def get_dataset_class_from_string(dataset: str):
    if dataset == "cifar10":
        return CIFAR10
    elif dataset == "cifar100":
        return CIFAR100
    elif dataset=="speechcommands":
        return SpeechCommands
    elif dataset == 'shakespeare':
        return Shakespeare
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

def prepare_dataset(dataset, data_path, exp_dir: Path = None, number_of_clients=100, lda_alpha=1.0, val_ratio=0.1):

    if dataset == "cifar10" or dataset == "cifar100":
        dataset_cls = get_dataset_class_from_string(dataset)
        # Download CIFAR10 (testset can be the global testset)
        train_path = dataset_cls.get(data_path)

        # partition dataset (use a large `alpha` to make it IID;
        # a small value (e.g. 1) will make it non-IID)
        # This will create a new directory called "federated: in the directory where
        # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
        # its own train/set split.
        json_assoc, dirichlet_dist = dataset_cls.do_fl_partitioning(
            train_path, pool_size=number_of_clients, alpha=lda_alpha, val_ratio=val_ratio
        )

        # Now we follow the dirichlet_dist to append to `json_assoc_path` how to
        # partition the testset
        testset_json, _ = dataset_cls.get_json_assoc_for_testset(dirichlet_dist, number_of_clients, lda_alpha, data_path)

        # add to `json_assoc` test partition data
        for cidx, test_pp in enumerate(testset_json):
            idx = cidx
            if json_assoc[cidx]['client_id'] != test_pp['client_id']:
                # if not in same order, find client in json_assoc_path w/ id test_pp['client_id']
                for i, client_data in enumerate(json_assoc):
                    if client_data['client_id'] == test_pp['client_id']:
                        idx = i
            # add test partition entries
            json_assoc[idx]['test_ids'] = test_pp['train_ids']
            json_assoc[idx]['test_labels'] = test_pp['train_labels']

        dataset_cls = get_dataset_class_from_string(dataset)
        train_path = dataset_cls.get(data_path)
        fed_dir = dataset_cls.partition_in_fs(train_path, json_assoc, lda_alpha)
        globaldata_dir = fed_dir.parent.parent
        dataset_cls.unify_validation_and_training(fed_dir, globaldata_dir)



    elif dataset == "speechcommands":
        _ = SpeechCommands.get(data_path)
        json_assoc = None # SpeechCommands is partitioned by speaker id, so no need to do any of this

        fed_dir = SpeechCommands.get(data_path)
        globaldata_dir = fed_dir.parent

    elif dataset == "shakespeare":
        train_path = Shakespeare.get(data_path)
        json_assoc = Shakespeare.do_fl_partitioning(train_path, pool_size=number_of_clients,
                                                     alpha=lda_alpha, val_ratio=val_ratio)

        path_to_dataset = Shakespeare.get(data_path)
        fed_dir = Shakespeare.partition_in_fs(path_to_dataset, json_assoc, lda_alpha)
        globaldata_dir = fed_dir.parent
        Shakespeare.unify_validation_and_training(fed_dir, globaldata_dir)

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    # save client-data association to experiment directory
    if json_assoc is not None and exp_dir is not None:
        with open(exp_dir/"data_client_association.pkl", 'wb') as h:
            pickle.dump(json_assoc, h, pickle.HIGHEST_PROTOCOL)

    # `fed_dir` points to the path where all the partitions are
    # `globaldata_dir` points to the path where the unified training and validation set files (.pt) are
    return fed_dir, globaldata_dir


def load_dataset_per_client(datasetclass, data_path, cid, batch_size, num_workers, num_classes,
                            train_included=True, transforms=None):
    testtransforms = transforms
    if isinstance(transforms, cabc.Sequence):
        transforms, testtransforms = transforms
    trainloader = None
    if train_included:
        trainloader = datasetclass.get_dataloader(
                data_path,
                partition_name="train",
                batch_size=batch_size,
                num_classes=num_classes,
                workers=num_workers,
                transforms=transforms,
                cid=cid,
            )

    testloader = datasetclass.get_dataloader(
                data_path,
                partition_name="val",
                batch_size=batch_size,
                num_classes=num_classes,
                workers=num_workers,
                transforms=testtransforms,
                cid=cid,
            )

    return trainloader, testloader

def load_global_dataset(datasetclass, globaldata_path, batch_size, num_classes, num_workers, transforms, include_test: bool = False,
                        val_sample_ratio: float=1.0):
    testtransforms = transforms
    if isinstance(transforms, cabc.Sequence):
        transforms, testtransforms = transforms

    globaldata_path = Path(globaldata_path)
    train_loader = datasetclass.get_global_dataloader(globaldata_path, partition='train', transforms=transforms, batch_size=batch_size,
                                                      num_classes=num_classes, num_workers=num_workers, shuffle=True, sampler_ratio=1.0)
    val_loader = datasetclass.get_global_dataloader(globaldata_path, partition='val', transforms=testtransforms, batch_size=batch_size,
                                                    num_classes=num_classes, num_workers=num_workers, shuffle=False, sampler_ratio=val_sample_ratio)

    test_loader = None
    if include_test:
        test_loader = datasetclass.get_global_dataloader(globaldata_path, partition='test', transforms=testtransforms,
                                                         batch_size=batch_size, num_classes=num_classes, num_workers=num_workers,
                                                         shuffle=False, sampler_ratio=1.0)

    return train_loader, val_loader, test_loader


def get_transforms(dataset_cls, cid):
    norm_params = dataset_cls.get_norm_params(cid)
    return [dataset_cls.get_train_transforms(norm_params), dataset_cls.get_eval_transforms(norm_params)]