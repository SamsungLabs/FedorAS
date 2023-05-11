
# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import os
import json
import glob
import copy
import pathlib
from PIL import Image
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import transforms as TTT
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url


from src.datasets.augmentations import RandAugment, Cutout
from src.datasets.h5_datasets import ShakespeareDataset
from src.datasets.speechcommands import get_speechcommands_and_partition_it, raw_audio_to_mfcc_transforms, PartitionedSPEECHCOMMANDS
from src.utils.independentrng import IndependentRng, get_rng_state, set_rng_state

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def npy_loader_func(path):
    with open(path, 'rb') as f:
        return np.load(f)

def one_minus(l):
    return 1 - l


class FLDatasetFolder(datasets.DatasetFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_class = { v: int(k) for k, v in self.class_to_idx.items() }

    def __getitem__(self, *args, **kwargs):
        im, lab = super().__getitem__(*args, **kwargs)
        lab = self.idx_to_class[lab]
        return im, lab


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.
    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Optional[Callable] = None,
        t_transform: Optional[Callable] = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform
        self.target_transform = t_transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.
    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def _partition_dataset(dataset, path_to_dataset, pool_size: int, alpha: float, num_classes: int, val_ratio: float, dirichlet_dist: np.ndarray = None, dump: bool = True):
    _rng_state = get_rng_state(inc_numpy=True, inc_torch=True)
    from flwr.dataset.utils.common import create_lda_partitions
    set_rng_state(_rng_state)

    rng = IndependentRng(seed=2022, inc_torch=True, inc_numpy=True)
    with rng.activate():
        # ! Flower code creates its own random generator, messing up
        # with rng... We opt to create the dirichtlet distribution outside
        # Flower code and use to generate LDA partitions
        if dirichlet_dist is None:
            dirichlet_dist = np.random.dirichlet(
                alpha=[alpha]*num_classes, size=pool_size
            )
        partitions, dirichlet_dist = create_lda_partitions(
            dataset, dirichlet_dist=dirichlet_dist,
            num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
        )
        partitions_, dirichlet_dist = create_lda_partitions(
            dataset, dirichlet_dist=dirichlet_dist,
            num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
        )

    partitions_named_list = []
    for p in range(pool_size):
        line_dict = {}
        labels = partitions[p][1]
        image_idx = partitions[p][0]
        train_idx = list(range(len(partitions[p][0]))) # unless val_ratio>0, all images id will be part of the training set for this client
        # imgs = images[image_idx]

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            # val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            # remaining images for training
            # imgs = imgs[train_idx]
            labels = labels[train_idx]

            # check to ensure val ids are not in train ids and viceversa
            idx = copy.deepcopy(train_idx)
            idx.extend(val_idx)
            assert len(idx) == len(set(idx)), "Sets are not disjoint!!"

        line_dict['client_id'] = p
        line_dict['train_ids'] = image_idx[train_idx]
        line_dict['train_labels'] = labels
        if val_ratio > 0.0:
            line_dict['val_ids'] = image_idx[val_idx]
            line_dict['val_labels'] = val_labels
        partitions_named_list.append(line_dict)

    if dump:
        filename = os.path.join(os.path.dirname(path_to_dataset), 'lda_partitioning.json')
        # os.makedirs(path_to_dataset)
        with open(filename, 'w') as f:
            json.dump(partitions_named_list, f, cls=NumpyEncoder)

    return partitions_named_list, dirichlet_dist


def construct_random_sampler(dataset, ratio_to_sample: float, sampler=None):
    """Constructs a randomsampler to be passed to a dataloader. If a sampler is already
    setup for the specific dataset, it will be re-weighted so it matches the specified
    ratio of elements allowed to be sampled."""

    if ratio_to_sample == 1.0:
        # no sampler needed
        return sampler

    num_samples = len(dataset)
    weights = [1] * int(num_samples*ratio_to_sample)
    weights.extend([0] * int(num_samples * (1.0-ratio_to_sample)))

    if sampler:
        print("TODO")
    else:
        sampler = WeightedRandomSampler(weights, num_samples=sum(weights), replacement=False)

    return sampler

class AbstractDataset(ABC):

    @staticmethod
    @abstractmethod
    def get(path_to_data):
        """
        Downloads dataset and generates a unified training set (it will
        be partitioned later using the LDA partitioning mechanism.
        """

        pass

    @staticmethod
    @abstractmethod
    def get_dataset(path_to_data, cid, partition, transforms):
        pass

    @staticmethod
    @abstractmethod
    def get_global_dataloader(path_to_data: Path, partition: str, transforms: transforms.Compose,
                              batch_size: int, num_workers: int, shuffle: bool,
                              **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_dataloader(path_to_data, partition_name, batch_size, workers, transforms, cid, **kwargs):
        """
        Generates trainset/valset object and returns appropiate dataloader.
        """

        pass

    @staticmethod
    def unify_validation_and_training(fed_dir, globaldata_dir):
        """
        Fuse all validation and training partitions into a single directory/file.
        """

        pass

    def do_fl_partitioning(path_to_dataset, pool_size, alpha, val_ratio=0.0):
        """
        Partition dataset. Use of LDA if non-federated or pre-split partitions if federated.
        """

        pass

    def partition_in_fs(path_to_dataset, lda_json, lda_alpha):
        """
        Given a dataset and a dictionary association, it organises the folders in the filesystem.
        """

        pass

    @staticmethod
    @abstractmethod
    def get_norm_params(client_id=None):

        pass

    @staticmethod
    @abstractmethod
    def get_train_transforms(norm_params):

        pass

    @staticmethod
    @abstractmethod
    def get_train_transforms_fl(norm_params):

        pass

    @staticmethod
    @abstractmethod
    def get_eval_transforms(norm_params):

        pass


class CIFAR(AbstractDataset):
    dataset_class = None
    dir_name = None

    @classmethod
    # this happens on the client and server side (deterministic)
    def get(cls, path_to_data="./data"):
        # download dataset and load train set
        train_set = cls.dataset_class(root=path_to_data, train=True, download=True)

        test_set = cls.dataset_class(root=path_to_data, train=False, download=True)

        # fuse all data splits into a single "training.pt"
        data_loc = Path(path_to_data) / cls.dir_name
        training_data = data_loc / "training.pt"
        if not training_data.exists():
            print("Generating unified CIFAR dataset")
            torch.save([train_set.data, np.array(train_set.targets)], training_data)

        test_data = data_loc / "federated" / "test.pt"
        if not test_data.exists():
            test_data.parent.mkdir(parents=True)
            torch.save([test_set.data, np.array(test_set.targets)], test_data)

        # returns path where training data is
        return training_data

    @classmethod
    def get_json_assoc_for_testset(cls, dirichlet_dist, num_clients: int, alpha: float, path_to_data="./data"):
        test_set = cls.dataset_class(root=path_to_data, train=False, download=True)
        labels = np.array(test_set.targets)
        idx = np.array(range(len(test_set.data)))
        dataset = [idx, labels]
        return _partition_dataset(dataset, path_to_data, num_clients, alpha, cls.num_classes, 0.0, dirichlet_dist, dump=False)

    @staticmethod
    def get_dataset(path_to_data: Path, cid: int, partition: str, transforms: list, num_classes: int):
        if cid is None:
            path_to_data = path_to_data / (partition + ".pt")
        else:
            # generate path to cid's data
            path_to_data = path_to_data / str(cid) / (partition + ".pt")

        return TorchVision_FL(path_to_data, transform=transforms)

    @classmethod
    def get_global_dataloader(cls, path_to_data: Path, partition: str, transforms: transforms.Compose, batch_size: int,
                              num_workers: int, shuffle: bool, num_classes, sampler_ratio: float, **kwargs):

        dataset = cls.get_dataset(path_to_data, cid=None, partition=partition, transforms=transforms, num_classes=num_classes)
        kwargs = {"num_workers": num_workers, "pin_memory": True, "drop_last": False}
        sampler = construct_random_sampler(dataset, sampler_ratio)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, sampler=sampler, persistent_workers=True, **kwargs)

    @classmethod
    def get_dataloader(
        cls, path_to_data: str, partition_name: str, batch_size: int, workers: int,
        transforms: list, cid: str, num_classes, **kwargs):

        """Generates trainset/valset object and returns appropiate dataloader."""

        shuffle = True if partition_name == "train" else False
        dataset = cls.get_dataset(Path(path_to_data), cid, partition_name, transforms, num_classes=num_classes)

        # we use as number of workers all the cpu cores assigned to this actor
        kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, persistent_workers=True, **kwargs)

    @staticmethod
    def unify_validation_and_training(fed_dir, globaldata_dir):
        sources = ['train.pt', 'val.pt']

        for s in sources:
            if (globaldata_dir/s).exists():
                continue

            unified_data = []
            unified_targets = []
            for p_path in fed_dir.iterdir():
                if p_path.is_dir():
                    data, targets = torch.load(p_path/s)
                    unified_data.append(data)
                    unified_targets.append(targets)

            torch.save([np.concatenate(unified_data), np.concatenate(unified_targets)], globaldata_dir/s)

    @classmethod
    def do_fl_partitioning(cls, path_to_dataset, pool_size, alpha, val_ratio=0.0):
        images, labels = torch.load(path_to_dataset)
        idx = np.array(range(len(images)))
        dataset = [idx, labels]
        # These partitions are in a list of tuples.
        # Each tuple element is one partition.
        # First tuple has all the sample ids
        # Second tuple has all the sample labels.

        return _partition_dataset(dataset, path_to_dataset, pool_size, alpha, cls.num_classes, val_ratio)

    @staticmethod
    def partition_in_fs(path_to_dataset, lda_json, lda_alpha):
        # Load dataset
        images, _ = torch.load(path_to_dataset)
        images_test, _ = torch.load(Path(path_to_dataset).parent/"federated" / "test.pt") #! caution changing this

        # now save partitioned dataset to disk
        # first delete dir containing splits (if exists), then create it
        splits_dir = path_to_dataset.parent / "federated" / str(lda_alpha) / str(len(lda_json))
        if splits_dir.exists():
            # shutil.rmtree(splits_dir)
            print("Partitions found in FS, skipping...")
            return splits_dir
        Path.mkdir(splits_dir, parents=True)

        pool_size = len(lda_json)
        for p in range(pool_size):
            Path.mkdir(splits_dir / str(p))

            train_ids = lda_json[p]["train_ids"]
            train_labels = lda_json[p]["train_labels"]
            train_imgs = images[train_ids]
            with open(splits_dir / str(p) / "train.pt", "wb") as f:
                torch.save([train_imgs, train_labels], f)

            val_ids = lda_json[p]["val_ids"]
            val_labels = lda_json[p]["val_labels"]
            val_imgs = images[val_ids]
            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            test_ids = lda_json[p]["test_ids"]
            test_labels = lda_json[p]["test_labels"]
            test_imgs = images_test[test_ids]
            with open(splits_dir / str(p) / "test.pt", "wb") as f:
                torch.save([test_imgs, test_labels], f)

        return splits_dir

    @classmethod
    def get_norm_params(cls, client_id=None):
        return cls.norm_params

    @staticmethod
    def get_train_transforms(norm_params):
        return transforms.Compose([
            getattr(transforms, 'RandAugment', RandAugment)(2, 9),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop((32,32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(*norm_params),
            Cutout(1, 16, 1.0)
        ])

    @staticmethod
    def get_train_transforms_fl(norm_params):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop((32,32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(*norm_params),
        ])

    @staticmethod
    def get_eval_transforms(norm_params):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*norm_params)
        ])

class CIFAR10(CIFAR):
    dataset_class = datasets.CIFAR10
    dir_name = "cifar-10-batches-py"
    num_classes = 10
    norm_params = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)

class CIFAR100(CIFAR):
    dataset_class = datasets.CIFAR100
    dir_name = "cifar-100-python"
    num_classes = 100
    norm_params = (0.5070588235294118, 0.48666666666666664, 0.4407843137254902), (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)

    @staticmethod
    def get_dataset(path_to_data: Path, cid: int, partition: str, transforms: list, num_classes: int):
        if cid is None:
            path_to_data = path_to_data / (partition + ".pt")
        else:
            # generate path to cid's data
            path_to_data = path_to_data / str(cid) / (partition + ".pt")

        if CIFAR100.num_classes == num_classes:
            t_transforms = None
        elif num_classes == 20:
            t_transforms = TTT.Compose([
                TTT.Lambda(CIFAR100.sparse2coarse)
            ])
        else:
            raise ValueError("Invalid number of classes. Choose between {} and {}.", CIFAR100.num_classes, 20)

        return TorchVision_FL(path_to_data, transform=transforms, t_transform=t_transforms)

    @staticmethod
    def sparse2coarse(targets):
        """Convert Pytorch CIFAR100 sparse targets to coarse targets. Taken from here:
        https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
        Usage:
            trainset = torchvision.datasets.CIFAR100(path)
            trainset.targets = sparse2coarse(trainset.targets)
        """
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                   18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        return coarse_labels[targets]


class SpeechCommands():

    @staticmethod
    def get(path_to_data="./data", version: int = 2):
        #! Note here I'm hardcoding v2 -- ideally this would be set via the specific datatset `.yaml`` config
        return get_speechcommands_and_partition_it(Path(path_to_data), version)

    @staticmethod
    def get_global_dataloader(path_to_data: Path, partition: str, transforms: transforms.Compose, num_classes: int,
                              batch_size: int, num_workers: int, shuffle: bool, sampler_ratio: float, **kwargs):

        # Speechcommands dataset in torchaudio has some hardcoded labels to differenciate between paritions
        # here we do some "translation"
        if partition == "train":
            subset = "training"
        elif partition == "val":
            subset = "validation"
        else:
            subset = "testing"

        #! Note here I'm hardcoding v2 -- ideally this would be set via the specific datatset `.yaml`` config
        version = 2
        if (num_classes == 35 and version == 2) or (num_classes == 30 and version == 1):
            classes = "all"
        elif num_classes == 12:
            classes = 'meh'
        else:
            raise NotImplementedError()

        dataset = PartitionedSPEECHCOMMANDS(path_to_data, subset=subset, transforms=transforms, classes=classes)

        sampler = None
        if subset == "training":
            sampler, _ = dataset.get_balanced_sampler()
        elif subset == "validation":
            sampler = construct_random_sampler(dataset, sampler_ratio)

        # ! Just stick to 0 workers... things are literally 100x faster
        kwargs = {"num_workers": num_workers, "pin_memory": True, "drop_last": False, 'persistent_workers': True if num_workers>0 else False}
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=dataset._collate_fn, sampler=sampler, **kwargs)
        return dataloader

    @staticmethod
    def get_norm_params(client_id=None):
        return None

    @staticmethod
    def get_train_transforms(norm_params):
        return raw_audio_to_mfcc_transforms()

    @staticmethod
    def get_train_transforms_fl(norm_params):
        return raw_audio_to_mfcc_transforms()

    @staticmethod
    def get_eval_transforms(norm_params):
        return raw_audio_to_mfcc_transforms()

    @staticmethod
    def get_dataloader(
        path_to_data: str, partition_name: str, batch_size: int, num_classes: int,
        workers: int, transforms: transforms.Compose, cid: str, **kwargs
    ):

        if partition_name != "train":
            #! We don't have validation splits for each client. Instead, we have clients left out for validation.
            return None
        else:

            #! Note here I'm hardcoding v2 -- ideally this would be set via the specific datatset `.yaml`` config
            version = 2
            if (num_classes == 35 and version == 2) or (num_classes == 30 and version == 1):
                classes = "all"
            elif num_classes == 12:
                classes = 'meh'
            else:
                raise NotImplementedError(num_classes)

            partition = "training"
            path_to_client_data = Path(path_to_data)/str(cid)
            dataset = PartitionedSPEECHCOMMANDS(path_to_client_data, subset=partition, transforms=transforms, classes=classes)
            sampler, _ = dataset.get_balanced_sampler()
            # ! Just stick to 0 workers... things are literally 100x faster
            kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False, 'persistent_workers': True if workers>0 else False}
            return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=dataset._collate_fn, sampler=sampler, **kwargs)

def extract_bz2(filename, path="."):
    import tarfile
    with tarfile.open(filename, "r:bz2") as tar:
        tar.extractall(path)


class Shakespeare(AbstractDataset):
    class FSTensorDataset(Dataset):
        def __init__(self, root, transform=None, target_transform=None):
            self.root = root
            root_dir = pathlib.Path(root)
            self.data_files_to_read = []
            self.label_files_to_read = []
            for f in root_dir.glob('*_data.npy'):
                l = f.with_name(f.name.replace('_data', '_labels'))
                assert self.__resolve_symlink(l).exists(), f"Does not exist: {l}"
                self.data_files_to_read.append(str(f.absolute()))
                self.label_files_to_read.append(str(l.absolute()))

            assert len(self.data_files_to_read) == len(self.label_files_to_read), \
                f"{len(self.data_files_to_read)} != {len(self.label_files_to_read)}"
            for f1, f2 in zip(self.data_files_to_read, self.label_files_to_read):
                assert (os.path.basename(f1).split("_")[0] == os.path.basename(f2).split("_")[0])
            self.num_data = len(self.data_files_to_read)

            self.transform = transform
            self.target_transform = target_transform

        def __resolve_symlink(self, path: Path):
            return Path(os.readlink(path)) if os.path.islink(path) else path

        def __len__(self):
            return self.num_data

        def __getitem__(self, idx):
            data_path = self.__resolve_symlink(self.data_files_to_read[idx])
            label_path = self.__resolve_symlink(self.label_files_to_read[idx])
            x = npy_loader_func(data_path)
            y = npy_loader_func(label_path)

            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)

            return x, y

    @staticmethod
    def get(path_to_data):
        dataset_src = 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2'
        data_loc = Path(path_to_data) / "shakespeare.tar.bz2"
        target_extract_loc = Path(path_to_data) / 'shakespeare'
        if not data_loc.exists():
            print("Downloading Shakespeare dataset")
            download_url(dataset_src, path_to_data)
        for f in ["shakespeare_train.h5", "shakespeare_test.h5"]:
            ff = target_extract_loc / f
            if not ff.exists():
                extract_bz2(data_loc, target_extract_loc)  # Creates shakespeare_{train,test}.h5

        return target_extract_loc

    @staticmethod
    def _get_h5_dataset(path_to_data, cid, partition, transforms):
        train = partition == 'train'
        return ShakespeareDataset(path_to_data, train=train, client_id=cid, transforms=transforms)

    @staticmethod
    def get_dataset(path_to_data: Path, cid: int, partition: str, transforms: list):
        if cid is None:
            path = Path(path_to_data)/partition # the global partition
        else:
            path = Path(path_to_data)/str(cid)/partition # the client partition

        t_transforms = None #labels_to_one_hot

        return Shakespeare.FSTensorDataset(root=path, transform=transforms, target_transform=t_transforms)

    @staticmethod
    def do_fl_partitioning(path_to_dataset, pool_size, alpha, val_ratio=0.0):
        import pandas as pd
        import os
        Shakespeare.get(os.path.dirname(path_to_dataset))
        filename = os.path.join(path_to_dataset, 'lda_partitioning.json')
        if Path(filename).is_file():
            print(f"{filename} already exists. Skipping fl partitioning ...")
            with open(filename, 'r') as f:
                to_return = json.load(f)
        else:
            dataset = Shakespeare._get_h5_dataset(Path(path_to_dataset), cid=None, partition="train", transforms=transforms.Compose([]))

            # clients = []
            # for cid, num_data in dataset.clients_num_data.items():
            #     for i in range(num_data):
            #         clients.append((cid, i))
            client_and_indices_df = pd.DataFrame(dataset.client_and_indices, columns=['cid', 'ordinal'])

            val_df = client_and_indices_df.groupby('cid').sample(frac=val_ratio)
            val_df['partition'] = 'val'
            train_df = client_and_indices_df[~client_and_indices_df.index.isin(val_df.index)]
            train_df['partition'] = 'train'
            all_df = pd.concat([train_df, val_df])
            to_return = []

            for idx, cid_group in all_df.groupby(['cid']):
                dict_line = {}
                dict_line['client_id'] = idx
                for jdx, partition_group in cid_group.groupby('partition'):
                    dict_line[f"{jdx}_ids"] = partition_group['ordinal'].tolist()
                for p in ['train', "val"]:
                    if dict_line.get(f"{p}_ids", None) is None:
                        dict_line[f"{p}_ids"] = []
                to_return.append(dict_line)

            with open(filename, 'w') as f:
                json.dump(to_return, f, cls=NumpyEncoder)

        return to_return

    @staticmethod
    def partition_in_fs(path_to_dataset, lda_json, lda_alpha):
        Shakespeare.get(os.path.dirname(path_to_dataset))

        splits_dir = Path(path_to_dataset) / "federated" / str(len(lda_json) - 1)
        if splits_dir.exists():
            print("Partitions found in FS, skipping...")
            return splits_dir.parent

        print("Partitioning dataset in FS")

        indexed_lda_json = {}
        for line in lda_json:
            indexed_lda_json[line['client_id']] = {
                "train_ids": set(line['train_ids']),
                "val_ids": set(line['val_ids'])
            }

        for partition in ('train', 'test'):
            dataset = Shakespeare._get_h5_dataset(path_to_dataset, cid=None, partition=partition, transforms=transforms.Compose([]))
            x = dataset[0]  # dataset is lazily populated, so I do this to be abel to access it later without hitting None.

            for i, (client_id, idx) in enumerate(dataset.client_and_indices):
                l_partition = partition
                if partition != 'test':
                    if idx in indexed_lda_json[client_id]["train_ids"]:
                        l_partition = 'train'
                    elif idx in indexed_lda_json[client_id]["val_ids"]:
                        l_partition = 'val'
                    else:
                        raise ValueError("Unknown id for training")

                # create directory for client_id/{train,test,val}
                prefix = Path(path_to_dataset) / "federated" / f"{client_id}" / f"{l_partition}"
                x = dataset.data[client_id][0][idx]
                y = dataset.data[client_id][1][idx]
                local_filename = os.path.join(prefix, f"{idx}")
                os.makedirs(os.path.dirname(local_filename), exist_ok=True)
                for suffix, data in {"data": x, "labels": y}.items():
                    local_filename_full = local_filename + f'_{suffix}.npy'
                    if i % 1000 == 0:
                        print(f"Persisting for client_id={client_id}, idx={idx} in {local_filename_full}")
                    with open(local_filename_full, 'wb') as f:
                        np.save(f, data.detach().numpy())

        return Path(path_to_dataset) / "federated"

    @staticmethod
    def unify_validation_and_training(fed_dir, globaldata_dir):
        print("Unifying dataset")
        for partition in ['train', 'test', 'val']:
            partition_dir = Path(globaldata_dir) / partition
            if partition_dir.exists():
                print(f"{partition_dir} exists, skipping {partition} symlinking ...")
                continue
            files = glob.glob(f"{fed_dir}/*/{partition}/*")
            print(f"Creating files for {partition} in {partition_dir}")
            for i, f in enumerate(files):
                tokens = f.split('/')
                cid = tokens[-3]
                filename = tokens[-1]
                dst_filename = Path(globaldata_dir) / partition / f"{cid}_{filename}"
                os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
                if i % 1000 == 0:
                    print(f"Symlinking {f} to {dst_filename}")
                os.symlink(f, dst_filename)

    @staticmethod
    def get_dataloader(path_to_data, partition_name, batch_size,
                       workers, transforms, cid, **kwargs):

        shuffle = True if partition_name == "train" else False

        dataset = Shakespeare.get_dataset(path_to_data, cid, partition_name, transforms)

        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle, collate_fn=Shakespeare.collate)

    @staticmethod
    def collate(items):
        xs, ys = zip(*items)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.concatenate(ys))

    @staticmethod
    def get_global_dataloader(path_to_data: Path, partition: str, transforms: transforms.Compose, batch_size: int,
                              num_workers: int, shuffle: bool, sampler_ratio: float, **kwargs):

        dataset = Shakespeare.get_dataset(path_to_data, cid=None, partition=partition, transforms=transforms)
        kwargs = {"num_workers": num_workers, "pin_memory": True, "drop_last": False, 'persistent_workers': True if num_workers>0 else False}

        sampler = construct_random_sampler(dataset, sampler_ratio)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=Shakespeare.collate, **kwargs)

        return dataloader

    @staticmethod
    def _get_h5_dataloader(path_to_data, is_train, batch_size, workers, transforms, cid):
        partition = "train" if is_train else "val"
        shuffle = True if partition == "train" else False

        dataset = Shakespeare._get_h5_dataset(path_to_data, cid, partition, transforms)

        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle, persistent_workers= True if workers>0 else False)

    @staticmethod
    def get_norm_params(client_id=None):
        return None
