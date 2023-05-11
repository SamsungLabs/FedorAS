# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import os

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FLDataset(Dataset):
    def get_num_clients(self):
        return self.num_clients

    def set_client(self, index=None):
        raise NotImplementedError


class H5TFFDataset(FLDataset):
    def __init__(self, h5_path, client_id, data_key):
        self.h5_path = h5_path
        self.dataset = None
        self.clients = list()
        self.clients_num_data = dict()
        self.client_and_indices = list()
        # self.indices_to_orig_key = dict()
        with h5py.File(self.h5_path, 'r') as file:
            data = file['examples']
            for client in list(data.keys()):
                self.clients.append(client)
                n_data = len(data[client][data_key])
                for i in range(n_data):
                    self.client_and_indices.append((client, i))
                self.clients_num_data[client] = n_data
        self.num_clients = len(self.clients)
        self.length = len(self.client_and_indices)

        self.set_client(client_id)

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            self.length = len(self.client_and_indices)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.clients_num_data[self.clients[index]]

    def _get_item_preprocess(self, index):
        # loading in getitem allows us to use multiple processes for data loading
        # because hdf5 files aren't pickelable so can't transfer them across processes
        # https://discuss.pytorch.org/t/hdf5-a-data-format-for-pytorch/40379
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')["examples"]
        if self.client_id is None:
            client, i = self.client_and_indices[index]
        else:
            client, i = self.clients[self.client_id], index
        return client, i

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.length

SHAKESPEARE_VOCAB = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
SHAKESPEARE_EVAL_BATCH_SIZE = 4

class ShakespeareDataset(FLDataset):
    def __init__(self, data_path, train=True,
                 batch_size=SHAKESPEARE_EVAL_BATCH_SIZE,
                 client_id=None, transforms=None):
        self.train = train
        if train:
            data_path = os.path.join(data_path, 'shakespeare_train.h5')
        else:
            data_path = os.path.join(data_path, 'shakespeare_test.h5')
        self.batch_size = batch_size
        self.dataset = ShakespeareH5(data_path)
        self.dummy_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.num_clients = self.dataset.num_clients
        self.train = train

        self.available_clients = list()
        self.data = dict()
        self.clients_num_data = dict()
        self.client_and_indices = list()

        if train:
            self._add_client_train(client_id)
        else:
            self._add_client_test(client_id)
            # self._add_test()
        self.set_client(client_id)

    def _add_client_train(self, client_id):
        client_ids = range(self.num_clients) if client_id is None else [client_id]
        for cid in client_ids:
            if cid in self.available_clients:
                continue
            self.dataset.set_client(cid)
            x_data = torch.cat([x[0] for x, y in self.dummy_loader], dim=0)
            y_data = torch.cat([y[0] for x, y in self.dummy_loader], dim=0)
            self._update_data(cid, x_data, y_data)

    def _add_client_test(self, client_id):
        client_ids = range(self.num_clients) if client_id is None else [client_id]
        if client_ids is None:
            self._add_test()
        else:
            self._add_client_train(client_id)

    def _add_test(self):
        self.dataset.set_client(None)
        x_data = torch.cat([x[0] for x, y in self.dummy_loader], dim=0)
        y_data = torch.cat([y[0] for x, y in self.dummy_loader], dim=0)
        # reorder data  such that consequent batches follow speech order
        n_zeros = int(np.ceil(len(x_data) / self.batch_size) * self.batch_size) - len(x_data)
        # append zeros if necessary
        x_data = torch.cat([x_data, torch.zeros(n_zeros, self.dataset.seq_len).long()], dim=0)
        y_data = torch.cat([y_data, torch.zeros(n_zeros, self.dataset.seq_len).long()], dim=0)

        order = np.arange(len(x_data))
        order = order.reshape(self.batch_size, -1).T.reshape(-1)
        x_data, y_data = x_data[order], y_data[order]
        self._update_data(None, x_data, y_data)

    def _update_data(self, cid, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        # if self.train:
        self.available_clients.append(cid)
        self.clients_num_data[cid] = x_data.shape[0]
        self.data[cid] = (x_data, y_data)
        self.client_and_indices.extend([(cid, i) for i in range(x_data.shape[0])])

    def _get_item_preprocess(self, index):
        if self.client_id is None:
            client, i = self.client_and_indices[index]
        else:
            client, i = self.client_id, index
        return client, i

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            # if self.train and len(self.available_clients) < self.num_clients:
            if len(self.available_clients) < self.num_clients:
                self._add_client_train(index)
            self.length = len(self.client_and_indices)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            # if self.train:
            if index not in self.available_clients:
                self._add_client_train(index)
            # else:
            #     raise ValueError('Individual clients are not supported for test set.')
            self.length = self.clients_num_data[index]

    def __getitem__(self, index):
        client, i = self._get_item_preprocess(index)
        return tuple(tensor[i] for tensor in self.data[client])

    def __len__(self):
        return self.length


class ShakespeareH5(H5TFFDataset):
    def __init__(self, h5_path, client_id=None, seq_len=80):
        super(ShakespeareH5, self).__init__(h5_path, client_id, 'snippets')
        self.seq_len = seq_len
        # vocabulary
        self.vocab = SHAKESPEARE_VOCAB
        self.char2idx = {u: i for i, u in enumerate(self.vocab, 1)}
        self.idx2char = {i: u for i, u in enumerate(self.vocab, 1)}
        # out of vocabulary, beginning and end of speech
        self.oov = len(self.vocab) + 1
        self.bos = len(self.vocab) + 2
        self.eos = len(self.vocab) + 3

    def __getitem__(self, index):
        client, i = self._get_item_preprocess(index)
        record = self.dataset[client]['snippets'][i].decode()

        indices = np.array([self.char2idx[char] if char in self.char2idx else self.oov for char in record])
        len_chars = 1 + len(indices)  # beginning of speech
        pad_size = int(np.ceil(len_chars/self.seq_len) * self.seq_len - len_chars)
        indices = np.concatenate(([self.bos], indices, [self.eos], torch.zeros(pad_size)), axis=0)
        x = torch.from_numpy(indices[:-1]).reshape(-1, self.seq_len)
        y = torch.from_numpy(indices[1:]).reshape(-1, self.seq_len)

        return x.long(), y.long()
