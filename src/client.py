# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

from asyncio.log import logger
from typing import Dict
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar


from src.train import train, eval
from src.models.utils import instantiate_supernet, Sampler
from src.datasets.utils import get_dataset_class_from_string


def get_clientfn_and_ray_resources(cfg, fed_dir: Path):

    def client_fn(cid: str) -> FedorASClient:
        return cfg.type(cid=cid, fed_dir=fed_dir)

    return client_fn, cfg['ray_resources']

class FedorASClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir: str, client_cfg: Dict, model_cfg: Dict):
        self.cid = cid
        self.fed_dir = Path(fed_dir)
        self.model_cfg = model_cfg
        self.client_cfg = client_cfg
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        round = config['round']
        tier = config['tier']
        flops_budget = config['flops_budget']
        finetune_stage = config.get('finetune_stage', False)
        task_id = config.get('task_id', None)
        print(f"> Spawned client {self.cid}, belongs to tier {tier} with bracket {flops_budget} FLOPs [task_id: {task_id}]")

        # Init supernet
        supernet = instantiate_supernet(self.model_cfg, task_id=task_id, strip=finetune_stage)

        if finetune_stage:
            # extract model from supernet
            self.model = supernet.realise(config['path'])
        else:
            self.model = supernet
            # Define sampler and attach to supernet
            sampler = Sampler(config['cost_mat'], config['fix_costs'], flops_budget[1], valid_ops=config['valid_ops'])
            self.model.spos_set_sampler(sampler)

        self.set_parameters(parameters)

        # update LR for optimizer if passed in config dict sent by server
        lr = config.get('lr', self.client_cfg.optim.type.keywords['lr'])

        if config.get('freeze', False):
            # we are in a transfer setting and we first need to freeze all the model except the output fully connected layer
            fc_parameters = getattr(self.model, f'classifier_{task_id}').parameters()
            optim = self.client_cfg.optim.type(params=fc_parameters, lr=lr)
        else:
            # Instantiate optimizer passing all model parameters
            optim = self.client_cfg.optim.type(params=self.model.parameters(), lr=lr)

        # load data for this client and get trainloader
        num_classes = self.client_cfg['num_classes'] if task_id is None else self.client_cfg['num_classes'][task_id]
        dataset_cls = get_dataset_class_from_string(self.client_cfg['dataset'])
        norm_params = dataset_cls.get_norm_params(self.cid)
        train_transforms = dataset_cls.get_train_transforms_fl(norm_params) if finetune_stage else dataset_cls.get_train_transforms(norm_params)
        trainloader = dataset_cls.get_dataloader(self.fed_dir, partition_name="train",
                                               batch_size=self.client_cfg['batch_size'],
                                               workers=self.client_cfg['num_workers'],
                                               transforms=train_transforms,
                                               cid=self.cid,
                                               num_classes=num_classes
                                               )

        # send model to device
        self.model.to(self.device)

        # train
        metrics = train(self.model, optim, trainloader, self.device, self.client_cfg.train_cfg, round, self.cid, tier)

        # return local model and statistics
        if not(finetune_stage):
            metrics['histogram'] = self.model.spos_get_sampler().get_decisions_histogram(examples=True)

        return self.get_parameters(), len(trainloader.dataset), metrics

    def evaluate(self, parameters, config):

        raise NotImplementedError("Federated Evaluation not implemented")
        # init supernet
        self.supernet = instantiate_supernet(self.model_cfg)

        # print(f"fit() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        dataset_cls = get_dataset_class_from_string(self.client_cfg['dataset'])

        norm_params = dataset_cls.get_norm_params(self.cid)
        train_transforms = dataset_cls.get_eval_transforms(norm_params)
        valloader = dataset_cls.get_dataloader(self.fed_dir, partition_name="val",
                                               batch_size=self.client_cfg['batch_size'],
                                               workers=self.client_cfg['num_workers'],
                                               transforms=train_transforms,
                                               num_classes=self.client_cfg['num_classes']
                                               )

        # send model to device
        self.supernet.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}