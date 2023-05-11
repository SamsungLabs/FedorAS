# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import random
import logging
from math import cos, pi
from pathlib import Path
from copy import deepcopy
import collections.abc as cabc
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common.parameter import parameters_to_weights
from flwr.common.typing import Parameters, Scalar, Weights, FitIns, FitRes

from src.models.spos import SPOSMixin
from src.models.utils import instantiate_supernet, compute_cost_mat, get_parameters, set_parameters
from src.datasets.utils import get_dataset_class_from_string, load_global_dataset, get_transforms

logger = logging.getLogger("fedoras")

def __get_server_eval_fn(
    global_dir: Path, dataset: str, eval_fn_cfg: Dict, model_cfg: Dict,
    decision_path: List[int]=None, include_test: bool=False, task_id: int=None, strip: bool=False,
) -> Callable[[Weights], Optional[Tuple[float, Dict[str, float]]]]:
    """Returns an evaluation function for centralized evaluation."""
    datasetclass = get_dataset_class_from_string(dataset)
    eval_cfg = eval_fn_cfg['eval_cfg']
    eval_func = eval_cfg['func']
    num_classes = eval_fn_cfg['num_classes'] if task_id is None else eval_fn_cfg['num_classes'][task_id]
    _, val_loader, test_loader = load_global_dataset(datasetclass,
                                                    globaldata_path=global_dir,
                                                    batch_size=eval_fn_cfg['batch_size'],
                                                    num_workers=eval_fn_cfg['num_workers'],
                                                    num_classes=num_classes,
                                                    transforms=get_transforms(datasetclass, None),
                                                    include_test=include_test)

    def evaluate(weights: Weights, round: int, do_test: bool=False) -> Optional[Tuple[float, Dict[str, float]]]:
        """Evaluates the current supernet weight on the global validation set. If a path
        is passed, the it is assumed we are interested in evaluating a concrete model in
        the supernet. This is the case in Stage-III, where we perform per-tier FL"""

        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialise the supernet
        supernet = instantiate_supernet(model_cfg, task_id=task_id, strip=strip)
        if decision_path is not None:
            print(decision_path)
            model = supernet.realise(decision_path)
        else:
            model = supernet

        set_parameters(model, weights)

        model.to(device)

        metrics = eval_func(model, test_loader if do_test else val_loader, device, eval_cfg, round, is_test=do_test)
        loss = metrics[f"{'test' if do_test else 'val'}_loss"]
        # return statistics
        return loss, metrics

    return evaluate


def prep_server_eval_fn(global_dir: Path, dataset: str, eval_fn_cfg: Dict, model_cfg: Dict,
    decision_path: List[int]=None, include_test: bool=False, tasks_per_tier: List[int]=None, strip: bool=False):
    """Returns the evaluation funtion that will be called by strategy.evaluate(). This essentially
    is the way we evaluate the performance of the global model. In a multi-task setting, a mapping of
    functions will be returned, one per task. Set `strip` to True to ensure the instantiated model discards
    elements that are not involved in the task at hand (this is relevenat in the final finetuning/end2end training stage)"""

    if tasks_per_tier is None:
        # All clients collaboratively do a single task
        eval_fn = __get_server_eval_fn(global_dir, dataset, eval_fn_cfg, model_cfg, decision_path, include_test, strip=strip)
    elif len(tasks_per_tier)==1:
        # All clients collaboratively do a single task
        eval_fn = __get_server_eval_fn(global_dir, dataset, eval_fn_cfg, model_cfg, decision_path, include_test, task_id=tasks_per_tier[0], strip=strip)
    else:
        logger.info(f"Detected multi-task setting! Tasks per tier --> {tasks_per_tier}")
        # multi-task setting
        eval_fn = {}
        for task_id in set(tasks_per_tier):
            eval_fn[task_id] = __get_server_eval_fn(global_dir, dataset, eval_fn_cfg, model_cfg, decision_path, include_test, task_id=task_id, strip=strip)

    return eval_fn


class PlainStrategy(FedAvg):
    def __init__(self,
                 strategy_cfg: Dict, # specify how to split clients into clusters, assing sampling budgets to clients etc
                 num_total_clients: int,
                 num_rounds: int,
                 eval_fn: Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]],
                 reporter,
                 clusters_info: Dict=None,
                 initial_model: nn.Module=None, # initial parameters for the global model (if not supplied, global model is randomly initialised)
                 # below arguments should be set, if at all, only for finetune/end2end stage (i.e. not doing supernet FL training)
                 decision_path: List[int]=None, # extracts a model from the supernet, this will become the global mdoel that will be trained/finetuned in FL fashion
                 task_id: int=None, # in multi-task scenarios, use this to set the tasks. Normally this only needs to be manually specified when training a model extracted from a supernet (i.e. Stage-III in FedorAS)
                ) -> None:

        randinit_supernet = instantiate_supernet(strategy_cfg.model, task_id=task_id, strip=False)
        self.decision_path = decision_path

        # if decision_path is set, the FL experiment no longer operates with a supernet but with a concrete model extracted from it
        if self.decision_path is not None:
            logger.info(f"Global model is model with decision path: {self.decision_path}")
            randinit_supernet = randinit_supernet.realise(decision_path)
            # at this point `randinit_supernet` is no longer a supernet but a model extracted from it. We keep maintian the same name of the variable for consistency throughout this class (which is purposely design for FL supernet training)

        # if `initial_model` is set, the global model (supernet or a concrete model extracted from it) is no longer randomly initialised
        if initial_model is not None:
            logger.info("Replacing random weights with supplied weights")
            randinit_supernet.load_state_dict(initial_model.state_dict(), strict=False)
            if task_id is not None:
                # set the new target taks (which might be different form that used during search)
                randinit_supernet.set_task_id(task_id=task_id, strip=True)

        initial_parameters = get_parameters(randinit_supernet)
        self.current_weights = parameters_to_weights(initial_parameters) # we want to keep a copy in the strategy (we'll update this on every round)

        self.num_rounds = num_rounds
        self.clients_per_round = strategy_cfg['clients_per_round']
        self.num_total_clients = num_total_clients
        fraction_fit = float(self.clients_per_round)/num_total_clients
        self.strategy_cfg = strategy_cfg
        fraction_eval = 0.0 # to disable federated evaluation
        super().__init__(fraction_fit=fraction_fit, min_fit_clients=self.clients_per_round,
                         min_available_clients=num_total_clients, eval_fn=eval_fn,
                         initial_parameters=initial_parameters, fraction_eval=fraction_eval)

        self.current_round = 0
        self.eval_fn_every = strategy_cfg['eval_fn_every_n']
        self.save_global_every_n = strategy_cfg['save_global_every_n']

        # Generate cost matrix
        self.cost_mat_flops, self.cost_mat_params, self.fix_flops, self.fix_params, self.max_flops, self.max_params = compute_cost_mat(strategy_cfg.model)

        # Add reporter
        self.reporter = reporter

        # saving the cost matrix etc
        self.reporter.save({'flops': self.cost_mat_flops, 'params': self.cost_mat_params,
                            'fix_costs': {'flops': self.fix_flops, 'params': self.fix_params}}, filename='cost_mat')

        # in a multi-task setting, different tiers of devices perform different tasks
        self.tasks_per_tier = strategy_cfg.get('tasks_per_tier', None)

        if decision_path is None: # i.e. we are dealing with a supernet
            # Supernet sampling mode and limit
            self.subspace_mode = strategy_cfg.sampling['mode']
            self.comms_limt = float(strategy_cfg.sampling.get('comms_limt', self.max_params))

        # Generate clusters/tiers
        if clusters_info is not None:
            logger.info("Given clusters info found")
            self.clusters = clusters_info['clusters']
            self.id_to_cluster_map = clusters_info['id_to_cluster_map']
            self.clusters_to_resources = clusters_info['clusters_to_resources']
        else:
            self.__init_clustering()
            # Associate clients to clusters (and budgets)
            self.__generate_flops_tier_brackets()

            self.reporter.save({'clusters': self.clusters, 'id_to_cluster_map': self.id_to_cluster_map,
                                'clusters_to_resources': self.clusters_to_resources}, filename='clusters_info')

        logger.info(f"Clusters: {self.clusters}")
        self.this_round_metrics = [] # will store the metrics received from clients after a fit() round is completed
        self.is_finetune_stage = False
        self.finetune_task_id = task_id

        self.transfer_config = None # config regarding how to do transfer (to be set only if experiment is supposed to do transfer after stage I&II)

    def __init_clustering(self):
        """Splits clients into clusters"""

        clustering_cfg = self.strategy_cfg.client_clustering
        self.num_clusters = clustering_cfg['num_clusters']
        client_distribution = clustering_cfg['client_distribution']

        self.clusters = [[] for _ in range(self.num_clusters)]
        self.id_to_cluster_map = [0] * self.num_total_clients

        clients = list(range(self.num_total_clients))
        random.shuffle(clients)

        # with suffled client ids, assign the exact number of clients to each tier as specified in the config
        for cluster in range(self.num_clusters):
            n = int(self.num_total_clients*client_distribution[cluster])
            cids_in_cluster = clients[:n]
            self.clusters[cluster] = cids_in_cluster
            for cid in cids_in_cluster:
                self.id_to_cluster_map[cid] = cluster
            del clients[:n]


    def __generate_flops_tier_brackets(self):
        """Splits the FLOPs range spawned by the searchspace is into brackets. Each Tier
        of device falls within one of thes FLOPs brackets."""
        # right and left limits on the PDF set from yaml config
        pdf_csum = self.strategy_cfg.client_clustering['pdf_csum']
        pdf_l_csum = self.strategy_cfg.client_clustering['pdf_l_csum']
        top_brackets = self.strategy_cfg.client_clustering.get('brackets')

        if top_brackets is not None:
            logger.info('Using supplied brackets (skipping supernet sampling)')
            brackets = [self.fix_flops] + top_brackets + [self.max_flops]

        else:

            flops_mat = self.cost_mat_flops
            num_layers = len(flops_mat)
            num_candidate_ops = len(flops_mat[list(flops_mat.keys())[0]])
            N = 100000
            paths = [np.random.randint(0, num_candidate_ops, num_layers).tolist() for _ in range(N)]
            # compute FLOPs for each sampled path
            flops = []
            for p in paths:
                flops.append(sum([flops_mat[l][d_i] for l, d_i in zip(flops_mat.keys(), p)]) + self.fix_flops)
            flops = np.sort(np.array(flops), kind='mergesort')
            flops_csum = np.cumsum(flops/flops.sum())
            min_idx = np.argwhere(flops_csum > pdf_l_csum)[0] # get the first one that's just above the 0.03 limit
            idx = np.argwhere(flops_csum > pdf_csum)[0] # get the first one that's just above the 99.7% limit
            max_resource = int(flops[idx])
            fixed_cost = int(flops[min_idx]) if pdf_l_csum else self.fix_flops

            if pdf_l_csum:
                brackets = np.linspace(fixed_cost, max_resource, self.num_clusters-1)
            else:
                brackets = np.linspace(fixed_cost, max_resource, self.num_clusters)

            # let the upper bracket of the highest tier to be the max
            if pdf_l_csum:
                brackets = np.concatenate(([self.fix_flops], brackets))
            brackets = np.concatenate((brackets, [self.max_flops]))

        # brackets should be a list of (lower, upper) brackets
        self.clusters_to_resources = []
        for i, brr in enumerate(brackets[1:]):
            self.clusters_to_resources.append((brackets[i], brr))

        logger.info(f"FLOPs brackets for each Tier: {self.clusters_to_resources}")

    def save_global_model(self, file_name: str):
        """Saves global model state_dict"""
        model = self.get_supernet(task_id=self.finetune_task_id if self.is_finetune_stage else None)
        torch.save(model.state_dict(), self.reporter.exp_dir/f"{file_name}.pt")

    def __save_results(self, data: Dict, save_model: bool):
        """Saves results (client/server metrics) and supernet."""
        supernet = self.get_supernet(task_id=self.finetune_task_id if self.is_finetune_stage else None)
        self.reporter.save_round_data(data, model=supernet if save_model else None, round=self.current_round, model_name="model" if self.is_finetune_stage else "supernet")

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function. In a multi-task setting
        a list of evaluation functions is expected."""

        if self.current_round % self.eval_fn_every == 0:
            weights = parameters_to_weights(parameters)

            if isinstance(self.eval_fn, Dict):
                logger.info("Multi-task central evaluation")
                # we are in a multi-task setting therefore we have to evaluate the supernet on each tasks
                scalars_prefix="supernet" if self.decision_path is None else "model"
                metrics = {}
                for task_id, eval_fn in self.eval_fn.items():
                    eval_res = eval_fn(weights, self.current_round)
                    loss, metrics_ = eval_res
                    # add gobal validation metrics to tensorboard
                    self.reporter.add_scalars(metrics_, step=self.current_round, prefix=scalars_prefix+f"_task{task_id}")
                    metrics[f'task{task_id}'] = metrics_ # add to metrics
            else: # not a multi-task setting
                logger.info("Single-task central evaluation")
                eval_res = self.eval_fn(weights, self.current_round)
                loss, metrics = eval_res
                # add gobal validation metrics to tensorboard
                self.reporter.add_scalars(metrics, step=self.current_round, prefix="supernet" if self.decision_path is None else "model")

            # append to this round metrics the results from the global validation stage
            round_data = {'server_metrics': metrics, 'round': self.current_round, 'client_metrics': self.this_round_metrics}
            self.__save_results(data=round_data, save_model=True if self.current_round%self.save_global_every_n==0 else False)
            self.current_round += 1
            return loss, metrics
        else:
            logger.info(f"Only running global eval_fn every {self.eval_fn_every} rounds")
            round_data = {'round': self.current_round, 'client_metrics': self.this_round_metrics}
            self.__save_results(data=round_data, save_model=True if self.current_round%self.save_global_every_n==0 else False)
            self.current_round += 1
            return None

    def evaluate_on_testset(self):
        """Evaluates`self.eval_fn` signaling that the testset should be used"""
        test_res = self.eval_fn(self.current_weights, self.current_round, do_test=True)
        loss, metrics = test_res
        self.reporter.add_scalars(metrics, step=0, prefix="model_test")
        logger.info(test_res)

    def _client_fit_error(self, failures):
        if failures:
            logger.info("\n----------------------------------- ERROR (fit_round) ----------------------------------")
            for f in failures:
                logger.info(f)

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException],
                     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self._client_fit_error(failures)
        self.this_round_metrics = [fit_res.metrics for _, fit_res in results]
        return super().aggregate_fit(rnd, results, failures)


    def __get_subspace(self):
        """Extracts a sub-supernet from the supernet that does not exceed the communication budget."""
        if self.subspace_mode == 'supernet':  # Supernet means sending the whole space
            return None
        elif self.subspace_mode == 'subnet_random':
            all_visited = False
            available_ops_flattened = []
            params_cost_mat_list = list(self.cost_mat_params.values())
            valid_ops = [None] * len(params_cost_mat_list)
            visited_ops = [None] * len(params_cost_mat_list)
            rows_without_identities = [i for i, row in enumerate(params_cost_mat_list) if row[-1] != 0]

            for idx, v in enumerate(params_cost_mat_list):
                valid_ops[idx] = [True] * len(v)
                for jdx, el in enumerate(v):  # do not sample the identity
                    if el != 0:
                        available_ops_flattened.append((idx, jdx, el))  # they don't include identities

                if idx not in rows_without_identities:
                    visited_ops[idx] = [False] * (len(v) - 1)
                    assert(v[-1] == 0)  # Cost of identity in terms of parameters is zero.
                else:
                    visited_ops[idx] = [False] * len(v)


            pre_selected_ops = []
            for row_idx in rows_without_identities:
                    sel_jdx = np.random.choice(
                        len(params_cost_mat_list[row_idx])
                    )
                    pre_selected_ops.append((row_idx, sel_jdx, params_cost_mat_list[row_idx][sel_jdx]))
                    visited_ops[row_idx][sel_jdx] = True

            # Cost of preselected
            preselected_cost = sum([el[2] for el in pre_selected_ops])

            current_budget = self.max_params
            num_elements = 0

            # Random sampling of operations until you heat the  self.comms_limt
            ops_norm_freqs_flattened = [1/(len(available_ops_flattened) - len(pre_selected_ops))] * len(available_ops_flattened)
            for i, (idx, jdx, val)  in enumerate(available_ops_flattened):
                if (idx, jdx, val) in pre_selected_ops:
                    ops_norm_freqs_flattened[i] = 0

            ops_norm_freqs_flattened = np.array(ops_norm_freqs_flattened)
            available_ops_flattened = [",".join([str(idx), str(jdx), str(el)]) for idx, jdx, el in available_ops_flattened]  # for compat with np.random.choice

            # count zeros
            num_zeros = len(ops_norm_freqs_flattened) - np.count_nonzero(ops_norm_freqs_flattened)
            nonzero = np.count_nonzero(ops_norm_freqs_flattened)
            eps = 1e-12

            # if nonzero:
            ops_norm_freqs_flattened[ops_norm_freqs_flattened==0] += eps
            ops_norm_freqs_flattened[ops_norm_freqs_flattened!=0] -= eps/nonzero

            sampled_ops_order = np.random.choice(available_ops_flattened, size=len(available_ops_flattened),
                                                replace=False, p=ops_norm_freqs_flattened)
            sampled_ops_order = [((int(idx), int(jdx)), int(cost)) for idx, jdx, cost in [map(int, el.split(',')) for el in sampled_ops_order]]

            # Select from the rest of the operations
            while current_budget > self.comms_limt - preselected_cost:
                sel, sel_cost = sampled_ops_order[num_elements]
                num_elements += 1
                assert not visited_ops[sel[0]][sel[1]]
                visited_ops[sel[0]][sel[1]] = True
                valid_ops[sel[0]][sel[1]] = False
                current_budget -= sel_cost

                all_visited = all([all(x) for x in visited_ops])
                if all_visited:
                    break

            for i,vo in enumerate(valid_ops):
                if i not in rows_without_identities:
                    assert vo[-1], "Identity not included in subspace"

            logger.info("Valid_ops:")
            for valid in valid_ops:
                logger.info(f"{[int(op) for op in valid ]}")

            return valid_ops

        else:
            raise NotImplementedError(f"{self.subpsace_mode} not implemented")

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # record current server parameters
        self.current_weights = parameters_to_weights(parameters)

        # dictionary that will be passed to each client's fit() method
        cost_mat = [layer_flops for layer_flops in self.cost_mat_flops.values()] # to list of lists
        config = {'round': rnd, 'cost_mat': cost_mat, 'fix_costs': self.fix_flops}

        # generate valid_ops mask (supernet sampling -- if enabled)
        config['valid_ops'] = None if self.is_finetune_stage else self.__get_subspace()

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # Sample clients uniformly (i.e. ignoring clustering)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # if lr_decay has been set, use the config dict to override the client's default LR
        lr_decay_config = self.strategy_cfg.lr_decay
        if lr_decay_config is not None:
            init_lr = lr_decay_config['init_lr']
            lr = init_lr
            if lr_decay_config.mode == 'cosine':
                min_lr = init_lr * lr_decay_config['factor']
                lr = min_lr + 0.5*(init_lr - min_lr)*(1 + cos((rnd - 1) / (self.num_rounds - 1) * pi))
            elif lr_decay_config.mode == 'step':
                if rnd >= self.num_rounds//2:
                    lr *= lr_decay_config['factor']
                if rnd >= 3*self.num_rounds//4:
                    lr *= lr_decay_config['factor']
            else:
                raise ValueError(f'Unknown LR scheduler: {lr_decay_config.mode}')

            if lr != init_lr:
                logger.info(f"Updated LR to: {lr}")
                config['lr'] = lr

        # Return client/config pairs with updated config based on each client's tier
        cfg_fit = []
        for cli in clients:
            tier = self.id_to_cluster_map[int(cli.cid)]
            config['tier'] = tier
            config['flops_budget'] = self.clusters_to_resources[tier]
            if self.is_finetune_stage:
                config['path'] = self.decision_path
                config['finetune_stage'] = self.is_finetune_stage
                config['task_id'] = self.finetune_task_id
            # specify which tasks client has to perform (if multi-task setting)
            if self.tasks_per_tier is not None:
                config['task_id'] = self.tasks_per_tier[tier]
            if self.transfer_config is not None:
                # we are in a transfer scenario
                config['is_transfer'] = True
                config['task_id'] = self.transfer_config['task'][tier]
                config['freeze'] = True if rnd < self.transfer_config['transfer_rounds'] else False
            fit_ins = FitIns(parameters, deepcopy(config))

            # append
            cfg_fit.append((cli, fit_ins))

        return cfg_fit

    def get_supernet(self, params=None, task_id: int=None):
        """Returns supernet with the current weights in the server if `params` is None,
        else it is populated with the supplied parameters. In a multi-task setting, use
        `task_id` to specify a particular task for the supernet. If left as `None` the whole
        supernet will be returned"""
        supernet = instantiate_supernet(self.strategy_cfg.model, task_id=task_id, strip=False if task_id is None else True)
        if self.decision_path is not None:
            supernet = supernet.realise(self.decision_path)
        set_parameters(supernet, self.current_weights if params is None else params)
        return supernet

    def get_costs_flops(self):
        cost_mat = [layer_flops for layer_flops in self.cost_mat_flops.values()]
        return cost_mat, self.fix_flops

class OPA(PlainStrategy):
    """OPerator Aggregation (OPA), an aggregation method that weights updates based on the relative
    experience of an operator across all clients that have updated that operator. Concretely, this is
    a generalisation of FedAvg where normalisation is performed independently for each layer, rather
    than collectively for full models. In order to enable that, we keep track of how many examples were
    used to update each searchable operation, independently on each client, and later use this information
    to weight updates."""
    def __init__(self,
                 strategy_cfg: Dict, # specify how to split clients into clusters, assing sampling budgets to clients etc
                 num_total_clients: int,
                 num_rounds: int,
                 eval_fn: Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]],
                 reporter,
                 clusters_info: Dict=None,
                 initial_model: nn.Module=None,
                ) -> None:
        super().__init__(strategy_cfg, num_total_clients, num_rounds, eval_fn, reporter, clusters_info,
                         initial_model=initial_model)

        self.remove_single_user_updates = False


    def __conservative_sum(self, it):
        ''' Like normal "sum" but returns `None` if an empty iterable is given
            and does not perform "first + start" if a length-1 sequence is given,
            instead returns the first and only element directly
        '''
        try:
            first = next(it)
        except StopIteration:
            return None

        return sum(it, first)

    def __aggr(self, supernet: SPOSMixin, client_supernets: List, factors):
            """Aggregates client updates using pre-computed normalization factors"""

            aggregate_params = {}
            for iter_fn, get_fn in [(supernet.named_parameters, 'get_parameter'), (supernet.named_buffers, 'get_buffer')]:
                for name, _ in iter_fn():
                    if not any(name in f for f in factors):
                        continue # not a single update to weights

                    # Sanity checks, can be safely removed without affecting functionality
                    fsum = sum(f.get(name, 0.0) for f in factors)
                    assert all(f.get(name, 0.0) >= 0.0 for f in factors), f'Negative normalisation factor encountered for layer {name!r}'
                    assert 0 <= round(fsum, 6) <= 1.0, f'Normalisation factors for a layer {name!r} do not sum to 1, current value: {fsum}'
                    # Actual functionality
                    agg = self.__conservative_sum(f[name] * getattr(cli_supernet, get_fn)(name) for f, cli_supernet in zip(factors, client_supernets) if name in f)
                    assert agg is not None
                    if round(fsum, 6) != 1:
                        agg += (1 - fsum) * getattr(supernet, get_fn)(name)
                    aggregate_params[name] = agg
            return aggregate_params

    def __normalise_ops_samples(self, results, num_layers_supernet: int):

        # Retrieve histograms describing the paths in the supernet sampled by each client in the round
        histograms = [fit_res.metrics['histogram'] for _, fit_res in results]
        metrics = [fit_res.metrics for _, fit_res in results]
        client_parameters = [parameters_to_weights(fit_res.parameters) for _, fit_res in results]

        all_examples = sum(m['total_examples'] for m in metrics)
        sum_of_histograms = []
        for idx in range(num_layers_supernet):
            this_op_histograms = np.asarray([h[idx] for h in histograms])
            sum_of_histograms.append(this_op_histograms.sum(axis=0))

        # Sanity checks
        if not self.remove_single_user_updates:  #! These assertions are no longer valid when you are changing the histograms of single user updates
            assert all(all(hh.sum() == m['total_examples'] for hh in h) for m, h in zip(metrics, histograms)), 'Mixop called fewer times than a full model?'
            assert all(sh.sum() == all_examples for sh in sum_of_histograms), 'All mixops called fewer times than all models?'

        factors = []
        client_supernets = []
        for params, m, h in zip(client_parameters, metrics, histograms):
            # inflate params into a proper pytorch supernet
            supernet = self.get_supernet(params=params)
            client_supernets.append(supernet)
            this_model_factors = {}
            for iter_fn in [nn.Module.named_parameters, nn.Module.named_buffers]:
                for name, _ in iter_fn(supernet):
                    meta_info = supernet.name_to_module_mapping(name)
                    if meta_info is None: # param/buffer belongs to a fixed layer, like stem
                        this_model_factors[name] = m['total_examples'] / all_examples
                    else:
                        layer_idx, op_idx = meta_info
                        if not sum_of_histograms[layer_idx][op_idx] or not h[layer_idx][op_idx]: # the op hasn't been selected at all or hasn't been updated by this module
                            continue
                        this_model_factors[name] = h[layer_idx][op_idx] / sum_of_histograms[layer_idx][op_idx]

            factors.append(this_model_factors)

        return factors, client_supernets

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException],
                     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using a histogra-informed weighted averaging."""

        # record for saving to pickle
        self.this_round_metrics = [fit_res.metrics for _, fit_res in results]

        self._client_fit_error(failures)
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # instantiate PyTorch supernet (note that in the server the supernet is a list of numpy arrays, we instead
        # convert it into its PyTorch representation to make the aggregation logic clearer)
        supernet = self.get_supernet()

        # Compute biased normalization factors
        factors, client_supernets = self.__normalise_ops_samples(results, supernet.spos_get_num_decisions())

        # Now aggregate
        aggregated_weights = self.__aggr(supernet, client_supernets, factors)
        supernet.load_state_dict(aggregated_weights, strict=False)

        # Now convert PyTorch weights into list of numpy arrays
        parameters_aggregated = get_parameters(supernet)

        return parameters_aggregated, {}
