# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import random
import logging
from typing import Dict, List
import collections.abc as cabc
from collections import OrderedDict

import torch
import numpy as np
from flwr.common.typing import Parameters
from flwr.common.parameter import weights_to_parameters

from src.models.spos import init_module, SelectOp

logger = logging.getLogger("fedoras")

def get_parameters(model) -> Parameters:
    """Returns parameters from a model."""
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = weights_to_parameters(weights)

    return parameters

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)

def prep_supernet_from_config(config):

    gen_keys = config.type.keywords
    return config.type, gen_keys, gen_keys['input_size'], config.get('blocks', None)


def instantiate_supernet(model_cfg: Dict, task_id: int=None, strip: bool=False):
    dataset_cls, args, _, _ = prep_supernet_from_config(model_cfg)
    supernet = dataset_cls(**args)
    init_module(supernet)
    if task_id is not None:
        num_classes = model_cfg.type.keywords['num_classes']
        if isinstance(num_classes, cabc.Sequence):
            supernet.set_task_id(task_id, strip=strip)
            logging.info(f"Instantiated supernet for task-id: {task_id}")
        else:
            logging.warning(f"Supernet is not multi-task. Not setting task-id")
    return supernet


def remove_flops_counting_methods(net_main_module):
    del net_main_module.start_flops_count
    del net_main_module.stop_flops_count
    del net_main_module.reset_flops_count
    del net_main_module.compute_average_flops_cost


def get_flops(m, is_=None):
    if is_ is None:
        is_ = m.get_input_size()
    try:
        raise Exception
    except:
        import ptflops
        if callable(is_):
            ret = ptflops.get_model_complexity_info(m, (m,), print_per_layer_stat=False, as_strings=False, input_constructor=is_)[0]
        else:
            ret = ptflops.get_model_complexity_info(m, tuple(is_[1:]), print_per_layer_stat=False, as_strings=False)[0]
        remove_flops_counting_methods(m)
        return ret

def get_params(m, is_=None):
    acc = 0
    for param in m.parameters():
        acc += param.numel()

    return acc


def compute_cost_mat(model_cfg: Dict):
    """Computes the associated FLOPs and parameter count metrics for each
    candidate operator in a SPOSMixin supernet. This function returns in addition
    returns the fixed costs (in terms of FLOPs and params) for the layers that are
    not searched over."""

    def record_input_shape(self, input):
        self.input_shape = input[0].shape

    # instantiate
    # if this is a multi-task model, we'll fix the final FC layer to use the target task (i.e. num_classes[1])
    # regardless of which FC to use to compute the cost matrix, the change in FLOPs/Params will be marginal
    model = instantiate_supernet(model_cfg, task_id=1)

    input_size = model_cfg.type.keywords['input_size']
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, SelectOp):
            hooks.append(m.register_forward_pre_hook(record_input_shape))

    # pass input
    model.eval()
    _ = model(torch.randn(input_size))

    cost_mat_flops = OrderedDict()
    cost_mat_params = OrderedDict()

    # retrieve input shapes to each block
    for name, m in model.named_modules():
        if isinstance(m, SelectOp):
            # remove hook
            h = hooks.pop(0)
            h.remove()

            for op in m.ops:
                flops = get_flops(op, m.input_shape)
                params = get_params(op)
                # add metrics to dict
                if name not in cost_mat_flops.keys():
                    cost_mat_flops[name] = [flops]
                    cost_mat_params[name] = [params]
                else:
                    cost_mat_flops[name].append(flops)
                    cost_mat_params[name].append(params)

    # meausring fixed FLOPs costs (i.e. by selecting the op w/ fewer FLOPs in each layer)
    model = instantiate_supernet(model_cfg, task_id=1)
    for name, m in model.named_modules():
        if isinstance(m, SelectOp):
            min_op_idx = cost_mat_flops[name].index(min(cost_mat_flops[name]))
            m.ops = torch.nn.ModuleList([m.ops[min_op_idx]])
    fix_flops = get_flops(model, input_size)

    # meausring fixed PARAMs costs (by selecting the OP with most parameters in each layer)
    model = instantiate_supernet(model_cfg, task_id=1)
    for name, m in model.named_modules():
        if isinstance(m, SelectOp):
            min_op_idx = cost_mat_params[name].index(min(cost_mat_params[name]))
            m.ops = torch.nn.ModuleList([m.ops[min_op_idx]])
    fix_params = get_params(model)

    # compute max flops and max params a single path over the supernet can have
    max_flops = sum(max(flps) for flps in cost_mat_flops.values()) + fix_flops
    max_params = sum(sum(parm) for parm in cost_mat_params.values()) + fix_params

    for k, v in cost_mat_flops.items():
        logger.info(f"{k} FLOPS --> {v}")

    logger.info(f"fix_flops: {fix_flops}")
    logger.info(f"max_flops: {max_flops}")

    for k, v in cost_mat_params.items():
        logger.info(f"{k} PARAMS --> {v}")
    logger.info(f"fix_params: {fix_params}")
    logger.info(f"max_params: {max_params}")


    return cost_mat_flops, cost_mat_params, fix_flops, fix_params, max_flops, max_params

class Sampler():
    def __init__(self, cost_matrix, fix_cost, client_budget, valid_ops=None):

        self.cost_matrix = cost_matrix
        self.fix_cost = fix_cost
        self.budget = client_budget
        self.paths = [] # list of lists containing indices of sampled nodes. Each list contains N integers (w/ N being depth of tree)

        if valid_ops is None:
            self.valid_ops = [[True] * len(cost_matrix[i]) for i in range(len(cost_matrix))]
        else:
            self.valid_ops = valid_ops

        self.last_flops = 0
        self.layer_choices = [len(costs) for costs in self.cost_matrix]

    def _get_probabilities(self, remaining_budget, layer_idx):
        """Returns probabilities of sampling each children node of the current head node.
        This method assigns uniform probability to all child nodes that do not exhaust the
        remaining budget."""

        mask = [c for c in self.cost_matrix[layer_idx]]

        # Zero-out the probability if it is an invalid op
        for jdx in range(len(mask)):
            mask[jdx] = float('inf') if not self.valid_ops[layer_idx][jdx] else mask[jdx]

        # given the current head and the running_cost, construct prob distribution
        mask_ = [float(remaining_budget >= cost) for cost in mask]

        # probability of sampling each childnode/operator (here assuming uniform distribution
        # over the masked operators)
        den = sum(mask_)
        assert den > 0, "Woops, budget exceeded"

        return [m/den for m in mask_]

    def sample(self, batch_size=None, track_paths: bool = True):
        """Returns a list of integers indicating operation to choose in each layer. The order
        at which each layer is sampled is randomized."""

        # reset
        remaining_budget = self.budget

        num_layers = len(self.layer_choices)
        # randomize layer sampling order
        layers_without_identity = set([i for i, row in enumerate(self.cost_matrix) if row[-1] != 0])
        order = random.sample(set(range(num_layers)) - layers_without_identity, num_layers - len(layers_without_identity))
        decision = [0]*num_layers

        # substracting fix cost
        remaining_budget -= self.fix_cost
        self.last_flops = self.fix_cost

        # Always select from layers without identities first
        for layer_idx in layers_without_identity:
            probs = self._get_probabilities(remaining_budget, layer_idx)
            op_id = random.choices(range(len(self.cost_matrix[layer_idx])), probs)[0]
            remaining_budget -= self.cost_matrix[layer_idx][op_id]
            self.last_flops += self.cost_matrix[layer_idx][op_id]
            decision[layer_idx] = op_id

        # for each layer select an op without exceeding budget
        for layer_idx in order:
            probs = self._get_probabilities(remaining_budget, layer_idx)

            # sample
            op_id = random.choices(range(len(self.cost_matrix[layer_idx])), probs)[0]
            decision[layer_idx] = op_id

            # update budget
            remaining_budget -= self.cost_matrix[layer_idx][op_id]

            # keep counting
            self.last_flops += self.cost_matrix[layer_idx][op_id]

        # append path
        if track_paths:
            self.paths.append((decision, batch_size))

        return decision

    def get_decisions_histogram(self, examples=False):
        """ Constructs and returns a per-layer histogram showing the frequency
            each node was sampled.

            Arguments:
                examples : if ``False``, each value returned is the number of
                    forward passes that were run using a relevant operation, otherwise
                    the number is the number of examples experienced by the operation
                    (roughly speaking, the number of forward passes times batch size,
                    but the actual value might be slightly different if batch size
                    does not divide dataset size)
        """

        # construct histogram
        hist = [np.zeros((num,), dtype=np.int32) for num in self.layer_choices]
        for path, bs in self.paths:
            for i, choice in enumerate(path):
                hist[i][choice] += 1 if not examples else (bs or 0)

        return hist

    def get_last_flops(self):
        return self.last_flops

    def count_flops_in_path(self, path: List[int]):
        """Given a path, this counts the FLOPs"""

        flops = self.fix_cost + sum(self.cost_matrix[layer][op] for layer, op in enumerate(path))
        return flops
