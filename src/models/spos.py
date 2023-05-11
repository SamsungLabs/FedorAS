# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import copy
import random
import pickle
from typing import List

import torch.nn as nn
from src.utils.utils import setattr_recursively

class SelectOp(nn.Module):
    def __init__(self, *candidate_ops, count=False):
        super().__init__()
        self.ops = nn.ModuleList([op() for op in candidate_ops])
        self.counter = None
        self.current_option = None
        if count:
            self.counter = [0 for _ in self.ops]

    def forward(self, *x, **kwargs):
        if self.counter is not None:
            self.counter[self.current_option] += 1
        return self.ops[self.current_option](*x, **kwargs)


class SPOSMixin():
    def _init_spos(self, orig_class, sampler=None):
        self.__fields = []
        self.__unique_choices = 0
        self.__sampler = sampler
        self.__path = []
        self.orig_class = orig_class

        for m in self.modules():
            if isinstance(m, SelectOp):
                self.__fields.append(m)

        self.__unique_choices = len(self.__fields)

        #! Note this is only adding to the dict layers with buffers (i.e. Identities,activations won't be added despite these potentially being part of the searchspace)
        # mapping from parameter name to layer/op index (this will be useful during OPA aggregation)
        self._tensor_name_to_module = {}
        layer_idx = 0
        op_idx = 0
        for name, m in self.named_modules():
            if isinstance(m, SelectOp):
                op_idx = 0
                for op in m.ops:
                    for name_, _ in op.named_buffers():
                        self._tensor_name_to_module[name+'.ops.'+str(op_idx)+"."+name_] = (layer_idx, op_idx )

                    op_idx += 1
                layer_idx += 1

    def name_to_module_mapping(self, name: str):
        return self._tensor_name_to_module.get(name, None)

    def spos_set_sampler(self, sampler):
        self.__sampler = sampler

    def spos_get_sampler(self):
        return self.__sampler

    def spos_get_last_path(self):
        return self.__path

    def spos_get_num_decisions(self):
        return self.__unique_choices

    def realise(self, decision: List[int]):
        """Reverts model to the original class and replaces SelectOp layers
        with the operator whose indices are passed in `decision`. Essentially,
        this method explicitly extracts a model from the supernet."""

        assert len(decision) == self.__unique_choices, f"Decision expected to be of length {self.__unique_choices}, but got {len(decision)}."

        realization = copy.deepcopy(self)
        realization.__class__ = self.orig_class
        decision_itr = iter(decision)

        names_updates = {}
        for name, m in realization.named_modules():
            if isinstance(m, SelectOp):
                names_updates[name] = m.ops[next(decision_itr)]

        for k,v in names_updates.items():
            setattr_recursively(realization, k, v)
        return realization


    def __sample(self, batch_size=None):
        if self.__sampler is not None:
            sample = self.__sampler.sample(batch_size, self.training)
        else:
            sample = [random.randint(0, len(field.ops)-1) for field in self.__fields]

        return sample

    def __set_path(self, path):
        self.__path.clear()
        opt_i = iter(path)
        for field in self.__fields:
            field.current_option = next(opt_i)

    def __reset(self):
        for field in self.__fields:
            field.current_option = None

    def __call__(self, *x, **kwargs):

        batch_size = None
        try:
            batch_size = x[0].shape[0]
        except:
            pass

        sample = self.__sample(batch_size)
        self.__set_path(sample)
        ret = super().__call__(*x, **kwargs)
        self.__reset()
        return ret


# Below, utility functions to construct a SPOSMixin model given a standard nn.Module model with spos.SelectOp layers

_mixed_classes_register = {}

def mix_classes(*classes):
    if not classes:
        return None
    if len(classes) == 1:
        return classes[0]

    global _mixed_classes_register
    result = _mixed_classes_register.get(classes, None)
    if result is None:
        bases_names = "__".join(c.__module__.replace('.', '_') + "_" + c.__qualname__.replace('.', '_') for c in classes)
        name = f'FedorAS__{bases_names}'
        if name in globals():
            raise KeyError('Mixup class name conflict!')

        result = type(name, classes, { '__mixup_bases__': classes, '__reduce__': mixup_reduce })
        globals()[name] = result
        _mixed_classes_register[classes] = result

    return result


def mixup_recreate(classes, recargs):
    # we need to make sure that the target class is created
    mix_classes(*classes)
    # then proceed with what would have happened normally
    rec, args = pickle.loads(recargs)
    return rec(*args)


def mixup_reduce(self):
    rec, args, *rest = super(type(self), self).__reduce__()
    return (mixup_recreate, (self.__mixup_bases__, pickle.dumps((rec, args)))) + tuple(rest)


def init_module(module):
    original_class = type(module)
    module.__class__ = mix_classes(SPOSMixin, original_class)
    module._init_spos(orig_class=original_class)

