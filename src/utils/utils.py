# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import logging
from pathlib import Path


def set_logger(logger, exp_dir):

    for hndlr in logger.handlers:
        logger.removeHandler(hndlr)
    FORMAT = logging.Formatter("%(levelname)s | %(asctime)s | [%(filename)s:%(lineno)d] | %(message)s")
    h = logging.FileHandler(f'{exp_dir}/log.txt')
    h.setFormatter(FORMAT)
    h.setLevel(logging.DEBUG)
    logger.addHandler(h)
    return logger

def set_fedoras_logger(exp_dir:str, tone_down_flwr_log: bool=False):
    Path(exp_dir).mkdir(parents=True)
    logger = logging.getLogger("fedoras")
    logger = set_logger(logger, exp_dir)
    if tone_down_flwr_log:
        flwr_log = logging.getLogger("flower")
        flwr_log.propagate = False
        set_logger(flwr_log, exp_dir)
    return logger

def setattr_recursively(module, layer_str, value):
    """Givent a string in the form a.b.c pointing to a particular submodule in `module`,
    this function traversers `module`'s submouldes until the leaf attribute is found.
    Then it gets replaced with `value`."""
    if "." in layer_str:
        pos = layer_str.find(".")
        module = setattr_recursively(getattr(module, layer_str[:pos]), layer_str[pos+1:], value)
    else:
        # no more "." in string, set attribute
        setattr(module,layer_str, value)


def getattr_recursively(module, layer_str):
    if "." in layer_str:
        pos = layer_str.find(".")
        return getattr_recursively(getattr(module, layer_str[:pos]), layer_str[pos+1:])
    else:
        # no more "." in string, set attribute
        return getattr(module, layer_str)