
# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

# TODO: probably theres's a better name for this script
import sys
import pickle
import awesomeyaml
import argparse
from typing import Dict, List, OrderedDict
from pathlib import Path

import torch

from src.utils.utils import getattr_recursively
from src.finetune import finetune_model_in_cluster

parser = argparse.ArgumentParser(description="FedorAS_finetune")

parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory")
parser.add_argument("--best_model_data", type=str, help="Pickle in exp_dir containing the best model data to finetune (e.g. best_model_bucket_0.pkl)")
parser.add_argument("--decision", type=str, help="Coma separated integers denoting a path along the supernet") # if 1.0, then only one client will run per GPU.
parser.add_argument("--end2end", action='store_true', help="Indicates the model will not use pretrained weights from the supernet")
parser.add_argument("--supernet", type=str, default='supernet.pt', help="Path to a supernet state_dict. By default it loads`supernet.pt` in `exp_dir`")
parser.add_argument("--not-strict", action='store_true', help="Set if loading supernet state_dict shouldn't be strict")
parser.add_argument('yamls', nargs='+', type=str)

torch.random.manual_seed(2022)


def assign_decision_to_tier(exp_dir: Path, decision: List[int], clusters_info: Dict):
    """Maps a given decision to a tier. Here we assume the cost metric is FLOPs"""
    with open(exp_dir/"cost_mat.pkl", 'rb') as h:
        cost_mat = pickle.load(h)

    # count flops in model
    flops = cost_mat['fix_costs']['flops'] + sum([flops_layer[op] for flops_layer, op in zip(cost_mat['flops'].values(), decision)])
    print(f"Decision {decision} has {flops} FLOPs")

    # get tier
    tier = -1
    for i, brr in enumerate(clusters_info['clusters_to_resources']):
        if brr[0] <= flops < brr[1]:
            tier = i
            break

    print(f"Decision {decision} is a Tier {tier} model")

    return tier


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(asctime)s | [%(filename)s:%(lineno)d] | %(message)s")

    args = parser.parse_args()

    if bool(args.best_model_data) == bool(args.decision):
        print("Specify --best_model_data or --decision")
        sys.exit()

    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)
    print(cfg)

    exp_dir = Path(args.exp_dir)
    assert exp_dir.exists(), f"exp_dir: {args.exp_dir} does not exist"

    with open(exp_dir/"clusters_info.pkl", 'rb') as h:
        clusters_info = pickle.load(h)

    if args.best_model_data is not None: # loading a pre-trained model extracted from the supernet
        best_model_data = exp_dir/args.best_model_data
        assert best_model_data.exists(), f"best_model_data: {best_model_data} does not exist"

        with open(best_model_data, 'rb') as h:
            model_data = pickle.load(h)

    else: # extracting a model from the supernet
        from src.models.utils import instantiate_supernet
        decision_as_list = [int(d) for d in args.decision.split(',')]
        supernet = instantiate_supernet(cfg.model)
        if not(args.end2end): # if not end2end, load weights from existing supernet
            if args.supernet=="supernet.pt":
                supernet_path = exp_dir/"supernet.pt"
                orig_supernet_statedict = torch.load(supernet_path)
            else:
                # a different path to a supernet state dict is provided
                supernet_statedict = torch.load(args.supernet)
                orig_supernet_statedict = OrderedDict()
                for k, v in supernet_statedict.items():
                    if args.not_strict:
                        # don't include those elements that do not match in terms of shape
                        module_shape = getattr_recursively(supernet, k).shape
                        if v.shape != module_shape:
                            print(f"{v.shape} and {module_shape} do not match in shape... skipping")
                            continue
                    orig_supernet_statedict[k] = v

            missing, unexpected = supernet.load_state_dict(orig_supernet_statedict, strict=not(args.not_strict))
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")
        mmodel = supernet.realise(decision_as_list)

        # construct model_data dictionary to be passed to src.finetune
        model_data = {}
        model_data['model'] = mmodel
        model_data['path'] = decision_as_list
        # determine to which tier this model belongs to
        model_data['bucket'] = assign_decision_to_tier(exp_dir, decision_as_list, clusters_info)

    finetune_model_in_cluster(model_data, cfg, clusters_info, exp_dir, args.end2end, new_dir=True)
