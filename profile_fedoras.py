# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import random
import argparse

import torch
import awesomeyaml
from ptflops import get_model_complexity_info

from src.models.utils import instantiate_supernet, compute_cost_mat

parser = argparse.ArgumentParser(description="FedorAS")
parser.add_argument('yamls', nargs='+', type=str)
parser.add_argument("--decision", type=str, required=True, help="path to extract from supernet")
parser.add_argument("--batch", type=int, default=64, help="batch size to use for profiler")

def main():

    # parse config
    args = parser.parse_args()
    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)

    # parsing decision
    decision_as_list = [int(d) for d in args.decision.split(',')]

    # compute cost matrix and fix costs for supernet
    cost_mat_flops, _, _, _, _, _ = compute_cost_mat(cfg.model)

    # measure flops/params in model by passing a single input
    supernet = instantiate_supernet(cfg.model)
    mmodel = supernet.realise(decision_as_list)
    input_shape = cfg.model.type.keywords['input_size'][1:]
    print(f"Detected input shape for {cfg.dataset.name} dataset: {input_shape}")
    flops, params = get_model_complexity_info(mmodel, tuple(input_shape), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
    print(f"FLOPS: {flops/1e6:.2f} MFLOPs // params: {params/1e6:.3f} Million")

    # Samples 10 random paths
    supernet = instantiate_supernet(cfg.model)
    N = 10
    mmodels = []
    for _ in range(N):
        # randomly sample an OP from each layer in the supernet
        decision = [random.randrange(0,len(num_ops)) for num_ops in cost_mat_flops.values()]
        print(f"paths to be profiled: {decision}")
        mmodels.append(supernet.realise(decision))

    # append the model obtained from the decision passed as input arg
    mmodels.append(mmodel)
    
    # profile memory peak (as in training --> here we are keeping activations eventhough we do just the forward pass)
    # if you only care about inference peak memory, then wrap the lines below under `with torch.no_grad():`
    inputs = torch.randn((args.batch,)+tuple(input_shape))
    print(f"Batch size used for profiling: {inputs.shape}")
    with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=N+1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fedoras_profiler'),
                record_shapes=False,
                profile_memory=True,
                with_stack=False
        ) as prof:
        for i, model in enumerate(mmodels):
            print(i)
            model(inputs)
            prof.step()
    
    # then run: tensorboard --logdir log/fedoras_profiler to inspec the results

if __name__ == "__main__":

    args = parser.parse_args()
    main()