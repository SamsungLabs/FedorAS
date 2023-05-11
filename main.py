# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import pickle
import argparse

import torch
import awesomeyaml
import flwr as fl

from src.server import prep_server_eval_fn
from src.datasets.utils import prepare_dataset
from src.utils.utils import set_fedoras_logger
from src.finetune import finetune_model_in_cluster
from src.client import get_clientfn_and_ray_resources

parser = argparse.ArgumentParser(description="FedorAS")
parser.add_argument('yamls', nargs='+', type=str)
parser.add_argument("--supernet", type=str, default=None, help="Path to a pre-trained supernet state_dict")
parser.add_argument("--not-strict", action='store_true', help="Set if loading supernet state_dict shouldn't be strict")
parser.add_argument("--no-stage1", action='store_true', help="Set if you want to jump directly to Stage-II: Supernet search and validation")
parser.add_argument("--no-end2end", action='store_true', help="Set if you want to skip end2end training in Stage-III")


torch.random.manual_seed(2022)

def main():

    # parse config
    args = parser.parse_args()
    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)

    # setting up fedoras logger, first tone down Flower's logger
    logger = set_fedoras_logger(cfg.misc.exp_dir, tone_down_flwr_log=True)
    logger.info(cfg)

    awesomeyaml.yaml.dump(cfg.ayns.source, str(cfg.misc.exp_dir / 'config.yaml'), sort_keys=False)

    # ? Uncomment to test supernet instantation and extracting realisation
    # from src.models.utils import instantiate_supernet
    # supernet = instantiate_supernet(cfg.model)
    # mmodel = supernet.realise([0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6]) # replace w/ your decision

    # Download and partition dataset
    fed_dir, global_dir = prepare_dataset(cfg.dataset['name'], data_path=cfg.dataset['data_path'],
                                          number_of_clients=cfg.dataset['num_partitions'],
                                          lda_alpha=cfg.dataset['lda_alpha'],
                                          val_ratio=cfg.dataset['val_ratio'],
                                          exp_dir=cfg.misc.exp_dir)

    # Define function that will be run on the server to evaluate the global model
    eval_fn = prep_server_eval_fn(global_dir, cfg.dataset['name'], cfg.server.eval_fn, cfg.model, tasks_per_tier=cfg.server.strategy.tasks_per_tier)

    ########### Stage I: Federated Supernet Training ###########

    if args.supernet is None:
        supernet = None
    else:
        logger.info("Loading supplied supernet state_dict")
        from src.models.utils import instantiate_supernet
        supernet = instantiate_supernet(cfg.model)
        supernet_state_dict = torch.load(args.supernet)
        missing, unexpected = supernet.load_state_dict(supernet_state_dict, strict=not(args.not_strict))
        if len(missing) > 0:
            logger.warning(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected Keys: {unexpected}")

    # Define strategy
    strategy = cfg.server.strategy.fn(strategy_cfg=cfg.server.strategy, eval_fn=eval_fn, reporter=cfg.reporter(), initial_model=supernet)

    if not(args.no_stage1):
        # Define clinets and Ray resources
        client_fn, client_resources = get_clientfn_and_ray_resources(cfg.client, fed_dir=fed_dir)

        # start simulation
        if fed_dir.exists(): # this is an unnecessary if statement (but prevents the IDE from thinking the process exits after running start_simulation)
            hist = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=cfg.dataset['num_partitions'],
                client_resources=client_resources,
                num_rounds=cfg.server['num_rounds'],
                strategy=strategy,
                ray_init_args={'include_dashboard': False},
            )
        # save supernet as it was after the last FL round
        strategy.save_global_model("supernet")

    ########### Stage II: Per-Tier Search & Validation ###########

    # Retrieve supernet and other coponents from server
    supernet = strategy.get_supernet(task_id=cfg.server.eval_fn["task_id"])
    cost_mat, fix_costs = strategy.get_costs_flops()
    reporter = strategy.reporter

    # Define search method and launch search
    searcher = cfg.search.method(supernet, cost_mat, fix_costs, strategy.clusters_to_resources)
    # searcher.search_sequentially(global_dir, cfg.server.eval_fn)
    searcher.search(global_dir, cfg.server.eval_fn)
    for i, bckt in enumerate(searcher.path_buckets):
        reporter.save(bckt, f'search_data_bucket_{i}')
    # Retrieve best model for each Tier of devices
    best_models = searcher.get_best_models()
    # save each best model
    for i, model in enumerate(best_models):
        reporter.save(model, f'best_model_bucket_{i}')

    logger.info("Proceeding to Stage-III (see logs in finetune directories)")
    ########### Stage III: Federated per-tier Fine-tuning ###########

    exp_dir = reporter.exp_dir
    with open(exp_dir/"clusters_info.pkl", 'rb') as h:
        clusters_info = pickle.load(h)

    for model in best_models:
        # Finetune model using weights extracted from the supernet
        finetune_model_in_cluster(model, cfg, clusters_info, exp_dir)

        if not(args.no_end2end):
            # Train from scratch that same model (doesn't use supernet weights)
            finetune_model_in_cluster(model, cfg, clusters_info, exp_dir, end2end=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(asctime)s | [%(filename)s:%(lineno)d] | %(message)s")
    main()
