# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import copy
from typing import Dict
from pathlib import Path

import flwr as fl

from src.utils.utils import set_fedoras_logger
from src.datasets.utils import prepare_dataset
from src.models.utils import instantiate_supernet
from src.client import get_clientfn_and_ray_resources
from src.server import prep_server_eval_fn, PlainStrategy


def finetune_model_in_cluster(model_data: Dict, config: Dict, clusters_info: Dict, exp_dir: Path, end2end: bool=False, new_dir: bool=False):
    """Finetunes a model in a FL manner only considering the clients in the tier and those in higher tiers. The dict `model_data` contains
    the model itself, the tier it belongs to, the path it was follow to extract it from the supernet, and other metrics. If `end2end`=True,
    the model is randomly initialised (instead of using the weights extracted from the supernet)."""

    current_tier = model_data['bucket']
    dir = "end2end" if end2end else "finetune"
    finetune_dir = exp_dir/dir/f"Tier{model_data['bucket']}"

    if new_dir:
        # extend directory so to not override the results that are already there
        import datetime
        finetune_dir = finetune_dir/str(datetime.datetime.now().strftime('%b%d_%H:%M:%S'))

    # update filehandler in fedoras logger to point to a new file in the directory where the current fintuning directory
    logger = set_fedoras_logger(finetune_dir, tone_down_flwr_log=True)
    logger.info(f"Starting finetuning for a model from Tier {current_tier} --> {finetune_dir}")

    # Download and partition dataset
    fed_dir, global_dir = prepare_dataset(config.dataset['name'], data_path=config.dataset['data_path'],
                                        number_of_clients=config.dataset['num_partitions'],
                                        lda_alpha=config.dataset['lda_alpha'],
                                        val_ratio=config.dataset['val_ratio'])

    config = copy.deepcopy(config)
    if end2end:
        supernet = instantiate_supernet(config.model)
        # extract model from randomly initialised supernet
        initial_model = supernet.realise(model_data['path'])
        num_rounds = config.finetune['num_rounds_end2end']

        # we also overwrite the `init_lr` parameter in the config with that in `init_lr_end2end`. (keeps the logic in strategy cleaner)
        config.finetune.strategy.lr_decay['init_lr'] = config.finetune.strategy.lr_decay['init_lr_end2end']
        # similar logic for number of clients to sample on each round
        config.finetune.strategy['clients_per_round'] = config.finetune.strategy['clients_per_round_end2end']
    else:
        initial_model = model_data['model']
        num_rounds = config.finetune['num_rounds']

    # get eval function, this time including test loader. Test eval will be performed at the end of the finetuning/end2end stage
    tasks_per_tier = config.server.strategy.get('tasks_per_tier')
    task_id = None
    if tasks_per_tier is not None:
        # We are now finetuning or training end2end a particular model that falls within a Tier bracket but that was trained
        # in a multi-task setting in Stage-I. Even though clients from above tiers can participate (if they can run the model)
        # we still maintain the task that this Tier was envisioned to do. This means that all clients in this stage do the same task
        task_id = tasks_per_tier[current_tier]


    # determnie which clients should participate
    if config.finetune.transfer is None:
        # we consider sampling clients that belong to this tier and above
        # count number of clients in this tier and get client ids
        clients_in_tier = len(clusters_info['clusters'][current_tier])
        this_tier_cids = [str(cid) for cid in clusters_info['clusters'][current_tier]]
        logger.info(f"There are {clients_in_tier} clients in Tier {current_tier}")
        logger.info(f"This tier cids: {this_tier_cids}")
        cids_to_consider = this_tier_cids
        for tier_id, cids in enumerate(clusters_info['clusters']):
            if tier_id > current_tier:
                logger.info(f"Also considering clients in tier {tier_id}")
                cids_to_consider.extend([str(cid) for cid in cids])
    else:
        # we consider only tiers that are flagged as `true` in transfer_mask
        # the idea here is that only higher tier clients would be able to perform the more complex task
        # therefore for lower tier models, these are only trained in clients that belong to higher tiers
        cids_to_consider = []
        logger.info("Transfer setting detected")
        for tier_id, this_tier_mask, cids in zip(range(len(clusters_info['clusters'])), config.finetune.transfer['tiers_mask'], clusters_info['clusters']):
            if this_tier_mask:
                logger.info(f"Considering clients in tier {tier_id}")
                cids_to_consider.extend([str(cid) for cid in cids])

        # this is a transfer setting, we should look for task_id info somewhere else in the config
        task_id = config.finetune.transfer['task'][current_tier]

    total_clients = len(cids_to_consider)
    logger.info(f"Clients consider in this finetuning: {cids_to_consider}")

    eval_fn = prep_server_eval_fn(global_dir, config.dataset['name'], config.finetune.eval_fn,
                                 config.model, decision_path=model_data['path'], include_test=True, tasks_per_tier=[task_id], strip=True)

    client_fn, client_resources = get_clientfn_and_ray_resources(config.finetune.client, fed_dir)

    # Define strategy
    reporter = config.finetune.reporter(exp_dir=finetune_dir)
    strategy = PlainStrategy(strategy_cfg=config.finetune.strategy, num_total_clients=total_clients,
                             eval_fn=eval_fn, reporter=reporter, clusters_info=clusters_info, decision_path=model_data['path'],
                             initial_model=initial_model, num_rounds=num_rounds, task_id=task_id)
    strategy.is_finetune_stage = True
    strategy.transfer_config = config.finetune.transfer

    if fed_dir.exists(): # this is an unnecessary if statement (but prevents the IDE from thinking the process exits after running start_simulation)
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=total_clients,
            clients_ids=cids_to_consider,
            client_resources=client_resources,
            num_rounds=num_rounds,
            strategy=strategy,
            ray_init_args={'include_dashboard': False},
        )
    # save supernet as it was after the last FL round
    strategy.save_global_model("model")

    # eval on testset
    strategy.evaluate_on_testset()
