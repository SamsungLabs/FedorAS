# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import logging
import copy
from pathlib import Path
from typing import Dict, List, OrderedDict, Tuple

import ray
import torch
import numpy as np
from tqdm import tqdm

import src.nsga2 as nsga2
from src.models.utils import Sampler
from src.models.spos import SPOSMixin
from src.datasets.utils import get_dataset_class_from_string, load_global_dataset, get_transforms

logger = logging.getLogger("fedoras")

class Search():
    def __init__(self, supernet: SPOSMixin, cost_mat: List[List], fix_costs: float, brackets: List[Tuple],
                 iterations: List[int], dataset: str, best_metric: str, is_max_metric: bool, task_per_tier: List[int],
                 nsga2_settings: Dict=None):
        self.sampler = Sampler(cost_mat, fix_costs, client_budget=float('inf'))
        self.supernet = supernet
        self.brackets = brackets
        assert len(brackets) == len(iterations), f"Length of brackets (= #clusters) and length of iterations must match. Got {len(iterations)}, expecte {len(brackets)}"
        self.iterations = iterations # the search budget: a list of integers indicating how many models are considered during search for each bucket (i.e. how many models are extracted from the supernet and evaluated)
        self.dataset = dataset
        self.best_metric = best_metric # metric to determnine goodness of a model
        self.is_max_metric = is_max_metric # if true, best model maximises `best_metric`
        self.task_per_tier = task_per_tier # if multi-task setting, else None
        self.nsga2_cfg = nsga2_settings

        self.cache = [] # keeps track of all considered models (so we don't repeat models)
        self.path_buckets = [OrderedDict() for _ in range(len(self.brackets))] # will store the paths (i.e. op idx) of each model sampled falling in a given bracket. It also stores their validation acc/loss on the global validation set

    def _prep_search(self, global_dir: Path, eval_fn: Dict, task_id: int=None):
        """Returns validation dataloader and function to be run for each model during search."""

        datasetclass = get_dataset_class_from_string(self.dataset)
        eval_func = eval_fn.eval_cfg['func']
        num_classes = eval_fn['num_classes'] if task_id is None else eval_fn['num_classes'][task_id]
        print(f"{num_classes = }")
        _, val_loader, _ = load_global_dataset(datasetclass,
                                               globaldata_path=global_dir,
                                               batch_size=eval_fn['batch_size'],
                                               num_workers=eval_fn['num_workers'],
                                               num_classes=num_classes,
                                               transforms=get_transforms(datasetclass, None),
                                               include_test=False,
                                               val_sample_ratio=eval_fn["val_global_sample_ratio"])
        return eval_func, val_loader

    def _path_to_str(self, path: List[int]):
        return "".join([str(d) for d in path])

    def _is_flops_in_bracket(self, flops:float, bucket_id: int):
        return self.brackets[bucket_id][0] < flops <= self.brackets[bucket_id][1]

    def _get_valid_path(self, br_low, br_high, bucket_id):
        """Returns a valid path (i.e. falls within the limits of the id-th bucket)
        along the supernet. It also returns the flops of a model along such path"""

        self.sampler.budget = br_high # set upper limit for sampler
        while not(len(self.path_buckets[bucket_id]) == self.iterations[bucket_id]): # while bucket is not full

            # sample a path along the supernet
            path = self.sampler.sample(track_paths=False)
            str_path = self._path_to_str(path)
            model_flops = self.sampler.get_last_flops()

            if str_path not in self.path_buckets[bucket_id].keys() and br_low < model_flops <= br_high:
                break

        return path, str_path, model_flops

    def get_best_models(self):
        """Returns the best model according to `self.best_metric` in each bucket. If `self.is_max_metric`
        set to True, the ranking of models in a bucket is done in descent order according to `metric`
        (e.g. for accuracy), if set to False, it will be in ascendant order (e.g. when using loss or
        perplexity as metric)"""

        # Sort buckets
        for bckt_id, bckt in enumerate(self.path_buckets):
            self.path_buckets[bckt_id] = OrderedDict(sorted(bckt.items(), key=lambda t: t[1][self.best_metric], reverse=self.is_max_metric))

        # Extract first element (the best model) in each bucket
        best_models = [next(iter(bucket.values())) for bucket in self.path_buckets]

        logger.info(f"Best models for each Tier according to metric: {self.best_metric}")
        for i, m in enumerate(best_models):
            logger.info(f"Tier {i} --> {m}")

        # Include model realisation
        for best_m in best_models:
            best_m['model'] = self.supernet.realise(best_m['path'])


        return best_models

    def _eval_models_in_bucket_with_ray(self, bckt, eval_fn, eval_func, val_loader, task_id, use_tqdm: bool=True):
        """Evaluates models in a bucket using Ray"""

        @ray.remote(num_cpus=4, num_gpus=0.5)
        def remote_eval_func(path: List[int], eval_funcc, supernet, remote_val_loader, path_str: str, task_id: int):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = supernet.realise(path)
            if task_id is not None:
                model.set_task_id(task_id)
            model.to(device)
            # evaluate validation set
            res = eval_funcc(model, remote_val_loader, device, {**eval_fn['eval_cfg'], 'epochs':1}, round=None, cid=None, search_stage=True)
            res['path_str'] = path_str
            return res

        eval_results = []
        supernet_ref = ray.put(self.supernet)
        val_loader_ref = ray.put(val_loader)
        for path_str, path_metrics in bckt.items():
            # submit Ray tasks, one per path to be evaluated
            eval_results.append(remote_eval_func.remote(path_metrics['path'], eval_func, supernet_ref, val_loader_ref, path_str, task_id))

        # Wait until Tasks are completed
        done_tasks_refs = []
        with tqdm(total=len(eval_results), desc=f"SEARCH STAGE", disable=not(use_tqdm)) as t:
            while len(eval_results):
                done_ref, eval_results = ray.wait(eval_results) # waits until one taks is completed
                done_tasks_refs.append(done_ref[0])
                t.update(1)

        return done_tasks_refs


class RandomSearch(Search):
    """Randomly samples (unique) models from the supernet and evaluates them. """

    def search(self, global_dir: Path, eval_fn: Dict):
        """Performs search. First, valid paths along the supernet are
        sampled for each tier/cluster. Then these are evaluated as Ray Tasks."""

        if self.task_per_tier is None:
            # if not a multi-task experiment, we can generate the validation loader once
            eval_func, val_loader = self._prep_search(global_dir, eval_fn)

        # generate paths in advance for each bucket
        logger.info("Generating paths...")
        for i, (br_low, br_high) in enumerate(self.brackets):
            while len(self.path_buckets[i]) < self.iterations[i]:
                path, str_path, model_flops = self._get_valid_path(br_low, br_high, i)
                # add to bucket
                self.path_buckets[i][str_path] = {'model_flops': model_flops, 'path': path, 'idx': len(self.path_buckets[i].values()), 'bucket': i}

        # Launch Ray Tasks
        for i, bckt in enumerate(self.path_buckets):

            if self.task_per_tier is not None:
                # if a multi-task experiment, get the validation loader relevant for this Tier/task
                eval_func, val_loader = self._prep_search(global_dir, eval_fn, task_id=self.task_per_tier[i])

            task_id = None if self.task_per_tier is None else self.task_per_tier[i]
            logger.info(f"Submitting {len(bckt.items())} Ray Tasks for bucket #{i}")
            done_tasks_refs = self._eval_models_in_bucket_with_ray(bckt, eval_fn, eval_func, val_loader, task_id)

            # Retrieve results and record
            results = ray.get(done_tasks_refs)
            for r in results:
                path_str = r['path_str']
                bckt[path_str] = {**bckt[path_str], **r}
                logger.info(bckt[path_str])

            logger.info(f"Finished evaluating {len(results)} paths for bucket #{i}")


class NSGA2Search(Search):

    def _split_valid_invalid(self, pop: OrderedDict, bucket_i: int):
        """Given a population, this method splits it into two dictionaries: one valid, meaning that the FLOPs
        fall inside the bracket of the current buckt/tier being considered; and invalid if otherwise"""

        # flag models as valid/invalid
        paths_to_discard = [k for k, v in pop.items() if not(self._is_flops_in_bracket(v['model_flops'], bucket_id=bucket_i)) or v['path'] in self.cache]
        pop_invalid = OrderedDict()
        for p in paths_to_discard:
            pop_invalid[p] = pop.pop(p)

        return pop, pop_invalid

    def _score_invalid(self, invalid_pop: OrderedDict, bucket_i: int):
        """Invalid models are not evaluated but still need to have a reward assigned for NSGA-II too work. We assign
        it to be a function of how far from the FLOPs bracket this model is"""

        br_low, br_high = self.brackets[bucket_i]
        for k, v in invalid_pop.items():
            flops = v['model_flops']
            invalid_pop[k][self.best_metric] = -(br_high + (br_low - flops)) if flops < br_low else br_high - flops

    def _record_valid(self, valid_pop: OrderedDict, bucket_i: int):

        for p_k, p_v in valid_pop.items():
            val = copy.deepcopy(p_v)
            self.path_buckets[bucket_i][p_k] = val

    def _adjust_valid(self, valid_pop: OrderedDict):
        """NSGA-II will try to maximise both metrics (FLOPs and self.best_metric). We therefor need to ensure that
        the sign of the best_metric matches this objective. For example, in the case of perplexity lower is better,
        so we need to flip. """
        for k, v in valid_pop.items():
            if not(self.is_max_metric):
                valid_pop[k][self.best_metric] *= -1

    def search(self, global_dir: Path, eval_fn: Dict):

        pop_size = self.nsga2_cfg['pool_size']
        mutation_prob = self.nsga2_cfg['mutation_prob']
        sample_size = self.nsga2_cfg["sample_size"]
        num_ops_in_layers = [len(ops) for ops in self.sampler.cost_matrix]

        if self.task_per_tier is None:
            # if not a multi-task experiment, we can generate the validation loader once
            eval_func, val_loader = self._prep_search(global_dir, eval_fn)

        for i, (br_low, br_high) in enumerate(self.brackets): # for each tier

            # New tier/bucket --> erase cache
            self.cache = []

            if self.task_per_tier is not None:
                # if a multi-task experiment, get the validation loader relevant for this Tier/task
                eval_func, val_loader = self._prep_search(global_dir, eval_fn, task_id=self.task_per_tier[i])

            population = OrderedDict()

            # generate initial population
            while len(population) < pop_size:
                path, str_path, model_flops = self._get_valid_path(br_low, br_high, i)
                if path not in self.cache:
                    population[str_path] = {'path': path, 'idx': len(population), 'bucket': i, 'model_flops': model_flops, self.best_metric: -1} # negative flops since pareto logic assumes higher is better
                    self.cache.append(path) # record to cache

            task_id = None if self.task_per_tier is None else self.task_per_tier[i]

            logger.info(f"Submitting Initial Population (all valid): {len(population)} Ray Tasks for Tier #{i}")
            done_tasks_refs = self._eval_models_in_bucket_with_ray(population, eval_fn, eval_func, val_loader, task_id)

            # Retrieve results and record
            results = ray.get(done_tasks_refs)
            for r in results:
                population[r['path_str']][self.best_metric] = r[self.best_metric]
                logger.info(population[r['path_str']])

            # record valid
            self._record_valid(population, bucket_i=i)
            self._adjust_valid(population)

            # Now NSGA-II actually begins
            while len(self.path_buckets[i]) < self.iterations[i]:

                # prepare metrics matrix: construct a PxN matris wher P is the population size and N is the number of features/metrics to consider (e.g. flops and validation_accuracy)
                # In our implementation of NSGA-II we focus on the best_metric of interest only (so we set FLOPs of all paths to be the same value)
                metrics = np.array([[path[self.best_metric], float('inf')] for path in population.values()])

                # Select parents that will be used to generate offspring based on Pareto fronts
                parents = nsga2.select_parents(metrics, N=sample_size)

                # discard unselect paths from population
                paths_to_discard = [path for p_i, path in enumerate(population.keys()) if p_i not in parents]
                for num_deleted, path in enumerate(paths_to_discard):
                    if pop_size - num_deleted + sample_size <= sample_size:
                        break
                    population.pop(path)

                # cross over and mutation
                genes = np.array([path['path'] for path in population.values()])
                _, genes = nsga2.crossover(genes, self.cache, num_ops_in_layers)

                # randomly mutate some genes
                genes = nsga2.mutate(genes, num_ops_in_layers, prob=mutation_prob)

                # construct new population and flag them as valid for now
                new_population = OrderedDict()
                for genes_ in genes:
                    new_population[self._path_to_str(genes_)] = {'path': genes_.tolist(), 'idx': len(self.path_buckets[i]) + len(new_population), 'bucket': i,
                                                                 'model_flops': self.sampler.count_flops_in_path(genes_.tolist()), self.best_metric: -1}
                total_new_pop = len(new_population)

                # flag models as valid/invalid
                new_population, new_population_invalid = self._split_valid_invalid(new_population, bucket_i=i)

                # evaluate VALID new_population
                if len(new_population) > 0:
                    logger.info(f"Submitting {len(new_population)}/{total_new_pop} Ray Tasks for Tier #{i} -- ({len(new_population_invalid)} are invalid) --- progress: {len(self.path_buckets[i].keys())}/{self.iterations[i]}")
                    done_tasks_refs = self._eval_models_in_bucket_with_ray(new_population, eval_fn, eval_func, val_loader, task_id)

                    # Retrieve VALID results and record
                    results = ray.get(done_tasks_refs)
                    for r in results:
                        new_population[r['path_str']][self.best_metric] = r[self.best_metric]
                        logger.info(new_population[r['path_str']])

                # Now we deal with INVALID models
                self._score_invalid(new_population_invalid, bucket_i=i)

                # record valid
                self._record_valid(new_population, bucket_i=i)
                self._adjust_valid(new_population)

                # fuse population (i.e. this rounds' parents) and new_population both VALID and INVALID
                new_population.update(new_population_invalid)
                population.update(new_population)

                # update cache
                for values in new_population.values():
                    path = values['path']
                    # assert path not in self.cache, "Model found in cache, this shouldn't happen :("
                    if path not in self.cache:
                        self.cache.append(path)

            # completed all NSGA-II iterations for this bucket/tier.
