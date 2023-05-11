# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import random
from typing import List

import numpy as np


def get_pareto_fronts(P):
    ''' Yield consecutive pareto fronts of a set of points. Each front is returned as indices
        of relevant elements from ``P``.

        The first yielded pareto front is the normal pareto front, that is a subset of points
        such that each point in the subset is not pareto-dominated by any other from the full set.
        Each further front is a pareto-front after discarding all points belonging to the previous
        front.

        > **Note:** the function assumes higher values are better for all metrics.

        Arguments:
            P : a 2d numpy array with shape ``(N, M)`` where ``N`` is the number of points
                and ``M`` is the number of metrics.

        Yields:
            1d numpy arrays containing indices of points belonging to consecutive pareto-fronts
            of ``P``, starting from the first front.
    '''
    population_size = P.shape[0]

    # domination_map[i,j] == True iff P[i] dominates P[j]
    domination_map = np.zeros((population_size, population_size))
    for pidx in range(population_size):
        p = P[pidx:pidx+1] # 1, metrics
        gt = p > P
        ge = p >= P
        dominates = np.logical_and(np.any(gt, axis=1), np.all(ge, axis=1))
        domination_map[pidx] = dominates

    domination_counter = domination_map.sum(axis=0)
    F = np.argwhere(domination_counter == 0).flatten()

    while F.size:
        yield F

        removed_dominations = (domination_map[F] == 1).sum(axis=0)
        domination_counter[F] -= 1 # to prevent points from F from being selected again (make their counters -1)
        domination_counter -= removed_dominations
        F = np.argwhere(domination_counter == 0).flatten()


def crowding_distance(I, ptp=None):
    ''' Calculate crowding distance of points in ``I``.

        Arguments:
            I : a 2d numpy array with shape ``(N, M)`` where ``N`` is the number of points
                and ``M`` is the number of metrics.
            ptp : (optional) 1d numpy array with range of each metric (max-min), can be computed with ``numpy.ptp()``,
                if not provided the value will be determined from ``I`` as ``numpy.ptp(I, axis=0)``.

        Return:
            1d numpy array with shape ``(N,)`` containing crowding distance for each of the ``N`` points in ``I``.
            Points at the boundaries of each metric are assigned ``numpy.inf`` as their distance.
    '''
    points = I.shape[0]

    if ptp is None:
        ptp = np.ptp(I, axis=0)

    I /= ptp

    order = I.argsort(0)
    inv_order = np.empty_like(order)
    np.put_along_axis(inv_order, order, np.arange(points)[:,None], axis=0)
    sorted_values = np.take_along_axis(I, order, axis=0) # sorts each metric independently
    crowding = sorted_values[2:] - sorted_values[:-2] # because of sorting values in each row are not related so we can't add them yet
    crowding = np.pad(crowding, [[1,1], [0,0]], mode='constant', constant_values=np.inf)

    crowding = np.take_along_axis(crowding, inv_order, axis=0) # this should inverse sorting
    crowding = crowding.sum(axis=1)
    return crowding


def select_parents(P, N=None):
    ''' Select new parents from the population of ``P`` points based on their pareto-rank
        and crowding distance. Parents are returned as indices from ``P``.

        Arguments:
            P : a 2d numpy array with shape ``(K, M)`` where ``K`` is the number of points
                and ``M`` is the number of metrics.
            N : (optional) number of parents to select from ``P``, if not provided ``K//2``
                parents are selected.

        Return:
            1d numpy array with shape ``(N,)`` containing indices of the selected parents in ``P``.

        Raises:
            ``ValueError`` if provided ``N`` is larger than ``K``.
    '''
    if N is None:
        N = P.shape[0] // 2
    elif N > P.shape[0]:
        raise ValueError('Number of parents requested is larger than population! {} vs. {}'.format(N, P.shape[0]))
    elif N == P.shape[0]:
        return np.arange(P.shape[0])

    population_ptp = np.ptp(P, axis=0)
    total_points = 0
    PP = []
    for F in get_pareto_fronts(P):
        points = F.shape[0]
        if points + total_points > N:
            crowding = crowding_distance(P[F], population_ptp)
            sorted_F = F[crowding.argsort()]
            F = sorted_F[-(N-total_points):]
            points = F.shape[0]
            assert F.shape[0] + total_points == N

        PP.append(F)
        total_points += points
        if total_points >= N:
            break

    return np.concatenate(PP)


def crossover(P, cache: List[List[int]], genes: List[int], swap_genes: int=None):
    '''Given P parents, this function generates P offsprings by mutating genes
    of pairs of parents. By default half of the genes are swapped, unless `swap_genes`
    is specified. If a cache is porvided, the crossover guarantees that offspring is
    not in cache. `genes` is a list containing the number
    of possible values each gene in the sequence can get. The new population P||Q is returned.'''

    num_genes = P.shape[1]

    assert num_genes == len(genes), "Dimensions do not match"

    if swap_genes is None:
        swap_genes = int(num_genes // 2)

    # generate crossover pair indices
    mix_parents = sum([[(i, n) for i in range(n)] for n in range(P.shape[0])], [])
    random.shuffle(mix_parents)

    Q = [] # we'll append the new offspring

    for mix in mix_parents:
        childless = True
        max_iter = 10 # if two gene sequences do not generate a new (unseen before in PuQ) child, then repeat this many times. If exhausted, move on to the next parents pair

        while childless and max_iter>0:
            # random swapping mask
            mask = random.sample(range(num_genes), swap_genes)

            child = np.copy(P[mix[0]])
            child[mask] = P[mix[1]][mask]

            child = child.tolist()
            if cache is None:
                # we want the child is not already in P or Q
                if child not in P.tolist():
                    if child not in Q:
                        # append to new population
                        Q.append(child)
                        childless = False
            else:
                if child not in cache and child not in Q:
                    Q.append(child)
                    childless = False

            max_iter -= 1

        if len(Q) == P.shape[0]:
            Q = np.array(Q)
            # stop adding childs until Q equal to P in size
            return np.concatenate((P, Q)), Q


    # if we have exahusted mix_parents and still haven't generated enough offspring,
    # then we randomly generate the remaining children
    while len(Q) < P.shape[0]:
        new_child = [random.sample(range(num_ops), 1)[0] for num_ops in genes]
        if new_child not in cache and new_child not in Q:
            Q.append(new_child)
        else:
            cache.append(new_child)

    Q = np.array(Q)
    print(f"P shape: {P.shape}, Q shape: {Q.shape}, cache_size: {len(cache)}")
    # Concatenating P and Q
    return np.concatenate((P, Q)), Q


def mutate(P, genes: List[int], prob: float=0.01):
    '''Randomly mutates genes in the population P by replacing a gene with
    one in `genes` given a probability. Input argument `genes` is a list of
    lenght L, where L is the number of columns in P (i.e. number of layers in a
    path) and defining the number of candid '''

    num_genes = P.shape[1]
    for p_i in range(P.shape[0]):
        current_gene = np.copy(P[p_i])
        # mutation mask for each gene
        mmask = [p<(100*prob) for p in random.sample(range(100), num_genes)]

        # mutation replacement for each gene
        mrep = np.array([[random.sample(range(num_ops), 1)[0] for num_ops in genes]])

        current_gene[mmask] = mrep[0,mmask]

        # replace if the mutated is not already present somewhere else in P
        if current_gene.tolist() not in P.tolist():
            P[p_i] = current_gene

    return P
