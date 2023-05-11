# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import logging
from typing import Dict
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from src.models.spos import SPOSMixin

logger = logging.getLogger("fedoras")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() # K, B
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def epoch(model, optim, dataloader, training: bool, flops: float, device: torch.device,
          gradclip: float=None, perplexity: bool=False):
    """Trains/evaluates model on a given dataloader once."""

    total_acc = 0
    total_loss = 0
    total_perp = 0
    num_batches = 0
    examples = 0

    if training:
        model.train()
    else:
        model.eval()

    with ExitStack() as stack:
        if not training:
            stack.enter_context(torch.no_grad())

        dataloader_itr = iter(dataloader)
        for im, lab in dataloader_itr:
            examples += im.shape[0]
            im = im.to(device=device, non_blocking=True)
            lab = lab.to(device=device, non_blocking=True)
            out = model(im)

            # Accumulate FLOPs for the path used for this batch
            if isinstance(model, SPOSMixin) and training:
                sampler = model.spos_get_sampler()
                if sampler is not None:
                    flops += sampler.get_last_flops()

            loss = F.cross_entropy(out, lab)
            if training:
                optim.zero_grad()
                loss.backward()
                if gradclip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
                optim.step()

            acc = accuracy(out.detach(), lab, topk=(1,))[0]

            if perplexity:
                with torch.no_grad():
                    perp = torch.exp(loss)
                    total_perp += float(perp)

            total_loss += float(loss)
            total_acc += float(acc)
            num_batches += 1

    avg_acc = total_acc/num_batches
    avg_loss = total_loss/num_batches
    avg_perp = None

    if perplexity:
        avg_perp = total_perp/num_batches

    return avg_acc, avg_loss, avg_perp, flops, examples


def train(model, optim, trainloader, device: torch.device, train_cfg: Dict, round: int, cid: str, tier: int):
    """Trains the supernet for a number of epochs. Returns the acc/loss/perplexity metrics from the last epoch"""

    total_examples = 0
    flops_so_far = 0
    gradclip = train_cfg.get('gradclip', None)
    perplexity = train_cfg.get('perplexity', False)
    epochs = train_cfg['epochs']

    for e in range(epochs):
        avg_acc, avg_loss, avg_perp, flops_so_far, examples = epoch(model, optim, trainloader, True, flops_so_far, device, gradclip, perplexity)
        total_examples += examples
        logger.info(f"[TRAIN] Global Round: {round}, epoch: {e}, acc: {avg_acc:.2f}, loss: {avg_loss:.3f}, perp: {avg_perp}, cid: {cid} (tier: {tier}), FLOPs_so_far: {flops_so_far/1e6:.2f}M")

    return {'total_examples': total_examples, 'train_acc': avg_acc, 'train_loss': avg_loss, 'train_perplexity':avg_perp}


def eval(model, dataloader, device: torch.device, eval_cfg: Dict, round: int, cid: str, search_stage: bool=False, is_test: bool=False):
    """Evaluates the supernet for a number of epochs. Returns the averaged acc/loss/perplexity across all epochs."""
    total_examples = 0
    flops_so_far = 0
    perplexity = eval_cfg.get('perplexity', False)
    epochs = eval_cfg['epochs']
    acc = 0.0
    loss = 0.0
    perp = None
    prefix = 'test' if is_test else 'val'

    for e in range(epochs):
        avg_acc, avg_loss, avg_perp, flops_so_far, examples = epoch(model, None, dataloader, False, flops_so_far, device, None, perplexity)
        total_examples += examples
        acc += avg_acc
        loss += avg_loss
        if avg_perp is not None:
            if perp is None:
                perp = avg_perp
            else:
                perp += avg_perp
        if not(search_stage):
            logger.info(f"[EVAL] Global Round: {round}, epoch: {e}, acc: {avg_acc:.2f}, loss: {avg_loss:.3f}, perp: {avg_perp}, cid: {cid}, FLOPs_so_far: {flops_so_far/1e6:.2f}M")

    return {'total_examples': total_examples, f'{prefix}_acc': acc/epochs, f'{prefix}_loss': loss/epochs, f'{prefix}_perplexity': perp/epochs if perp is not None else perp}
