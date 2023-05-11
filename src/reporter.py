# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import pickle
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter


class Reporter():
    """Use this class to record the progress of a federated learning workload. A reporter
    is used in Stage I & III. In addition to store all metrics, this class also add sto
    TensorBoard the most relevant metrics."""
    def __init__(self, exp_dir: Path):

        self.exp_dir = exp_dir
        self.__create_dirs_and_writer()

    def __create_dirs_and_writer(self):

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.exp_dir, flush_secs=60)

    def save(self, data: Dict, filename):
        """Saves `data` to a pickle."""
        with open(self.exp_dir/f"{filename}.pkl", 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(self.exp_dir/f"{filename}.pkl", 'rb') as h:
            data = pickle.load(h)
        return data

    def save_round_data(self, round_data: Dict, round: int, model=None, model_name: str='supernet'):
        """Saves round data to new sub-directory. If passed, the model is also stored"""
        round_dir = self.exp_dir/"rounds"/str(round)
        round_dir.mkdir(parents=True)
        with open(round_dir/"reporter.pkl", 'wb') as h:
            pickle.dump(round_data, h, protocol=pickle.HIGHEST_PROTOCOL)

        if model is not None:
            # save model in this round's directory
            torch.save(model.state_dict(), round_dir/f"{model_name}.pt")
            # also keep a copy in the top level dir of the experiment
            torch.save(model.state_dict(), self.exp_dir/f"{model_name}.pt")


    def add_scalars(self, scalar_data: List[Dict], step: int, prefix: str=""):
        """Adds scalars to TensorBoard"""
        for scalar, value in scalar_data.items():
            if value is not None:
                self.writer.add_scalar(tag=f"{prefix}/{scalar}" if prefix else scalar, scalar_value=value, global_step=step)
