import torch
from torch import nn as nn

import abc
from collections import OrderedDict
from typing import Iterable

from lifelong_rl.core.rl_algorithms.batch.batch_rl_algorithm import BatchRLAlgorithm
from lifelong_rl.core.rl_algorithms.batch.mb_batch_rl_algorithm import MBBatchRLAlgorithm
from lifelong_rl.core.rl_algorithms.offline.offline_rl_algorithm import OfflineRLAlgorithm
from lifelong_rl.core.rl_algorithms.offline.offpolicy_rl_algorithm import OffpolicyRLAlgorithm
from lifelong_rl.core.rl_algorithms.offline.mb_offline_rl_algorithm import OfflineMBRLAlgorithm
from lifelong_rl.core.rl_algorithms.online.online_rl_algorithm import OnlineRLAlgorithm
from lifelong_rl.core.rl_algorithms.online.mbrl_algorithm import MBRLAlgorithm
from lifelong_rl.trainers.trainer import Trainer
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchMBRLAlgorithm(MBRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchMBBatchRLAlgorithm(MBBatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchOfflineRLAlgorithm(OfflineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

class TorchOffpolicyRLAlgorithm(OffpolicyRLAlgorithm):
    def to(self, device):
        for t in self.trainer_list:
            for net in t.networks:
                net.to(device)

    def training_mode(self, mode):
        for t in self.trainer_list:
            for net in t.networks:
                net.train(mode)

class TorchOfflineMBRLAlgorithm(OfflineMBRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch, indices, Qmin, eta):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch, indices, Qmin, eta)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    def try_update_target_networks(self):
        pass

    def get_policy(self):
        pass
    
    def add_qf_loss(self, loss):
        pass

    def get_qfs(self):
        pass
    
    def get_alpha(self):
        pass 

    def get_target_qfs(self):
        pass

    def set_qfs(self, qfs):
        pass

    def set_target_qfs(self, target_qfs):
        pass

    def train_from_torch(self, batch, indices, Qmin, eta):
        pass

    @property
    def networks(self) -> Iterable[nn.Module]:
        pass
