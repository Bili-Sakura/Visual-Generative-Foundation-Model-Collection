# Timestep samplers ported from OpenAI guided-diffusion guided_diffusion/resample.py

from abc import ABC, abstractmethod

import numpy as np
import torch


def create_named_schedule_sampler(name, diffusion):
    if name == "uniform":
        return UniformSampler(diffusion)
    if name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        raise NotImplementedError

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        self.update_with_all_losses(local_ts.detach().cpu().numpy().tolist(), local_losses.detach().cpu().tolist())

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        raise NotImplementedError


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([diffusion.num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
