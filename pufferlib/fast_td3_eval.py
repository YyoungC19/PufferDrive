"""PufferDrive benchmark interface for original FastTD3 actor checkpoints."""

import torch
import torch.nn as nn

from pufferlib.fast_td3 import Acto
from pufferlib.fast_td3_utils import EmpiricalNormalization


class FastTD3EvalPolicy(nn.Module):
    is_continuous = True
    is_deterministic = True

    def __init__(self, checkpoint, vecenv, device):
        super().__init__()
        args = checkpoint["args"]
        n_obs = vecenv.single_observation_space.shape[0]
        n_act = vecenv.single_action_space.shape[0]
        driver_env = vecenv.driver_env
        self.history_frames = int(getattr(driver_env, "obs_history_frames", 0))
        self.history_features = int(getattr(driver_env, "history_features", 0))
        self.history_agents = int(getattr(driver_env, "obs_slots_partners_n", 0)) + 1
        self.history_dim = (
            self.history_agents * self.history_frames * self.history_features
        )
        if self.history_frames <= 1:
            self.history_agents = self.history_frames = self.history_features = 0
            self.history_dim = 0
        self.base_n_obs = n_obs - self.history_dim
        self.actor = Actor(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=args["num_envs"],
            init_scale=args["init_scale"],
            hidden_dim=args["actor_hidden_dim"],
            std_min=args["std_min"],
            std_max=args["std_max"],
            sim_type=args["sim_type"],
            sim_dimension=args["sim_dimension"],
            seq_len=args["actor_seq_len"],
            device=device,
            history_agents=self.history_agents,
            history_frames=self.history_frames,
            history_features=self.history_features,
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if args["obs_normalization"]:
            self.obs_normalizer = EmpiricalNormalization(self.base_n_obs, device)
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        else:
            self.obs_normalizer = nn.Identity()

    def forward_eval(self, obs, state=None):
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            normalized_base = self.obs_normalizer(
                obs[..., : self.base_n_obs], update=False
            )
            obs = torch.cat([normalized_base, obs[..., self.base_n_obs :]], dim=-1)
        return self.actor(obs)
