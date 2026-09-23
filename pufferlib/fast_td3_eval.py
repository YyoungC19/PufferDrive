"""PufferDrive benchmark interface for original FastTD3 actor checkpoints."""

import torch.nn as nn

from pufferlib.fast_td3 import Actor
from pufferlib.fast_td3_utils import EmpiricalNormalization


class FastTD3EvalPolicy(nn.Module):
    is_continuous = True
    is_deterministic = True

    def __init__(self, checkpoint, vecenv, device):
        super().__init__()
        args = checkpoint["args"]
        n_obs = vecenv.single_observation_space.shape[0]
        n_act = vecenv.single_action_space.shape[0]
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
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if args["obs_normalization"]:
            self.obs_normalizer = EmpiricalNormalization(n_obs, device)
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        else:
            self.obs_normalizer = nn.Identity()

    def forward_eval(self, obs, state=None):
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            obs = self.obs_normalizer(obs, update=False)
        return self.actor(obs)
