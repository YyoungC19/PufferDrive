"""PufferDrive benchmark interface for original FastTD3 actor checkpoints."""

import torch
import torch.nn as nn

from pufferlib.fast_td3 import Actor, RecurrentActor
from pufferlib.fast_td3_utils import EmpiricalNormalization


class FastTD3EvalPolicy(nn.Module):
    is_continuous = True
    is_deterministic = True

    def __init__(self, checkpoint, vecenv, device):
        super().__init__()
        args = checkpoint["args"]
        n_obs = vecenv.single_observation_space.shape[0]
        n_act = vecenv.single_action_space.shape[0]
        self.is_recurrent = args.get("recurrent", False)
        actor_cls = RecurrentActor if self.is_recurrent else Actor
        actor_kwargs = dict(
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
        if self.is_recurrent:
            actor_kwargs.update(
                recurrent_action_embedding_size=args["recurrent_action_embedding_size"],
                recurrent_observation_embedding_size=args["recurrent_observation_embedding_size"],
                recurrent_reward_embedding_size=args["recurrent_reward_embedding_size"],
                recurrent_hidden_size=args["recurrent_hidden_size"],
                recurrent_num_layers=args["recurrent_num_layers"],
            )
        self.actor = actor_cls(**actor_kwargs)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if args["obs_normalization"]:
            self.obs_normalizer = EmpiricalNormalization(n_obs, device)
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        else:
            self.obs_normalizer = nn.Identity()

    def forward_eval(self, obs, state=None):
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            obs = self.obs_normalizer(obs, update=False)
        if not self.is_recurrent:
            return self.actor(obs)
        if state is None:
            state = self.initial_eval_state(obs.shape[0], obs.device)
        action, hidden = self.actor(
            obs,
            prev_actions=state["prev_action"],
            rewards=state["reward"],
            hidden=state["hidden"],
            return_hidden=True,
        )
        state["hidden"].copy_(hidden.to(state["hidden"].dtype))
        state["prev_action"].copy_(action)
        return action

    def initial_eval_state(self, batch_size, device):
        return {
            "hidden": self.actor.initial_state(batch_size),
            "prev_action": torch.zeros(batch_size, self.actor.n_act, device=device),
            "reward": torch.zeros(batch_size, 1, device=device),
        }

    def update_eval_state(self, state, rewards, finished):
        count = rewards.numel()
        finished = finished.reshape(-1, 1)
        state["reward"][:count].copy_(rewards.reshape(-1, 1))
        state["hidden"][:, :count].masked_fill_(finished.unsqueeze(0), 0)
        state["prev_action"][:count].masked_fill_(finished, 0)
        state["reward"][:count].masked_fill_(finished, 0)
