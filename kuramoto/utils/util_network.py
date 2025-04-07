import time

# import gym
import numpy as np
import torch.nn.functional as F

# import torch

from torch import nn
from torch import vmap
import os
import pickle as pkl


from torch.func import functional_call
from torch.func import stack_module_state
from torch import vmap


def unpack_batchv2(batch):
    return (
        batch.state,
        batch.action,
        batch.rewards,
        batch.next_state,
        batch.next_action,
        batch.next_reward,
        batch.next_next_state,
        batch.done,
    )


def unpack_batch(batch):
    return batch.state, batch.action, batch.next_state, batch.reward, batch.done


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def reset(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def set_step(self, step):
        self._step = step
        self._step_time = time.time()

    def time_cost(self):
        return time.time() - self._start_time

    def steps_per_sec(self, step):
        sps = (step - self._step) / (time.time() - self._step_time)
        self._step = step
        self._step_time = time.time()
        return sps


def eval_policy(policy, eval_env, eval_episodes=100, render=False, seed=0):
    """
    Eval a policy
    """
    ep_rets = []
    avg_len = 0.0
    for i in range(eval_episodes):
        ep_ret = 0.0
        # eval_env.seed(i)
        state, _ = eval_env.reset(seed=i + seed)
        done = False
        # print("eval_policy state", state)
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
            avg_len += 1
            if render:
                eval_env.render()
        ep_rets.append(ep_ret)

    avg_ret = np.mean(ep_rets)
    std_ret = np.std(ep_rets)
    avg_len /= eval_episodes

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: avg eplen {avg_len}, avg return {avg_ret:.3f} $\pm$ {std_ret:.3f}"
    )
    print("---------------------------------------")
    return avg_len, avg_ret, std_ret, ep_rets


def batch_eval(policy, eval_env, seed=0):
    import torch

    avg_len = 0.0
    ep_ret = torch.zeros((eval_env.sample_batch_size, 1), device=eval_env.device)
    # eval_env.seed(i)
    state, _ = eval_env.reset(seed=seed)
    done = False
    # print("eval_policy state", state)
    while not done:
        # action = policy.batch_select_action(state)
        action = policy.batch_select_action_network(state)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        # ep_ret += reward
        # print("ep ret shape", ep_ret.shape)
        # print("reward shape", reward.shape)
        # print("reward mean shape", reward.mean(dim = 1).shape)
        # print("reward[0].mean", reward[0].mean())
        ep_ret += reward.mean(dim=1, keepdims=True)

    avg_ret = ep_ret.mean().item()
    std_ret = ep_ret.std().item()

    print("---------------------------------------")
    print(f"Evaluation avg return {avg_ret:.3f} $\pm$ {std_ret:.3f}")
    print("---------------------------------------")
    return None, avg_ret, std_ret, ep_ret


def batch_eval_discounted(agent, eval_env, seed=0, return_features=True):
    import torch

    avg_len = 0.0
    V_ret = torch.zeros((eval_env.sample_batch_size, agent.N), device=eval_env.device)
    # eval_env.seed(i)
    state, _ = eval_env.reset(seed=seed)
    state_concat = agent.get_local_states_critic(state)
    critic_params, critic_buffers = stack_module_state(agent.critic_targets)
    if return_features == True:
        phi1_vmap, phi2_vmap, v1_vmap, v2_vmap = vmap(
            agent.fmodel_critic, in_dims=(0, 0, 1), out_dims=1
        )(critic_params, critic_buffers, state_concat)
    else:
        v1_vmap, v2_vmap = vmap(agent.fmodel_critic, in_dims=(0, 0, 1), out_dims=1)(
            critic_params, critic_buffers, state_concat
        )
    done = False
    # print("v1 shape", v1_vmap.shape)
    t = 0
    while not done:
        action = agent.batch_select_action_network(state)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        # print("reward.shape", reward.shape)
        V_ret += agent.discount**t * reward
    print("V ret shape", V_ret)
    print("v2", v2_vmap.reshape(state.size(0), agent.N))

    print("---------------------------------------")
    print(
        f"MSE of V_ret - V_pi_learned {torch.mean((V_ret - v1_vmap.reshape(state.size(0),agent.N))**2):.3f}"
    )
    print("---------------------------------------")
    return (
        None,
        V_ret,
        torch.mean((V_ret - v1_vmap.reshape(state.size(0), agent.N)) ** 2),
    )


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def clear_data(path):
    import pickle as pkl

    with open(path, "rb") as f:
        a = pkl.load(f)
        a["replay_buffer"] = None
        print(a)
    with open(path, "wb") as f:
        pkl.dump(a, f)
