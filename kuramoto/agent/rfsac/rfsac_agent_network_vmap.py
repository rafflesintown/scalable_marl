import copy
import torch
from torch import nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np
from repr_control.utils import util_network

from repr_control.utils.util import unpack_batch
from repr_control.agent.sac.sac_agent_network_vmap import SACAgent


from torch.func import stack_module_state
from torch.func import functional_call
import copy
from torch import vmap

from torchviz import make_dot

from torch.distributions import Normal
import math


class Critic(nn.Module):
    """
    Critic with random fourier features
    """

    def __init__(
        self,
        feature_dim,
        num_noise=20,
        hidden_dim=256,
    ):
        super().__init__()
        self.num_noise = num_noise
        self.noise = torch.randn(
            [self.num_noise, feature_dim], requires_grad=False, device=device
        )

        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, mean, log_std):
        """ """
        std = log_std.exp()
        batch_size, d = mean.shape

        x = mean[:, None, :] + std[:, None, :] * self.noise
        x = x.reshape(-1, d)

        q1 = F.elu(self.l1(x))  # F.relu(self.l1(x))
        q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q1 = F.elu(self.l2(q1))  # F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.elu(self.l4(x))  # F.relu(self.l4(x))
        q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q2 = F.elu(self.l5(q2))  # F.relu(self.l5(q2))
        # q2 = self.l3(q2) #is this wrong?
        q2 = self.l6(q2)

        return q1, q2


class RLNetwork(nn.Module):
    """
    An abstract class for neural networks in reinforcement learning (RL). In deep RL, many algorithms
    use DP algorithms. For example, DQN uses two neural networks: a main neural network and a target neural network.
    Parameters of a main neural network is periodically copied to a target neural network. This RLNetwork has a
    method called soft_update that implements this copying.
    """

    def __init__(self):
        super(RLNetwork, self).__init__()
        self.layers = []

    def forward(self, *x):
        return x

    def soft_update(self, target_nn: nn.Module, update_rate: float):
        """
        Update the parameters of the neural network by
            params1 = self.parameters()
            params2 = target_nn.parameters()

            for p1, p2 in zip(params1, params2):
                new_params = update_rate * p1.data + (1. - update_rate) * p2.data
                p1.data.copy_(new_params)

        :param target_nn:   DDPGActor used as explained above
        :param update_rate: update_rate used as explained above
        """

        params1 = self.parameters()
        params2 = target_nn.parameters()

        # bug?
        for p1, p2 in zip(params1, params2):
            new_params = update_rate * p1.data + (1.0 - update_rate) * p2.data
            p1.data.copy_(new_params)

    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# currently hardcoding s_dim
# this is a  V function
class RFVCritic(RLNetwork):
    def __init__(
        self, s_dim=3, embedding_dim=-1, rf_num=256, sigma=0.0, learn_rf=False, **kwargs
    ):
        super().__init__()
        self.n_layers = 1
        self.feature_dim = rf_num

        self.sigma = sigma

        if embedding_dim != -1:
            self.embed = nn.Linear(s_dim, embedding_dim)
        else:  # we don't add embed in this case
            embedding_dim = s_dim
            self.embed = nn.Linear(s_dim, s_dim)
            init.eye_(self.embed.weight)
            init.zeros_(self.embed.bias)
            self.embed.weight.requires_grad = False
            self.embed.bias.requires_grad = False

        fourier_feats1 = nn.Linear(embedding_dim, self.feature_dim)

        if self.sigma > 0:
            init.normal_(fourier_feats1.weight, std=1.0 / self.sigma)
            # pass
        else:
            init.normal_(fourier_feats1.weight)
        init.uniform_(fourier_feats1.bias, 0, 2 * np.pi)
        fourier_feats1.weight.requires_grad = learn_rf
        fourier_feats1.bias.requires_grad = learn_rf
        self.fourier1 = fourier_feats1  # unnormalized, no cosine/sine yet

        fourier_feats2 = nn.Linear(embedding_dim, self.feature_dim)

        if self.sigma > 0:
            init.normal_(fourier_feats2.weight, std=1.0 / self.sigma)
            # pass
        else:
            init.normal_(fourier_feats2.weight)
        init.uniform_(fourier_feats2.bias, 0, 2 * np.pi)
        fourier_feats2.weight.requires_grad = learn_rf
        fourier_feats2.bias.requires_grad = learn_rf
        self.fourier2 = fourier_feats2

        layer1 = nn.Linear(self.feature_dim, 1)  # try default scaling

        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False  # weight is the only thing we update
        self.output1 = layer1

        layer2 = nn.Linear(self.feature_dim, 1)  # try default scaling
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False  # weight is the only thing we update
        self.output2 = layer2

        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm1.bias.requires_grad = False

        self.norm2 = nn.LayerNorm(self.feature_dim)
        self.norm2.bias.requires_grad = False

    def forward(self, states: torch.Tensor):
        x = states
        x = self.embed(x)  # use an embedding layer

        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        # change to layer norm
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        return x1, x2, self.output1(x1), self.output2(x2)

    def get_norm(self):
        l1_norm = torch.norm(self.output1)
        l2_norm = torch.norm(self.output2)
        return (l1_norm, l2_norm)


class RFVCritic_no_layer_norm(RLNetwork):
    def __init__(
        self, s_dim=3, embedding_dim=-1, rf_num=256, sigma=0.0, learn_rf=False, **kwargs
    ):
        super().__init__()
        self.n_layers = 1
        self.feature_dim = rf_num

        self.sigma = sigma

        if embedding_dim != -1:
            self.embed = nn.Linear(s_dim, embedding_dim)
        else:  # we don't add embed in this case
            embedding_dim = s_dim
            self.embed = nn.Linear(s_dim, s_dim)
            init.eye_(self.embed.weight)
            init.zeros_(self.embed.bias)
            self.embed.weight.requires_grad = False
            self.embed.bias.requires_grad = False

        # fourier_feats1 = nn.Linear(sa_dim, n_neurons)
        fourier_feats1 = nn.Linear(embedding_dim, self.feature_dim)
        # fourier_feats1 = nn.Linear(s_dim,n_neurons)
        if self.sigma > 0:
            init.normal_(fourier_feats1.weight, std=1.0 / self.sigma)
            # pass
        else:
            init.normal_(fourier_feats1.weight)
        init.uniform_(fourier_feats1.bias, 0, 2 * np.pi)
        # init.zeros_(fourier_feats.bias)
        fourier_feats1.weight.requires_grad = learn_rf
        fourier_feats1.bias.requires_grad = learn_rf
        self.fourier1 = fourier_feats1  # unnormalized, no cosine/sine yet

        fourier_feats2 = nn.Linear(embedding_dim, self.feature_dim)
        # fourier_feats2 = nn.Linear(s_dim,n_neurons)
        if self.sigma > 0:
            init.normal_(fourier_feats2.weight, std=1.0 / self.sigma)
            # pass
        else:
            init.normal_(fourier_feats2.weight)
        init.uniform_(fourier_feats2.bias, 0, 2 * np.pi)
        fourier_feats2.weight.requires_grad = learn_rf
        fourier_feats2.bias.requires_grad = learn_rf
        self.fourier2 = fourier_feats2

        layer1 = nn.Linear(self.feature_dim, 1)  # try default scaling
        # init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False  # weight is the only thing we update
        self.output1 = layer1

        layer2 = nn.Linear(self.feature_dim, 1)  # try default scaling
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False  # weight is the only thing we update
        self.output2 = layer2

    def forward(self, states: torch.Tensor):
        x = states
        x = self.embed(x)  # use an embedding layer
        # print("x embedding norm", torch.linalg.norm(x))
        # x = F.relu(x)
        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        x1 = torch.cos(x1)
        x2 = torch.cos(x2)

        return self.output1(x1), self.output2(x2)
        # return self.output1(x1),self.output1(x1) #just testing if the min actually helps

    def get_phi(self, states: torch.Tensor):
        x = states
        x = self.embed(x)  # use an embedding layer
        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        x1 = torch.cos(x1)
        x2 = torch.cos(x2)

        return x1, x2
        # return self.output1(x1),self.output1(x1) #just testing if the min actually helps

    def get_norm(self):
        l1_norm = torch.norm(self.output1)
        l2_norm = torch.norm(self.output2)
        return (l1_norm, l2_norm)


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(util.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2


class CustomModelRFSACAgent(SACAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        action_range,
        dynamics_fn,
        rewards_fn,
        lr=3e-4,
        discount=0.99,
        target_update_period=2,
        tau=0.005,
        alpha=0.1,
        auto_entropy_tuning=True,
        hidden_dim=256,
        sigma=0.0,
        rf_num=256,
        learn_rf=False,
        use_nystrom=False,
        replay_buffer=None,
        device="cpu",
        N=1,
        eval_adjacency=None,
        policy_adjacency=None,
        kappa_obs_dim=1,
        eval_kappa_obs_dim=1,
        use_layer_norm=True,
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_range=action_range,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            device=device,
            N=N,
            policy_adjacency=policy_adjacency,
            kappa_obs_dim=kappa_obs_dim,
            **kwargs
        )
        self.lr = lr
        self.policy_adjacency = policy_adjacency
        self.eval_adjacency = eval_adjacency

        self.action_low = torch.tensor(action_range[0], device=self.device)
        self.action_high = torch.tensor(action_range[1], device=self.device)

        print("learn_rf", learn_rf)

        if use_nystrom == False:  # use RF
            if use_layer_norm == True:
                self.critics = [
                    RFVCritic(
                        s_dim=eval_kappa_obs_dim,
                        sigma=sigma,
                        rf_num=rf_num,
                        learn_rf=learn_rf,
                        **kwargs
                    ).to(self.device)
                    for i in range(N)
                ]
            else:
                self.critics = [
                    RFVCritic_no_layer_norm(
                        s_dim=eval_kappa_obs_dim,
                        sigma=sigma,
                        rf_num=rf_num,
                        learn_rf=learn_rf,
                        **kwargs
                    ).to(self.device)
                    for i in range(N)
                ]
        else:  # use nystrom
            feat_num = rf_num
            self.critic = nystromVCritic(
                sigma=sigma,
                feat_num=feat_num,
                buffer=replay_buffer,
                learn_rf=learn_rf,
                **kwargs
            ).to(self.device)

        self.base_model_critic = copy.deepcopy(self.critics[0])
        self.base_model_critic = self.base_model_critic.to("meta")
        self.critic_targets = copy.deepcopy(self.critics)
        all_critic_params = []
        for i in range(self.N):
            all_critic_params.extend(self.critics[i].parameters())
        self.critic_optimizer = torch.optim.Adam(
            all_critic_params, lr=lr, betas=[0.9, 0.999]
        )
        self.args = kwargs
        self.dynamics = dynamics_fn
        self.reward_fn = rewards_fn

    def get_reward(self, state, action):
        reward = self.reward_fn(state, action)
        # return torch.reshape(reward, (reward.shape[0], 1))
        return torch.reshape(reward, (reward.shape[0], -1))

    def fmodel_critic(self, params, buffers, x):
        return functional_call(self.base_model_critic, (params, buffers), (x,))

    def lstsq_fn(self, A, b, tol=1e-5):
        return torch.linalg.lstsq(A, b).solution

    def rescale_action(self, actions):
        """
        rescale action from [-1, 1] to action range.
        """
        result = (
            self.action_low.unsqueeze(dim=0)
            + (self.action_high - self.action_low).unsqueeze(dim=0)
            * (actions + 1)
            * 0.5
        )
        return result.to(torch.float32)  # make everything float32

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        # dist = self.actor(batch.state)
        obs = batch.state
        local_states_concat = self.get_local_states(obs)
        actor_params, actor_buffers = stack_module_state(self.actors)

        actor_params_need_grad_keys = []
        for key in actor_params.keys():
            if actor_params[key].requires_grad == True:
                actor_params_need_grad_keys += [key]

        new_actor_optimizer = torch.optim.Adam(
            [actor_params[key] for key in actor_params_need_grad_keys],
            lr=self.lr,
            betas=[0.9, 0.999],
        )

        mu_vmap, std_vmap = vmap(self.fmodel_actor, in_dims=(0, 0, 1), out_dims=1)(
            actor_params, actor_buffers, local_states_concat
        )
        noise = torch.randn(mu_vmap.shape).to(self.device)
        actions_vmap_og = mu_vmap + std_vmap * noise
        actions_vmap = torch.tanh(actions_vmap_og)
        norm_dist = Normal(loc=0.0, scale=1.0)
        # this is a more numerically stable version of log abs determinant for tanh
        # see https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        log_prob_vmap = norm_dist.log_prob(noise) - 2.0 * (
            math.log(2.0) - actions_vmap_og - F.softplus(-2.0 * actions_vmap_og)
        )
        log_prob_vmap = torch.reshape(log_prob_vmap, (log_prob_vmap.shape[0], self.N))
        action = torch.reshape(actions_vmap, (obs.size(0), -1))
        reward = self.get_reward(batch.state, action)  # use reward in q-fn

        next_state_noiseless_concat = self.get_local_states_critic(
            self.dynamics(batch.state, self.rescale_action(action))
        )
        critic_params, critic_buffers = stack_module_state(self.critics)
        phi1_vmap, phi2_vmap, q1_vmap, q2_vmap = vmap(
            self.fmodel_critic, in_dims=(0, 0, 1), out_dims=1
        )(critic_params, critic_buffers, next_state_noiseless_concat)

        q1_vmap = torch.reshape(q1_vmap, (q1_vmap.shape[0], self.N))
        q2_vmap = torch.reshape(q2_vmap, (q2_vmap.shape[0], self.N))

        q = self.discount * torch.min(q1_vmap, q2_vmap) + reward

        # actor_losses = (self.alphas/self.N * log_prob_vmap - q).mean(dim = 0)
        actor_losses = (self.alphas * log_prob_vmap - q).mean(dim=0)
        actor_loss = torch.sum(actor_losses)  # sum up the N (avg) losses

        self.actor_optimizer.zero_grad()
        new_actor_optimizer.zero_grad()
        actor_loss.backward()

        for i in range(self.N):
            for name, param in self.actors[i].named_parameters():
                if param.requires_grad == True:
                    param.grad = actor_params[name].grad[i].clone()
        self.actor_optimizer.step()

        info = {"actor_loss": actor_loss.item()}
        # print("actor loss", actor_loss)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_losses = (
                self.alphas * (-log_prob_vmap - self.target_entropy).detach()
            ).mean(dim=0)
            alpha_loss = torch.sum(alpha_losses)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            alphas_updated = self.alphas
            info["alpha_losses 1"] = alpha_losses[0]
            info["alpha 1"] = alphas_updated[0]  # this is the alphas before the update

        return info

    def critic_step(self, batch):
        """
        Critic update step
        """

        state, action, next_state, reward, done = unpack_batch(batch)

        critic_params, critic_buffers = stack_module_state(
            self.critics
        )  # don't no grad this!!
        with torch.no_grad():
            critic_target_params, critic_target_buffers = stack_module_state(
                self.critic_targets
            )  # don't no grad this!!
            next_state = batch.next_state
            local_next_states_concat = self.get_local_states(next_state)
            actor_params, actor_buffers = stack_module_state(self.actors)

            mu_vmap, std_vmap = vmap(self.fmodel_actor, in_dims=(0, 0, 1), out_dims=1)(
                actor_params, actor_buffers, local_next_states_concat
            )
            noise = torch.randn(mu_vmap.shape).to(self.device)
            next_actions_vmap_og = mu_vmap + std_vmap * noise
            next_actions_vmap = torch.tanh(next_actions_vmap_og)

            norm_dist = Normal(loc=0.0, scale=1.0)
            next_log_prob_vmap = norm_dist.log_prob(noise) - 2.0 * (
                math.log(2.0)
                - next_actions_vmap_og
                - F.softplus(-2.0 * next_actions_vmap_og)
            )

            next_action = torch.reshape(next_actions_vmap, (next_state.size(0), -1))

            next_next_state_noiseless_concat = self.get_local_states_critic(
                self.dynamics(batch.next_state, self.rescale_action(next_action))
            )

            _, _, next_q1, next_q2 = vmap(
                self.fmodel_critic, in_dims=(0, 0, 1), out_dims=1
            )(
                critic_target_params,
                critic_target_buffers,
                next_next_state_noiseless_concat,
            )

            next_q1 = torch.reshape(next_q1, (next_q1.shape[0], self.N))
            next_q2 = torch.reshape(next_q2, (next_q2.shape[0], self.N))
            next_action_log_pi = torch.reshape(
                next_log_prob_vmap, (batch.next_state.size(0), -1)
            )

            next_q = (
                self.discount * torch.min(next_q1, next_q2)
                - self.alphas * next_action_log_pi
            )

            next_reward = self.get_reward(
                next_state, self.rescale_action(next_action)
            )  # reward for new s,a
            target_q = next_reward + (1.0 - done) * self.discount * next_q

        critic_params_need_grad_keys = []
        for key in critic_params.keys():
            if critic_params[key].requires_grad == True:
                critic_params_need_grad_keys += [key]

        new_critic_optimizer = torch.optim.Adam(
            [critic_params[key] for key in critic_params_need_grad_keys],
            lr=self.lr,
            betas=[0.9, 0.999],
        )

        with torch.no_grad():
            next_state_noiseless_concat = self.get_local_states_critic(
                self.dynamics(state, self.rescale_action(action))
            )

        phi1, phi2, q1, q2 = vmap(self.fmodel_critic, in_dims=(0, 0, 1), out_dims=1)(
            critic_params, critic_buffers, next_state_noiseless_concat
        )
        q1 = torch.reshape(q1, (q1.shape[0], self.N))
        q2 = torch.reshape(q2, (q2.shape[0], self.N))

        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        new_critic_optimizer.zero_grad()
        q_loss.backward()
        for i in range(self.N):
            for name, param in self.critics[i].named_parameters():
                if param.requires_grad == True:
                    param.grad = critic_params[name].grad[i].clone()
        self.critic_optimizer.step()
        # print("self critic state", self.critic_optimizer.state_dict())

        info = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q1": q1.mean().item(),
            "q2": q2.mean().item(),
        }

        # maybe sth wrong this code
        # print("q1 shape", q1.shape)
        dist = {
            "td_error (0)": (torch.min(q1[:, 0], q2[:, 0]) - target_q[:, 0])
            .cpu()
            .detach()
            .clone()
            .numpy(),
            "q (0)": torch.min(q1[:, 0], q2[:, 0]).cpu().detach().clone().numpy(),
        }

        info.update({"critic_dist": dist})

        return info

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        batch = buffer.sample(batch_size)

        # Acritic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **critic_info,
            **actor_info,
        }
