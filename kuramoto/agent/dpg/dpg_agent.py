import torch
import torch.nn.functional as F
from repr_control.utils.buffer import Batch
from repr_control.agent.sac.sac_agent import ModelBasedSACAgent
from repr_control.agent.actor import DeterministicActor
from repr_control.agent.critic import DoubleQCritic
from repr_control.agent.actor import DeterministicActor
import numpy as np
from repr_control.utils import util


class DPGAgent(object):
    """
    DDPG Agent
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_range,
        lr=3e-4,
        discount=0.99,
        target_update_period=2,
        tau=0.005,
        alpha=0.1,
        auto_entropy_tuning=True,
        hidden_dim=1024,
        hidden_depth=2,
        device="cpu",
        **kwargs
    ):
        self.steps = 0

        self.device = torch.device(device)
        self.action_range = action_range
        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.learnable_temperature = auto_entropy_tuning

        # functions
        self.critic = DoubleQCritic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.critic_target = DoubleQCritic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = DeterministicActor(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        # self.log_alpha = torch.tensor(np.log(alpha)).float().to(self.device)
        # self.log_alpha.requires_grad = True
        # self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr, betas=[0.9, 0.999]
        )
        critic_lr = kwargs["critic_lr"] if "critic_lr" in kwargs.keys() else lr
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=[0.9, 0.999]
        )

    # @property
    # def alpha(self):
    # 	return self.log_alpha.exp()

    def select_action(self, state, explore=False):
        if isinstance(state, list):
            state = np.array(state)
        assert len(state.shape) == 1
        state = torch.from_numpy(state).to(self.device)
        state = state.unsqueeze(0)
        action = self.actor(state)
        # action = dist.sample() if explore else dist.mean
        action = action.clamp(
            torch.tensor(-1, device=self.device), torch.tensor(1, device=self.device)
        )
        assert action.ndim == 2 and action.shape[0] == 1
        return util.to_np(action[0])

    def update_target(self):
        if self.steps % self.target_update_period == 0:
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def critic_step(self, batch):
        """
        Critic update step
        """
        obs, action, next_obs, reward, done = util.unpack_batch(batch)
        not_done = 1.0 - done

        next_action = self.actor(next_obs)
        # = dist.rsample()
        # log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2)  #  - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": critic_loss.item(),
            "q1": current_Q1.mean().item(),
            "q2": current_Q1.mean().item(),
        }

    def update_actor_and_alpha(self, batch):
        obs = batch.state

        action = self.actor(obs)
        # action = dist.rsample()
        # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (-actor_Q).mean()  # self.alpha.detach() * log_prob

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {"actor_loss": actor_loss.item()}

        # if self.learnable_temperature:
        # 	self.log_alpha_optimizer.zero_grad()
        # 	alpha_loss = (self.alpha *
        # 				  (-log_prob - self.target_entropy).detach()).mean()
        # 	alpha_loss.backward()
        # 	self.log_alpha_optimizer.step()
        #
        # 	info['alpha_loss'] = alpha_loss
        # 	info['alpha'] = self.alpha

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


class ModelBasedDPGAgent(ModelBasedSACAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        action_range,
        dynamics,
        rewards,
        initial_distribution,
        horizon=250,
        lr=0.0003,
        discount=0.99,
        target_update_period=2,
        tau=0.005,
        alpha=0.1,
        auto_entropy_tuning=True,
        hidden_dim=1024,
        hidden_depth=2,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            state_dim,
            action_dim,
            action_range,
            dynamics,
            rewards,
            initial_distribution,
            horizon=horizon,
            lr=lr,
            discount=discount,
            target_update_period=target_update_period,
            tau=tau,
            alpha=alpha,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            device=device,
            **kwargs
        )
        self.actor = DeterministicActor(
            state_dim, action_dim, hidden_dim, hidden_depth
        ).to(self.device)

    def update_actor_and_alpha(self, batch):
        obs = batch.state
        log_probs = []
        rewards = torch.zeros([obs.shape[0]]).to(self.device)
        for i in range(self.horizon):
            action = self.actor(obs)
            # action = dist.rsample()
            # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            obs = self.dynamics(obs, action)
            if i == self.horizon - 1:
                rewards += self.rewards(obs, action, terminal=True)
            else:
                rewards += self.rewards(obs, action, terminal=False)
            # log_probs.append(log_prob)
        final_reward = self.rewards(obs, action, terminal=True)
        actor_loss = -1 * rewards.mean()
        # log_prob_all = torch.hstack(log_probs)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "terminal_cost": final_reward.mean().item(),
        }

        # if self.learnable_temperature:
        # 	self.log_alpha_optimizer.zero_grad()
        # 	alpha_loss = (self.alpha *
        # 				  (-log_prob_all - self.target_entropy).detach()).mean()
        # 	alpha_loss.backward()
        # 	self.log_alpha_optimizer.step()
        #
        # 	info['alpha_loss'] = alpha_loss.item()
        # 	info['alpha'] = self.alpha.item()

        return info

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        state = torch.from_numpy(self.initial_dist(batch_size)).float().to(self.device)
        batch = Batch(
            state=state,
            action=None,
            next_state=None,
            reward=None,
            done=None,
        )
        # Acritic step
        # critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            # **critic_info,
            **actor_info,
        }
