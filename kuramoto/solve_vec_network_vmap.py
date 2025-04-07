import argparse
import os
import pickle as pkl

from tensorboardX import SummaryWriter
from datetime import datetime
from repr_control.utils import util_network, buffer
from repr_control.agent.sac import sac_agent_network_vmap
from repr_control.agent.rfsac import rfsac_agent_network_vmap
from define_problem_kuramoto_v2_thdot import *
from gymnasium.envs.registration import register
import gymnasium
import yaml
from repr_control.utils.buffer import Batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### parameter that
    parser.add_argument(
        "--alg", default="rfsac", help="The algorithm to use. rfsac or sac."
    )
    parser.add_argument(
        "--env",
        default="custom_vec",
        help="Name your env/dynamics, only for folder names.",
    )  # Alg name (sac, vlsac)
    parser.add_argument(
        "--rf_num",
        default=2048,
        type=int,
        help="Number of random features. Suitable numbers for 2-dimensional system is 512, 3-dimensional 1024, etc.",
    )
    parser.add_argument(
        "--nystrom_sample_dim",
        default=8192,
        type=int,
        help="The sampling dimension for nystrom critic. After sampling, take the maximum rf_num eigenvectors..",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        type=str,
        help="pytorch device, cuda if you have nvidia gpu and install cuda version of pytorch. "
        "mps if you run on apple silicon, otherwise cpu.",
    )

    ### Parameters that usually don't need to be changed.
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed."
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps",
        default=0,
        type=float,
        help="the number of initial steps that collects data via random sampled actions.",
    )  # Time steps initial random policy is used
    parser.add_argument(
        "--eval_freq",
        default=100,
        type=int,
        help="number of iterations as the interval to evaluate trained policy.",
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps",
        default=1e4,
        type=float,
        help="the total training time steps / iterations.",
    )  # Max time steps to run environment
    parser.add_argument(
        "--batch_size", default=128, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=256, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument(
        "--tau", default=0.005, type=float
    )  # Target network update rate
    parser.add_argument(
        "--rf_sigma", default=0.0, type=float
    )  # Target network update rate
    # parser.add_argument("--tau", default=0.1)  # Target network update rate
    parser.add_argument(
        "--embedding_dim", default=-1, type=int
    )  # if -1, do not add embedding layer

    parser.add_argument("--use_nystrom", action="store_true")
    parser.add_argument(
        "--use_random_feature", dest="use_nystrom", action="store_false"
    )
    parser.add_argument("--use_layer_norm", action="store_false")
    # parser.add_argument("--n_agents", default = 1, type=int)
    parser.set_defaults(use_nystrom=False)
    args = parser.parse_args()

    learn_rf = False
    alg_name = args.alg
    exp_name = f'seed_{args.seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = vars(args)
    kwargs.update(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_range": action_range,
            "obs_space_high": np.clip(state_range[0], -3.0, 3.0).tolist(),
            "obs_space_low": np.clip(
                state_range[1], -3.0, 3.0
            ).tolist(),  # in case of inf observation space
            "device": curr_device,  # imported from the define_problem_*.py
            "learn_rf": learn_rf,
        }
    )

    # args from define_problem.py
    kwargs.update(
        {
            "N": N,
            "eval_adjacency": eval_adjacency,
            "policy_adjacency": policy_adjacency,
            "kappa_obs_dim": kappa_obs_dim,
            "eval_kappa_obs_dim": eval_kappa_obs_dim,
        }
    )

    # setup example_results
    log_path = f"log/{alg_name}/{env_name}/N={N}/kappa={eval_kappa}/sigma={sigma}/rf_sigma={args.rf_sigma}/rf_num={args.rf_num}/learn_rf={args.learn_rf}/batchsize={args.batch_size}/use_layer_norm={args.use_layer_norm}/tau={args.tau}/{exp_name}"
    summary_writer = SummaryWriter(log_path + "/summary_files")

    print("kwargs keys", kwargs.keys())
    print("policy adjacency", policy_adjacency)

    print("kwargs device", kwargs["device"])

    # Initialize policy
    if args.alg == "sac":
        agent = sac_agent_network.SACAgent(**kwargs)
    elif args.alg == "rfsac":
        # agent = rfsac_agent_network.CustomModelRFSACAgent(dynamics_fn = dynamics, rewards_fn = rewards, **kwargs)
        agent = rfsac_agent_network_vmap.CustomModelRFSACAgent(
            dynamics_fn=dynamics,
            rewards_fn=rewards,
            sigma=args.rf_sigma,
            #    learn_rf = True,
            **kwargs,
        )
    else:
        raise NotImplementedError("Algorithm not implemented.")

    # replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, device=args.device)

    if args.env == "custom":
        register(
            id="custom-v0",
            entry_point="repr_control.envs:CustomEnv",
            max_episode_steps=max_step,
        )
        env = gymnasium.make(
            "custom-v0",
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
        )
        eval_env = gymnasium.make(
            "custom-v0",
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
        )
        env = gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        eval_env = gymnasium.wrappers.RescaleAction(
            eval_env, min_action=-1, max_action=1
        )
    elif args.env == "custom_vec":
        from repr_control.envs.custom_env import CustomVecEnv

        env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
            sample_batch_size=args.batch_size,
            max_episode_steps=max_step,
            device=torch.device(args.device),
        )
        eval_env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
            sample_batch_size=1024,
            max_episode_steps=max_step,
            device=torch.device(args.device),
        )
    else:
        env = gymnasium.make(args.env)
        eval_env = gymnasium.make(args.env)
        env = gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        eval_env = gymnasium.wrappers.RescaleAction(
            eval_env, min_action=-1, max_action=1
        )

    # Evaluate untrained policy
    evaluations = []

    state, _ = env.reset()
    done = False
    episode_reward = torch.zeros((args.batch_size, N), device=torch.device(args.device))
    episode_timesteps = 0
    episode_num = 0
    timer = util_network.Timer()

    # keep track of best eval model's state dict
    best_eval_reward = -1e6
    best_actor = None
    best_critic = None

    # save parameters
    # kwargs.update({"action_space": None}) # action space might not be serializable
    with open(os.path.join(log_path, "train_params.yaml"), "w") as fp:
        yaml.dump(kwargs, fp, default_flow_style=False)

    with open(
        os.path.join(log_path, "train_params.pth"), "wb"
    ) as f:  # 'wb' mode for writing in binary
        torch.save(kwargs, f)
        print("pytorch kwargs saved")

    for t in range(int(args.max_timesteps + args.start_timesteps)):
        # print("started episode %d" %t)

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            # action = env.action_space.sample()
            action = env.sample_action()
        else:
            # action = agent.batch_select_action(state, explore=True)
            action = agent.batch_select_action_network(state, explore=True)

        # Perform action
        next_state, reward, terminated, truncated, rollout_info = env.step(action)
        done = truncated
        # print("reward shape", reward.shape)
        # replay_buffer.add(state, action, next_state, reward, done)

        batch = Batch(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=torch.zeros(
                size=(state.shape[0], N), device=torch.device(args.device)
            ),
        )

        state = next_state.clone()
        # print("reward shape", reward.shape)
        # print("episode reward shape", episode_reward.shape)
        episode_reward += reward
        info = {}

        if t >= args.start_timesteps:
            info = agent.batch_train(batch)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            avg_reward = episode_reward.mean().cpu().item()
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Average Reward: {avg_reward:.3f}"
            )
            # Reset environment
            # info.update({'ep_len': episode_timesteps})
            state, _ = env.reset()
            done = False
            episode_reward = torch.zeros(
                (args.batch_size, N), device=torch.device(args.device)
            )
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eval_len, eval_ret, _, _ = util_network.batch_eval(agent, eval_env)
            evaluations.append(eval_ret)

            if t >= args.start_timesteps:
                info.update({"eval_ret": eval_ret})

            print("Step {}. Steps per sec: {:.4g}.".format(t + 1, steps_per_sec))

            if eval_ret > best_eval_reward:
                # best_actor = agent.actor.state_dict()
                # best_critic = agent.critic.state_dict()
                best_actors = [agent.actors[i].state_dict() for i in range(N)]
                best_critics = [agent.critics[i].state_dict() for i in range(N)]

                # save best actor/best critic
                for i in range(N):
                    torch.save(best_actors[i], log_path + "/best_actor_%d.pth" % i)
                    torch.save(best_critics[i], log_path + "/best_critic_%d.pth" % i)

            best_eval_reward = max(evaluations)

            _, V_ret, V_mean_err = util_network.batch_eval_discounted(agent, eval_env)
            info.update({"mean V_err": V_mean_err})

        if (t + 1) % 500 == 0:
            for key, value in info.items():
                # print("key",key)
                # print("value", value)
                if "dist" not in key:
                    summary_writer.add_scalar(f"info/{key}", value, t + 1)
                else:
                    for dist_key, dist_val in value.items():
                        summary_writer.add_histogram(dist_key, dist_val, t + 1)
            summary_writer.flush()

    summary_writer.close()

    print("Total time cost {:.4g}s.".format(timer.time_cost()))

    for i in range(N):
        torch.save(agent.actors[i].state_dict(), log_path + "/actor_last_%d.pth" % i)
        torch.save(agent.critics[i].state_dict(), log_path + "/critic_last_%d.pth" % i)
