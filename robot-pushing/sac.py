import argparse
import numpy as np
import torch
import os
import datetime
import shutil
import pprint
import imageio
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import SACPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector

from her import HERReplayBuffer
from push_env import PushingEnvironment
from stick_push_env import StickPushingEnvironment
from input_norm import InputNorm

ENV_DICT = {
    "push": PushingEnvironment,
    "stick_push": StickPushingEnvironment,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--control-freq', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=10**6)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64, 64])
    parser.add_argument('--actor-lr', type=float, default=0.0003)
    parser.add_argument('--critic-lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=0.0003)
    parser.add_argument("--start-timesteps", type=int, default=300)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=5000)  # evaluation is performed once per epoch
    parser.add_argument('--step-per-collect', type=int, default=1)  # steps between policy updates
    parser.add_argument('--update-per-step', type=float, default=1)  # number of grad updates per step
    parser.add_argument('--n-step', type=int, default=1)  # number of steps to look ahead
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10)  # number of test episodes and workers (1 episode per worker)
    parser.add_argument('--train-num', type=int, default=1)  # number of train workers
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true')
    return parser.parse_args()


def main(args):
    if not args.watch:
        assert args.logdir is not None
    env_fn = ENV_DICT[args.env]
    env = env_fn(args.horizon, args.control_freq)
    train_envs = SubprocVectorEnv(
        [lambda: env_fn(args.horizon, args.control_freq) for _ in range(args.train_num)])
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = np.max(env.action_space.high)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    test_envs = SubprocVectorEnv(
        [lambda: env_fn(args.horizon, args.control_freq, renderable=args.watch) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = InputNorm(Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device))
    actor = ActorProb(
        net_a, args.action_shape, max_action=args.max_action,
        device=args.device, unbounded=True, conditioned_sigma=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = InputNorm(Net(args.state_shape, args.action_shape,
                           hidden_sizes=args.hidden_sizes,
                           concat=True, device=args.device))
    net_c2 = InputNorm(Net(args.state_shape, args.action_shape,
                           hidden_sizes=args.hidden_sizes,
                           concat=True, device=args.device))
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=args.tau, gamma=args.gamma, alpha=args.alpha,
        estimation_step=args.n_step, action_space=env.action_space)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    if not args.watch:
        train(policy, env, train_envs, test_envs, args)
    else:
        test(policy, test_envs, args)


def train(policy, env, train_envs, test_envs, args):
    def preprocess_fn(obs=None, obs_next=None, **kwargs):
        if obs_next is not None:
            states = obs_next
            kwargs["obs_next"] = obs_next
        else:
            states = obs
            kwargs["obs"] = obs
        for state in states:
            policy.actor.preprocess.update(state)
            policy.critic1.preprocess.update(np.concatenate([state, np.zeros(env.action_space.shape)]))
            policy.critic1_old.preprocess.update(np.concatenate([state, np.zeros(env.action_space.shape)]))
            policy.critic2.preprocess.update(np.concatenate([state, np.zeros(env.action_space.shape)]))
            policy.critic2_old.preprocess.update(np.concatenate([state, np.zeros(env.action_space.shape)]))

        return kwargs

    # collector
    buffer = HERReplayBuffer(env, total_size=args.buffer_size, buffer_num=args.train_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True, preprocess_fn=preprocess_fn)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'sac_seed_{args.seed}_{t0}'
    log_path = os.path.join(args.logdir, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'best_policy.pth'))

    def test_fn(epoch, step):
        torch.save(policy.state_dict(), os.path.join(log_path, f'policy_e{epoch}_s{step}.pth'))

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epochs,
        args.step_per_epoch, args.step_per_collect, args.test_num,
        args.batch_size, save_fn=save_fn, logger=logger, test_fn=test_fn,
        update_per_step=args.update_per_step, test_in_train=False)
    pprint.pprint(result)


def test(policy, test_envs, args):
    policy.eval()
    shutil.rmtree("render")
    os.makedirs("render")

    def preprocess_fn(info=None, **kwargs):
        if info is not None:
            for env_info in info:
                base = f"render/{env_info.env_id}"
                os.makedirs(base, exist_ok=True)
                imageio.imwrite(f"{base}/{env_info.step:03}.png", env_info.image)
                if "final_image" in info:
                    imageio.imwrite(f"{base}/{env_info.step + 1:03}.png", env_info.final_image)
        return kwargs

    collector = Collector(policy, test_envs, preprocess_fn=preprocess_fn)
    collector.collect(n_episode=args.test_num)


if __name__ == '__main__':
    main(get_args())