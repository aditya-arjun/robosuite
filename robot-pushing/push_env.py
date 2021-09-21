import itertools
import os
import shutil

import numpy as np
import gym
from gym import spaces

import robosuite
from robosuite.controllers import load_controller_config
import robosuite.utils.macros as macros
import imageio, tqdm

from her import HERReplayBuffer
from tianshou.data import Batch

macros.SIMULATION_TIMESTEP = 0.02
np.set_printoptions(suppress=True)


class PushingEnvironment(gym.Env):
    def __init__(self, horizon, control_freq, num_obstacles=0, renderable=False):
        self.num_obstacles = num_obstacles
        self.renderable = renderable
        self.env = robosuite.make(
            "Push",
            robots=["Panda"],
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=renderable,
            render_visual_mesh=renderable,
            render_collision_mesh=False,
            camera_names=["agentview"] if renderable else None,
            control_freq=control_freq,
            horizon=horizon,
            use_object_obs=True,
            use_camera_obs=renderable,
            hard_reset=False,
            num_obstacles=num_obstacles,
        )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low[:3], high=high[:3])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[12 + 6 * num_obstacles])
        self.curr_obs = None
        self.step_num = None

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)

    def _get_flat_obs(self, obs):
        return np.concatenate([
            obs["robot0_eef_pos"],
            obs["gripper_to_cube_pos"],
            obs["gripper_to_goal_pos"],
            obs["cube_to_goal_pos"],
        ] + list(itertools.chain.from_iterable(zip(
            [obs[f"gripper_to_obstacle{i}_pos"] for i in range(self.num_obstacles)],
            [obs[f"cube_to_obstacle{i}_pos"] for i in range(self.num_obstacles)]
        ))))

    def reset(self):
        self.curr_obs = self.env.reset()
        self.step_num = 0
        return self._get_flat_obs(self.curr_obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(np.concatenate([action, [0, 0, 0]]))
        info["TimeLimit.truncated"] = done
        return_obs = self._get_flat_obs(next_obs)
        if self.renderable:
            info["image"] = self.curr_obs["agentview_image"][::-1]
            info["step"] = self.step_num
            if done:
                info["final_image"] = next_obs["agentview_image"][::-1]
        self.curr_obs = next_obs
        self.step_num += 1
        return return_obs, reward, done, info

    def her(self, obs, obs_next):
        """
        Takes a list of observations (and next observations) from an entire episode and returns
        the HER-modified version of the episode in the form of 4 lists: (obs, obs_next, reward, done).
        """
        obs = np.array(obs)
        obs_next = np.array(obs_next)
        # final cube position
        fake_goal = obs_next[-1, :3] - obs_next[-1, 3:6]
        # gripper to goal pos
        obs[:, 6:9] = obs[:, :3] - fake_goal
        obs_next[:, 6:9] = obs_next[:, :3] - fake_goal
        # cube to goal pos
        obs[:, 9:] = (obs[:, :3] - obs[:, 3:6]) - fake_goal
        obs_next[:, 9:] = (obs_next[:, :3] - obs_next[:, 3:6]) - fake_goal
        rewards = [self.env.compute_reward(fake_goal, on[:3] - on[3:6], {}) for on in obs_next]
        # rewards = []
        # for on in obs_next:
        #     reward = self.compute_reward(fake_goal, on[:3] - on[3:6], {})
        #     rewards.append(reward)
        #     if reward == 0:
        #         break
        dones = np.full_like(rewards, False, dtype=bool)
        dones[-1] = True
        infos = {
            "TimeLimit.truncated": dones.copy()
        }
        return obs[:len(rewards)], obs_next[:len(rewards)], np.array(rewards), dones, infos

    def render(self, mode="human"):
        assert self.renderable
        return self.curr_obs["agentview_image"][::-1]


if __name__ == "__main__":
    shutil.rmtree("render")
    os.makedirs("render")
    env = PushingEnvironment(1, 2, 10, renderable=True)
    env.seed(0)
    # buf = HERReplayBuffer(env, total_size=20, buffer_num=1)
    obs = env.reset()
    # for i in range(3):
    #     buf.add(Batch(
    #         obs=[obs],
    #         obs_next=[obs],
    #         act=[[0, 0, 0]],
    #         rew=[-100],
    #         done=[False if i < 2 else True]
    #     ))
    # actions = [[0, 0, 1]] * 2 + [[0, -1, 0]] * 2 + [[1, 0, -1]] * 2 + [[0, 1, 0]] * 3\
    #     + [[0, 0, 0]] * 2 + [[1, 0, 0]] * 2 + [[0, 1, -1]] + [[-1, 0, 0]] * 4
    for i in tqdm.tqdm(range(300)):
        # print(env.env.robots[0]._joint_positions)
        img = env.render()
        imageio.imwrite(f"render/{i:03}.png", img)
        obs_next, rew, done, _ = env.step(env.action_space.sample())
        # if i == 17:
        #     done = True
        # buf.add(Batch(
        #     obs=[obs],
        #     obs_next=[obs_next],
        #     act=[actions[i]],
        #     rew=[rew],
        #     done=[done]
        # ))
        obs = obs_next
        if done:
            # env.seed(i // 30 + 10)
            env.reset()
