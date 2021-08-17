import numpy as np
import gym
from gym import spaces
import pickle
import robosuite
from robosuite.controllers import load_controller_config
import imageio
import shutil
import os
import tqdm
import robosuite.utils.macros as macros

macros.SIMULATION_TIMESTEP = 0.02
np.set_printoptions(suppress=True)


class StickPushingEnvironment(gym.Env):
    def __init__(self, horizon, control_freq, renderable=False, start_grasping="never"):
        self.renderable = renderable
        self.env = robosuite.make(
            "StickPush",
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
        )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(
            low=np.concatenate([low[:3], [low[-1]]]),
            high=np.concatenate([high[:3], [high[-1]]])
            # low=low[:3],
            # high=high[:3],
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[21])
        self.curr_obs = None
        self.step_num = None

        self.grasping_state = None
        if os.path.exists("grasping_state.pickle"):
            with open("grasping_state.pickle", "rb") as f:
                self.grasping_state = pickle.load(f)
        self.default_robot_init = self.env.robots[0].init_qpos

        self.start_grasping = start_grasping

    @property
    def start_grasping(self):
        return self._start_grasping

    @start_grasping.setter
    def start_grasping(self, value):
        assert value in ["never", "random", "always"]
        if value != "never":
            assert self.grasping_state is not None
        self._start_grasping = value

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
            obs["gripper_to_stick_pos"],
            obs["stick_to_cube_pos"],
            obs["stick_to_goal_pos"],
        ])

    def reset(self):
        self.env.reset()
        if self.start_grasping == "always" or (self.start_grasping == "random" and np.random.rand() < 0.5):
            self.env.sim.data.set_joint_qpos(self.env.stick.joints[0], self.grasping_state["stick_qpos"])
            self.env.robots[0].init_qpos = self.grasping_state["joint_pos"]
            self.env.robots[0].reset()
            self.curr_obs = self.env._get_observations(force_update=True)
        else:
            self.env.robots[0].init_qpos = self.default_robot_init
            self.env.robots[0].reset()
            self.curr_obs = self.env._get_observations(force_update=True)
        self.step_num = 0
        return self._get_flat_obs(self.curr_obs)

    def step(self, action):
        if np.array_equal(action.shape, self.action_space.shape):
            next_obs, reward, done, info = self.env.step(np.concatenate([action[:3], [0, 0, 0], [action[-1]]]))
            # next_obs, reward, done, info = self.env.step(np.concatenate([action[:3], [0, 0, 0, 1]]))
        else:
            next_obs, reward, done, info = self.env.step(action)
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
        obs[:, 9:12] = (obs[:, :3] - obs[:, 3:6]) - fake_goal
        obs_next[:, 9:12] = (obs_next[:, :3] - obs_next[:, 3:6]) - fake_goal
        # stick to goal pos
        obs[:, 18:21] = (obs[:, :3] - obs[:, 12:15]) - fake_goal
        obs_next[:, 18:21] = (obs_next[:, :3] - obs[:, 12:15]) - fake_goal
        # rewards
        rewards = [self.env.compute_reward(fake_goal, on[:3] - on[3:6], {}) for on in obs_next]
        # rewards = []
        # for on in obs_next:
        #     reward = self.compute_reward(fake_goal, on[:3] - on[3:6], {})
        #     rewards.append(reward)
        #     if reward == 0:
        #         break

        # dones
        dones = np.full_like(rewards, False, dtype=bool)
        dones[-1] = True
        return obs[:len(rewards)], obs_next[:len(rewards)], np.array(rewards), dones

    def render(self, mode="human"):
        assert self.renderable
        return self.curr_obs["agentview_image"][::-1]


if __name__ == "__main__":
    shutil.rmtree("render")
    os.makedirs("render")
    np.random.seed(1)
    env = StickPushingEnvironment(20, 2, True)
    obs = env.reset()
    for i in tqdm.tqdm(range(300)):
        # print(env.env.robots[0]._joint_positions)
        img = env.render()
        imageio.imwrite(f"render/{i:03}.png", img)
        obs, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
