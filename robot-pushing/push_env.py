import numpy as np
import gym
from gym import spaces
import robosuite
from robosuite.controllers import load_controller_config
import imageio, tqdm


class PushingEnvironment(gym.Env):
    def __init__(self, horizon, control_freq, renderable=False):
        self.renderable = renderable
        self.env = robosuite.make(
            "Push",
            robots=["Panda"],
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=renderable,
            render_visual_mesh=renderable,
            render_collision_mesh=False,
            camera_names=["frontview"] if renderable else None,
            control_freq=control_freq,
            horizon=horizon,
            use_object_obs=True,
            use_camera_obs=renderable,
        )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low[:2], high=high[:2])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[9])
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
            obs["cube_to_goal_pos"],
        ])

    def reset(self):
        self.curr_obs = self.env.reset()
        self.step_num = 0
        return self._get_flat_obs(self.curr_obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(np.concatenate([action, [0, 0, 0, 0]]))
        return_obs = self._get_flat_obs(next_obs)
        if self.renderable:
            info["image"] = self.curr_obs["frontview_image"][::-1]
            info["step"] = self.step_num
            if done:
                info["final_image"] = next_obs["frontview_image"][::-1]
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
        fake_goal = obs_next[-1, :3] - obs_next[-1, 3:6]  # final cube position
        obs[:, 6:] = (obs[:, :3] - obs[:, 3:6]) - fake_goal
        obs_next[:, 6:] = (obs_next[:, :3] - obs_next[:, 3:6]) - fake_goal
        reward = np.array([self.compute_reward(fake_goal, o[:3] - o[3:6], {}) for o in obs_next])
        done = reward == 0
        return obs, obs_next, reward, done

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = -1
        if self.env.check_success(desired_goal, achieved_goal):
            reward = 0
        return reward

    def render(self, mode="human"):
        assert self.renderable
        return self.curr_obs["frontview_image"][::-1]


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    target = np.array([0, 0])
    env = PushingEnvironment(30, 2, True)
    env.seed(13)
    obs = env.reset()
    for i in tqdm.tqdm(range(30)):
        # print(env.env.robots[0]._joint_positions)
        img = env.render()
        imageio.imwrite(f"render/{i:03}.png", img)
        obs, _, done, _ = env.step(env.action_space.sample())
        if done:
            # env.seed(i // 30 + 10)
            env.reset()
