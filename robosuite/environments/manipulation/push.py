import heapq
from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class Push(SingleArmEnv):
    """
    This class corresponds to the block pushing task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory.
            For this environment, setting a value other than the default ("WipingGripper") will raise an
            AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    CUBE_HALFSIZE = 0.025  # half of side length of block
    GOAL_RADIUS = 0.05  # radius of goal circle
    SPAWN_AREA_SIZE = 0.15  # half of side length of square where block and goal can spawn
    BLOCK_BOUNDS_SIZE = 0.175  # half of side length of region outside of which the block gets out of bounds reward
    GRIPPER_BOUNDS = np.array([
        [-0.4, 0.4],  # x
        [-0.4, 0.4],  # y
        [0, 0.8],  # z
    ])
    OBSTACLE_GRID_RESOLUTION = 5  # side length of obstacle grid
    OBSTACLE_HALF_SIDELENGTH = SPAWN_AREA_SIZE / OBSTACLE_GRID_RESOLUTION

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        # gripper_types="PushingGripper",
        initialization_noise=None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        num_obstacles=0,
        standard_reward=0,
        goal_reward=1,
        obstacle_reward=-2,
        out_of_bounds_reward=-2,
        hard_obstacles=False,
        keep_gripper_in_cube_plane=False,
        reward_shaping=False,
    ):
        self.num_obstacles = num_obstacles
        self.standard_reward = standard_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward
        self.out_of_bounds_reward = out_of_bounds_reward
        self.hard_obstacles = hard_obstacles
        self.keep_gripper_in_cube_plane = keep_gripper_in_cube_plane

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((-0.1, 0, 0.8))

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.cube_initializer = None
        self.goal_initializer = None

        self.reward_scale = 1.0
        self.reward_shaping = reward_shaping

        self.has_touched = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            # gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """

        goal_pos = np.array(self.sim.data.body_xpos[self.goal_body_id])
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])

        if self.reward_shaping:
            reward = 0.0

            if self.check_success(goal_pos, cube_pos):
                reward = 2.25
            else:
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                dist = np.linalg.norm(gripper_site_pos - cube_pos)
                reaching_reward = 1 - np.tanh(10.0 * dist)
                reward += reaching_reward

                if self._check_gripper_contact(gripper=self.robots[0].gripper, object_geoms=self.cube) or (self.has_touched and dist < 0.08):
                    self.has_touched = True

                    if not self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                        reward += 0.25

                    goal_dist = np.linalg.norm(goal_pos - cube_pos)
                    reaching_goal = 1 - np.tanh(10.0 * goal_dist)
                    reward += reaching_goal
                elif dist >= 0.08:
                    self.has_touched = False

            return reward / 2.25
        else:
            rew = self.compute_reward(
                goal_pos,
                cube_pos,
                {}
            )

            return rew

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # right side of table ([0, 0.2])
        # self.robots[0].init_qpos = np.array([
        #     0.2602561,
        #     1.09212466,
        #     0.03546359,
        #     -1.78849099,
        #     -0.2233546,
        #     2.93094696,
        #     1.27766025
        # ])

        # middle of table ([0, 0])
        # self.robots[0].init_qpos = np.array([
        #     0,
        #     1.0431,
        #     0,
        #     -1.9429,
        #     0,
        #     3.0427,
        #     0.78539816,
        # ])

        # middle of table (offset) ([-0.1, 0])
        # self.robots[0].init_qpos = np.array([
        #     0,
        #     0.96629869,
        #     0,
        #     -2.23725147,
        #     0,
        #     3.26003255,
        #     0.78539816
        # ])

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.cube = BoxObject(
            name="cube",
            size=[self.CUBE_HALFSIZE, self.CUBE_HALFSIZE, self.CUBE_HALFSIZE],
            rgba=(1, 0, 0, 1)
        )
        self.goal = CylinderObject(
            name="goal",
            size=[self.GOAL_RADIUS, 0.001],
            rgba=(0, 1, 0, 1),
            obj_type="visual",
            joints=None,
        )
        self.obstacles = [
            BoxObject(
                name=f"obstacle{i}",
                size=[
                    self.OBSTACLE_HALF_SIDELENGTH,
                    self.OBSTACLE_HALF_SIDELENGTH,
                    self.CUBE_HALFSIZE if self.hard_obstacles else 0.001
                ],
                rgba=(1, 1, 0, 1),
                obj_type="all" if self.hard_obstacles else "visual",
                joints="default" if self.hard_obstacles else None,
                density=1e8
            )
            for i in range(self.num_obstacles)
        ]

        # Create placement initializer
        if self.cube_initializer is not None:
            self.cube_initializer.reset()
            self.cube_initializer.add_objects(self.cube)
        else:
            self.cube_initializer = UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
                x_range=[-self.SPAWN_AREA_SIZE, self.SPAWN_AREA_SIZE],
                y_range=[-self.SPAWN_AREA_SIZE, self.SPAWN_AREA_SIZE],
                rotation=0,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )

        if self.goal_initializer is not None:
            self.goal_initializer.reset()
            self.goal_initializer.add_objects(self.goal)
        else:
            self.goal_initializer = UniformRandomSampler(
                name="GoalSampler",
                mujoco_objects=self.goal,
                x_range=[-self.SPAWN_AREA_SIZE, self.SPAWN_AREA_SIZE],
                y_range=[-self.SPAWN_AREA_SIZE, self.SPAWN_AREA_SIZE],
                rotation=0,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.cube, self.goal] + self.obstacles,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.goal_body_id = self.sim.model.body_name2id(self.goal.root_body)
        # self.gripper_body_id = self.sim.model.body_name2id(f"{self.robots[0].gripper.naming_prefix}panda_gripper")
        self.obstacle_body_ids = [
            self.sim.model.body_name2id(obstacle.root_body)
            for obstacle in self.obstacles
        ]
        self.table_geom_id = self.sim.model.geom_name2id(self.model.mujoco_arena.table_collision.get('name'))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            # @sensor(modality=modality)
            # def cube_quat(obs_cache):
            #     return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache else np.zeros(3)

            # goal-related observables
            @sensor(modality=modality)
            def goal_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.goal_body_id])

            @sensor(modality=modality)
            def gripper_to_goal_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["goal_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "goal_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cube_to_goal_pos(obs_cache):
                return obs_cache["cube_pos"] - obs_cache["goal_pos"] if \
                    "cube_pos" in obs_cache and "goal_pos" in obs_cache else np.zeros(3)

            def obstacle_pos(i):
                @sensor(modality=modality)
                def f(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.obstacle_body_ids[i]])

                f.__name__ = f"obstacle{i}_pos"
                return f

            def gripper_to_obstacle_pos(i):
                @sensor(modality=modality)
                def f(obs_cache):
                    return obs_cache[f"{pf}eef_pos"] - obs_cache[f"obstacle{i}_pos"] if \
                        f"{pf}eef_pos" in obs_cache and f"obstacle{i}_pos" in obs_cache else np.zeros(3)

                f.__name__ = f"gripper_to_obstacle{i}_pos"
                return f

            def cube_to_obstacle_pos(i):
                @sensor(modality=modality)
                def f(obs_cache):
                    return obs_cache["cube_pos"] - obs_cache[f"obstacle{i}_pos"] if \
                        "cube_pos" in obs_cache and f"obstacle{i}_pos" in obs_cache else np.zeros(3)

                f.__name__ = f"cube_to_obstacle{i}_pos"
                return f

            sensors = [cube_pos, gripper_to_cube_pos, goal_pos, gripper_to_goal_pos, cube_to_goal_pos]
            sensors += list(map(obstacle_pos, range(self.num_obstacles)))
            sensors += list(map(gripper_to_obstacle_pos, range(self.num_obstacles)))
            sensors += list(map(cube_to_obstacle_pos, range(self.num_obstacles)))
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # self.robots[0].controller.position_limits = self.GRIPPER_BOUNDS.T + self.table_offset

        # for geom in self.cube.contact_geoms + [g for obs in self.obstacles for g in obs.contact_geoms]:
        #     self.sim.model.geom_contype[self.sim.model.geom_name2id(geom)] = 0b100
        #     self.sim.model.geom_conaffinity[self.sim.model.geom_name2id(geom)] = 0b100
        # for i in range(self.sim.model.body_geomnum[self.gripper_body_id]):
        #     self.sim.model.geom_contype[i + self.sim.model.body_geomadr[self.gripper_body_id]] = 0b111
        # self.sim.model.geom_contype[self.table_geom_id] = 0b111


        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            gripper_pos = self.sim.data.body_xpos[self.robots[0].eef_site_id]
            cube_pos = gripper_pos
            while np.max(np.abs(cube_pos[:2] - gripper_pos[:2])) <= self.CUBE_HALFSIZE + 0.013:
                cube_placement = self.cube_initializer.sample()
                cube_pos, cube_quat, _ = cube_placement["cube"]

            self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([np.array(cube_pos), np.array(cube_quat)]))

            goal_placement = self.goal_initializer.sample(fixtures=cube_placement)
            goal_pos, goal_quat, _ = goal_placement["goal"]
            self.sim.model.body_pos[self.goal_body_id] = np.array([goal_pos])
            self.sim.model.body_quat[self.goal_body_id] = np.array([goal_quat])

            if self.num_obstacles > 0:
                cube_grid_index = (
                        (cube_pos[:2] - self.table_offset[:2] + self.SPAWN_AREA_SIZE) // (self.OBSTACLE_HALF_SIDELENGTH * 2)
                ).astype(int)  # cube pos discretized to integer index on the obstacle grid
                goal_pos_grid = (goal_pos[:2] - self.table_offset[:2] + self.SPAWN_AREA_SIZE)\
                    / (self.OBSTACLE_HALF_SIDELENGTH * 2)  # goal pos in grid coordinates but not discretized
                possible_obstacle_indices = np.mgrid[
                    :self.OBSTACLE_GRID_RESOLUTION,
                    :self.OBSTACLE_GRID_RESOLUTION
                ].reshape(2, -1).T  # shape (grid_res^2, 2)
                possible_obstacle_indices = possible_obstacle_indices[(
                        (possible_obstacle_indices != cube_grid_index)
                        & (possible_obstacle_indices != goal_pos_grid.astype(int))
                ).any(axis=-1)]
                rng = np.random.default_rng()
                while True:
                    obstacle_indices = rng.choice(possible_obstacle_indices, self.num_obstacles, replace=False, axis=0)
                    if self._check_path_to_goal(cube_grid_index, goal_pos_grid, obstacle_indices):
                        break

                obstacle_locations = obstacle_indices * self.OBSTACLE_HALF_SIDELENGTH * 2\
                    - self.SPAWN_AREA_SIZE + self.table_offset[:2] + self.OBSTACLE_HALF_SIDELENGTH
                for i, ((x, y), obstacle_id) in enumerate(zip(obstacle_locations, self.obstacle_body_ids)):
                    pos = np.array([x, y, self.table_offset[2] + self.obstacles[0].top_offset[2]])
                    quat = np.array([1, 0, 0, 0])
                    if self.hard_obstacles:
                        self.sim.data.set_joint_qpos(self.obstacles[i].joints[0], np.concatenate([pos, quat]))
                    else:
                        self.sim.model.body_pos[obstacle_id] = np.array([pos])
                        self.sim.model.body_quat[obstacle_id] = np.array([quat])

        self.obstacle_pos = np.array([
            self.sim.data.body_xpos[oid]
            for oid in self.obstacle_body_ids
        ])

    def _check_path_to_goal(self, cube_index, goal_pos, obstacle_indices):
        obstacle_indices = set(map(tuple, obstacle_indices))
        queue = [(np.linalg.norm(cube_index - goal_pos), 0, tuple(cube_index))]  # each element is (total_cost, forward_cost, position)
        visited = set()
        while queue:
            _, fcost, pos = heapq.heappop(queue)
            if pos in visited:
                continue
            # check if goal circle intersects the current grid square
            closest = np.maximum(pos, np.minimum(np.array(pos) + 1, goal_pos))
            if np.linalg.norm(closest - goal_pos) <= self.GOAL_RADIUS:
                return True
            visited.add(pos)
            for direction in np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]):
                next_pos = pos + direction
                if (
                    np.any(next_pos < 0) or
                    np.any(next_pos >= self.OBSTACLE_GRID_RESOLUTION) or
                    tuple(next_pos) in obstacle_indices
                ):
                    continue
                heapq.heappush(queue, (np.linalg.norm(next_pos - goal_pos) + fcost + 1, fcost + 1, tuple(next_pos)))
        return False

    def check_success(self, goal_pos, cube_pos):
        """
        Check if cube has reached goal.

        Returns:
            bool: True if cube has reached goal.
        """
        return np.linalg.norm(goal_pos[:2] - cube_pos[:2]) <= self.GOAL_RADIUS

    def _check_success(self):
        goal_pos = np.array(self.sim.data.body_xpos[self.goal_body_id])
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])

        return self.check_success(goal_pos, cube_pos)

    def compute_reward(self, goal_pos, cube_pos, info):
        if np.linalg.norm(goal_pos[:2] - cube_pos[:2]) <= self.GOAL_RADIUS:
            return self.goal_reward
        if np.any(np.abs(cube_pos[:2] - self.table_offset[:2]) >= self.BLOCK_BOUNDS_SIZE):
            return self.out_of_bounds_reward
        if self.num_obstacles > 0:
            if np.any(np.max(np.abs(self.obstacle_pos[:, :2] - cube_pos[:2]), axis=-1) <= self.OBSTACLE_HALF_SIDELENGTH):
                return self.obstacle_reward
        return self.standard_reward

    def _pre_action(self, action, policy_step=False):
        # if (self.keep_gripper_in_cube_plane
        #         and self.sim.data.body_xpos[self.gripper_body_id][2] - self.table_offset[2] > 0.05):
        #     action[2] = -0.1
        super()._pre_action(action, policy_step)
