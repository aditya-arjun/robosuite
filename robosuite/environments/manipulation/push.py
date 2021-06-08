from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
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
    GOAL_RADIUS = 0.05  # half of side length of goal square
    SPAWN_AREA_SIZE = 0.15  # half of side length of square where block and goal can spawn
    GRIPPER_BOUNDS_MIN = np.array([-0.2, -0.2, 0.034])  # x, y, z bounds of gripper position
    GRIPPER_BOUNDS_MAX = np.array([0.2, 0.2, 0.115])

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="PushingGripper",
        initialization_noise=None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
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
    ):

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((-0.1, 0, 0.8))

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.cube_initializer = None
        self.goal_initializer = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
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
        return self.compute_reward(
            np.array(self.sim.data.body_xpos[self.goal_body_id]),
            np.array(self.sim.data.body_xpos[self.cube_body_id]),
            {}
        )

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
        self.robots[0].init_qpos = np.array([
            0,
            0.96629869,
            0,
            -2.23725147,
            0,
            3.26003255,
            0.78539816
        ])

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size=[self.CUBE_HALFSIZE, self.CUBE_HALFSIZE, self.CUBE_HALFSIZE],
            material=redwood,
        )
        self.goal = CylinderObject(
            name="goal",
            size=[self.GOAL_RADIUS, 0.001],
            material=greenwood,
            obj_type="visual",
            joints=None,
        )

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
            mujoco_objects=[self.cube, self.goal],
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
        self.gripper_body_id = self.sim.model.body_name2id(f"{self.robots[0].gripper.naming_prefix}pushing_gripper")

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

            sensors = [cube_pos, gripper_to_cube_pos, goal_pos, gripper_to_goal_pos, cube_to_goal_pos]
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

        self.robots[0].controller.position_limits = np.array([
            self.table_offset + self.GRIPPER_BOUNDS_MIN,
            self.table_offset + self.GRIPPER_BOUNDS_MAX
        ])

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            gripper_pos = self.sim.data.body_xpos[self.gripper_body_id]
            cube_pos = gripper_pos
            while np.max(np.abs(cube_pos[:2] - gripper_pos[:2])) <= self.CUBE_HALFSIZE + 0.013:
                cube_placement = self.cube_initializer.sample()
                cube_pos, cube_quat, _ = cube_placement["cube"]

            self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([np.array(cube_pos), np.array(cube_quat)]))

            goal_placement = self.goal_initializer.sample(fixtures=cube_placement)
            goal_pos, goal_quat, _ = goal_placement["goal"]
            self.sim.model.body_pos[self.goal_body_id] = np.array([goal_pos])
            self.sim.model.body_quat[self.goal_body_id] = np.array([goal_quat])

    def check_success(self, goal_pos, cube_pos):
        """
        Check if cube has reached goal.

        Returns:
            bool: True if cube has reached goal.
        """
        return np.linalg.norm(goal_pos[:2] - cube_pos[:2]) <= self.GOAL_RADIUS

    def compute_reward(self, goal_pos, cube_pos, info):
        return 0 if self.check_success(goal_pos, cube_pos) else -1

    # def _post_action(self, action):
    #     """
    #     In addition to super method, terminate early if task is completed
    #
    #     Args:
    #         action (np.array): Action to execute within the environment
    #
    #     Returns:
    #         3-tuple:
    #
    #             - (float) reward from the environment
    #             - (bool) whether the current episode is completed or not
    #             - (dict) info about current env step
    #     """
    #     reward, done, info = super()._post_action(action)
    #     done = done or self._check_success()
    #     return reward, done, info

    # def _pre_action(self, action, policy_step=False):
    #     """Does bounds checking to prevent the gripper from leaving a certain area"""
    #     gripper_pos = self.sim.data.body_xpos[self.gripper_body_id] - self.table_offset
    #     mask = ((gripper_pos <= self.GRIPPER_BOUNDS_SIZE) | (action[:3] < 0))\
    #         & ((gripper_pos >= -self.GRIPPER_BOUNDS_SIZE) | (action[:3] > 0))
    #     action[:3] = np.where(mask, action[:3], 0)
    #     super()._pre_action(action, policy_step)
