from __future__ import annotations

import copy
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import clip
import imageio
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import tensorflow.compat.v1 as tf  # type: ignore

import vtamp.environments.pb_utils as pbu
from vtamp.environments.utils import Action, Environment, Task, Updater
from vtamp.policies.utils import Sampler
from vtamp.utils import get_log_dir

log = logging.getLogger(__name__)
BLOCK_SIZE = 0.04
MODELS_PATH = os.path.join(os.path.dirname(__file__), "../../models/")
VILD_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "../../../checkpoints/image_path_v2"
)
COLORS = {
    "blue": (78 / 255, 121 / 255, 167 / 255, 255 / 255),
    "red": (255 / 255, 87 / 255, 89 / 255, 255 / 255),
    "green": (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    "yellow": (237 / 255, 201 / 255, 72 / 255, 255 / 255),
    "orange": (251 / 255, 106 / 255, 74 / 255, 255 / 255),
    "purple": (123 / 255, 102 / 255, 210 / 255, 255 / 255),
    "pink": (247 / 255, 104 / 255, 161 / 255, 255 / 255),
    "teal": (68 / 255, 170 / 255, 153 / 255, 255 / 255),
    "brown": (166 / 255, 86 / 255, 40 / 255, 255 / 255),
}


# Hammer fall params
# LATERAL_FRICTION = 10.0
# ROLLING_FRICTION = 0.001
# SPINNING_FRICTION = 0.001

# Normal params
FINGER_LATERAL_FRICTION = 5.0
FINGER_ROLLING_FRICTION = 5.0
FINGER_SPINNING_FRICTION = 5.0
OBJ_LATERAL_FRICTION = 0.5
OBJ_ROLLING_FRICTION = 0.1
OBJ_SPINNING_FRICTION = 0.1

PIXEL_SIZE = 0.00267857
TABLE_BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0]])
EE_LINK_ID = 9
TIP_LINK_ID = 10
DEFAULT_JOINT_ANGLES = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0]
TABLE_CENTER = [0, -0.5, 0]

# Used as imports for the LLM-generated code
__all__ = [
    "RavenPose",
    "RavenObject",
    "RavenBelief",
    "TABLE_BOUNDS",
    "BLOCK_SIZE",
    "TABLE_CENTER",
    "GraspSampler",
    "RavenGrasp",
]


@dataclass
class RavenPose:
    x: float = 0
    y: float = 0
    z: float = 0
    roll: float = 0
    pitch: float = 0
    yaw: float = 0

    def __iter__(self):
        return iter([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])

    @property
    def point(self):
        return pbu.Point(self.x, self.y, self.z)

    @property
    def euler(self):
        return pbu.Euler(self.roll, self.pitch, self.yaw)

    @property
    def quat(self):
        return pbu.quat_from_euler(self.euler)

    def to_pbu(self):
        return pbu.Pose(point=self.point, euler=self.euler)

    @staticmethod
    def from_pbu(pose):
        euler = pbu.euler_from_quat(pose[1])
        return RavenPose(*pose[0], *euler)

    def dist(self, pose: RavenPose, rot_scale: float = 1e-2) -> float:
        pos_distance, ori_distance = pbu.get_pose_distance(self.to_pbu(), pose.to_pbu())
        return pos_distance + ori_distance * rot_scale

    def multiply(self, pose: RavenPose) -> RavenPose:
        return RavenPose.from_pbu(pbu.multiply(self.to_pbu(), pose.to_pbu()))


class RavenGrasp(RavenPose):
    pass


HOME_EE_POSE = RavenPose(x=0, y=-0.5, z=0.2, roll=np.pi, pitch=0, yaw=-np.pi / 2.0)


@dataclass
class RavenObject:
    category: str
    color: str
    pose: RavenPose = field(default_factory=lambda: RavenPose())
    body: Optional[int] = None

    def __str__(self):
        return 'RavenObject(category="{}", color="{}", pose={})'.format(
            self.category, self.color, [round(pel, 2) for pel in list(self.pose)]
        )


@dataclass
class RavenBelief:
    objects: Dict[str, RavenObject] = field(default_factory=dict)
    observations: List[Any] = field(default_factory=list)

    def __str__(self):
        content = ", ".join([f'"{k}": {v}' for k, v in self.objects.items()])
        full = "{" + str(content) + "}"
        return "RavenBelief({})".format(full)


@dataclass
class GraspSampler(Sampler):
    def sample(self) -> RavenGrasp:
        return RavenPose(
            x=random.uniform(-0.04, 0.04),
            y=random.uniform(-0.04, 0.04),
            pitch=np.pi,
            yaw=random.uniform(-math.pi, math.pi),
        ).multiply(RavenPose(z=-0.005))


# Currently, we assume full observability
class RavenState(RavenBelief):
    pass


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def create_object(category: str, color: str, client: int) -> int:
    if category == "block":
        REDUCED_BS = BLOCK_SIZE
        object_shape = client.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[REDUCED_BS / 2.0, REDUCED_BS / 2.0, REDUCED_BS / 2.0],
        )
        object_visual = client.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0],
        )
        object_id = client.createMultiBody(0.01, object_shape, object_visual)
        client.changeVisualShape(object_id, -1, rgbaColor=COLORS[color])
    elif category == "bowl":
        object_id = client.loadURDF(
            os.path.join(MODELS_PATH, "bowl/bowl.urdf"),
            useFixedBase=1,
        )
        client.changeVisualShape(object_id, -1, rgbaColor=COLORS[color])
    else:
        print("Creating ycb")
        object_id = create_ycb(category, client=client)

    return object_id


ROOT_PATH = os.path.abspath(os.path.join(__file__, *[os.pardir] * 3))
YCB_PATH = os.path.join(ROOT_PATH, "models/ycb")


def ycb_type_from_name(name):
    return "_".join(name.split("_")[1:])


def ycb_type_from_file(path):
    # TODO: rename to be from_dir
    return ycb_type_from_name(os.path.basename(path))


def all_ycb_names():
    return [ycb_type_from_file(path) for path in pbu.list_paths(YCB_PATH)]


def all_ycb_paths():
    return pbu.list_paths(YCB_PATH)


def get_ycb_obj_path(ycb_type, use_concave=False):
    path_from_type = {
        ycb_type_from_file(path): path
        for path in pbu.list_paths(YCB_PATH)
        if os.path.isdir(path)
    }

    if ycb_type not in path_from_type:
        return None

    if use_concave:
        filename = "google_16k/decomp.obj"
    else:
        filename = "google_16k/textured.obj"

    return os.path.join(path_from_type[ycb_type], filename)


def ycb_type_from_name(name):
    return name.split("_", 1)[-1]


def ycb_type_from_file(path):
    # TODO: rename to be from_dir
    return ycb_type_from_name(os.path.basename(path))


def get_ycb_obj_path(ycb_type, use_concave=False):
    path_from_type = {
        ycb_type_from_file(path): path
        for path in pbu.list_paths(YCB_PATH)
        if os.path.isdir(path)
    }

    if ycb_type not in path_from_type:
        return None

    if use_concave:
        filename = "google_16k/textured_vhacd.obj"
    else:
        filename = "google_16k/textured.obj"

    return os.path.join(path_from_type[ycb_type], filename)


def create_ycb(
    name,
    use_concave=True,
    client=None,
    scale=1.0,
    **kwargs,
):
    concave_ycb_path = get_ycb_obj_path(name, use_concave=use_concave)
    ycb_path = get_ycb_obj_path(name)
    mass = 0.02

    # TODO: separate visual and collision boddies
    color = pbu.WHITE

    import trimesh

    mesh = trimesh.load(ycb_path)

    # TODO: separate visual and collision geometries
    # TODO: compute OOBB to select the orientation
    visual_geometry = pbu.get_mesh_geometry(
        ycb_path, scale=scale
    )  # TODO: randomly transform
    collision_geometry = pbu.get_mesh_geometry(concave_ycb_path, scale=scale)
    geometry_pose = pbu.Pose(point=-mesh.center_mass)
    collision_id = pbu.create_collision_shape(
        collision_geometry, pose=geometry_pose, client=client
    )
    visual_id = pbu.create_visual_shape(
        visual_geometry, color=color, pose=geometry_pose, client=client
    )
    body = client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )

    client.changeDynamics(
        body,
        -1,
        lateralFriction=OBJ_LATERAL_FRICTION,
        spinningFriction=OBJ_SPINNING_FRICTION,
        rollingFriction=OBJ_ROLLING_FRICTION,
        frictionAnchor=True,
    )

    pbu.set_all_color(body, pbu.apply_alpha(color, alpha=1.0), client=client)

    return body


# A temporary hack because vision sucks
class RavenGroundTruthBeliefUpdater(Updater):
    def update(self, obs) -> RavenBelief:
        return obs["internal_state"]


class Robotiq2F85:
    """Gripper handling for Robotiq 2F85."""

    def __init__(self, robot, tool, teleport=False, client=None):
        self.robot = robot
        self.tool = tool
        self.client = client
        self.teleport = teleport
        pos = [0.1339999999999999, -0.49199999999872496, 0.5]
        rot = self.client.getQuaternionFromEuler([np.pi, 0, np.pi])

        urdf = os.path.join(MODELS_PATH, "robotiq_2f_85/robotiq_2f_85_nobar.urdf")

        self.body = self.client.loadURDF(urdf, pos, rot)
        self.n_joints = self.client.getNumJoints(self.body)
        self.activated = False

        self.joint_ids = pbu.get_movable_joints(self.body, client=self.client)
        self.gripper_T_arm = pbu.Pose(
            pbu.Point(0, 0, -0.015), pbu.Euler(0, 0, np.pi / 2)
        )
        # Connect gripper base to robot tool.
        self.client.createConstraint(
            self.robot,
            tool,
            self.body,
            -1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.gripper_T_arm[0],
            childFrameOrientation=self.gripper_T_arm[1],
        )

        # Set friction coefficients for gripper fingers.
        for i in range(self.client.getNumJoints(self.body)):
            self.client.changeDynamics(
                self.body,
                i,
                lateralFriction=FINGER_LATERAL_FRICTION,
                spinningFriction=FINGER_SPINNING_FRICTION,
                rollingFriction=FINGER_ROLLING_FRICTION,
                frictionAnchor=True,
            )

        # Start thread to handle additional gripper constraints.
        self.motor_joint = 1
        self.reset_state = None

    def set_reset_state(self):
        self.reset_state = pbu.get_joint_positions(
            self.body, self.joint_ids, client=self.client
        )

    def reset(self):
        assert self.reset_state is not None
        self.release()
        pbu.set_joint_positions(
            self.body, self.joint_ids, self.reset_state, client=self.client
        )

    def apply_transform(self):
        world_T_arm = pbu.get_link_pose(self.robot, self.tool, client=self.client)
        world_T_gripper = pbu.multiply(world_T_arm, pbu.invert(self.gripper_T_arm))
        pbu.set_pose(self.body, world_T_gripper, client=self.client)

    def update_gripper(self):
        """Update joint positions to enforce constraints on gripper
        behavior."""
        # This method now directly mirrors what was previously done in the `step` method within a thread.
        try:
            currj = [
                self.client.getJointState(self.body, i)[0] for i in range(self.n_joints)
            ]
            indj = [6, 3, 8, 5, 10]
            targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
            self.client.setJointMotorControlArray(
                self.body,
                indj,
                self.client.POSITION_CONTROL,
                targj,
                positionGains=np.ones(5),
            )
        except Exception as e:
            print(f"Failed to update gripper: {e}")

        # world_T_arm = pbu.get_link_pose(self.robot, self.tool, client=self.client)
        # world_T_gripper = pbu.get_link_pose(self.body, -1, client=self.client)
        # if(not self.teleport):
        #     print(f"Real arm_T_gripper: "+str(pbu.multiply(pbu.invert(world_T_arm), world_T_gripper)))

    # Close gripper fingers.
    def activate(self):
        self.client.setJointMotorControl2(
            self.body,
            self.motor_joint,
            self.client.VELOCITY_CONTROL,
            targetVelocity=1,
            force=5,
        )
        self.activated = True

    # Open gripper fingers.
    def release(self):
        self.client.setJointMotorControl2(
            self.body,
            self.motor_joint,
            self.client.VELOCITY_CONTROL,
            targetVelocity=-1,
            force=20,
        )
        self.activated = False

    # If activated and object in gripper: check object contact.
    # If activated and nothing in gripper: check gripper contact.
    # If released: check proximity to surface (disabled).
    def detect_contact(self):
        obj, _, ray_frac = self.check_proximity()
        if self.activated:
            empty = self.grasp_width() < 0.01
            cbody = self.body if empty else obj
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)

    #   else:
    #     return ray_frac < 0.14 or self.external_contact()

    # Return if body is in contact with something other than gripper
    def external_contact(self, body=None):
        if body is None:
            body = self.body
        pts = self.client.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]
        return len(pts) > 0  # pylint: disable=g-explicit-length-test

    def check_grasp(self):
        while self.moving():
            time.sleep(0.001)
        success = self.grasp_width() > 0.01
        return success

    def grasp_width(self):
        lpad = np.array(self.client.getLinkState(self.body, 4)[0])
        rpad = np.array(self.client.getLinkState(self.body, 9)[0])
        dist = np.linalg.norm(lpad - rpad) - 0.047813
        return dist

    def check_proximity(self):
        ee_pos = np.array(self.client.getLinkState(self.robot, self.tool)[0])
        tool_pos = np.array(self.client.getLinkState(self.body, 0)[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = self.client.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac


def setup_raven_environment(gui=False, teleport=False):
    dt = 1 / 480
    if gui:
        client = bc.BulletClient(connection_mode=p.GUI)
        client.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=0,
            cameraPitch=-15,
            cameraTargetPosition=[0, 0.5, 0],
        )
    else:
        client = bc.BulletClient(connection_mode=p.DIRECT)

    client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    client.setPhysicsEngineParameter(enableFileCaching=0)
    assets_path = os.path.dirname(os.path.abspath(""))
    client.setAdditionalSearchPath(assets_path)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.setTimeStep(dt)

    client.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    client.setGravity(0, 0, -9.8)

    # Temporarily disable rendering to load URDFs faster.
    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # Add robot.
    client.loadURDF("plane.urdf", [0, 0, -0.001])
    robot_id = client.loadURDF(
        os.path.join(MODELS_PATH, "ur5e/ur5e.urdf"),
        [0, 0, 0],
        flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
    )
    joint_ids = [
        client.getJointInfo(robot_id, i) for i in range(client.getNumJoints(robot_id))
    ]
    joint_ids = [j[0] for j in joint_ids if j[2] == p.JOINT_REVOLUTE]

    # Move robot to home configuration.
    for i in range(len(joint_ids)):
        client.resetJointState(robot_id, joint_ids[i], DEFAULT_JOINT_ANGLES[i])

    # Add gripper.
    gripper = Robotiq2F85(robot_id, EE_LINK_ID, teleport=teleport, client=client)
    gripper.release()

    # Add workspace.
    plane_shape = client.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            (TABLE_BOUNDS[0][1] - TABLE_BOUNDS[0][0]) / 2.0,
            (TABLE_BOUNDS[1][1] - TABLE_BOUNDS[1][0]) / 2.0,
            0.001,
        ],
    )
    plane_visual = client.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            (TABLE_BOUNDS[0][1] - TABLE_BOUNDS[0][0]) / 2.0,
            (TABLE_BOUNDS[1][1] - TABLE_BOUNDS[1][0]) / 2.0,
            0.001,
        ],
    )
    plane_id = client.createMultiBody(
        0, plane_shape, plane_visual, basePosition=TABLE_CENTER
    )
    client.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])
    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return client, gripper, robot_id, joint_ids


class RavenYCBEnv(Environment):
    def __init__(
        self,
        task: Task,
        render: bool = False,
        seed: int = None,
        teleport: bool = False,
        is_twin: bool = False,
        record_video: bool = False,
        **kwargs,
    ):
        super().__init__(task)

        self.teleport = teleport
        if is_twin:
            self.log_prefix = "[Twin]"
        else:
            self.log_prefix = "[Main]"

        self.attachments = []
        if seed is None:
            self.seed = np.random.randint(1, 2**8)
        else:
            self.seed = seed

        self.sim_step = 0
        (
            self.client,
            self.gripper,
            self.robot_id,
            self.joint_ids,
        ) = setup_raven_environment(gui=render, teleport=self.teleport)

        for _ in range(50):
            self.step_sim_and_render(teleport=False)

        self.gripper.set_reset_state()

        self.internal_state = None

        self.record_video = record_video
        if self.record_video:

            self.video_recorder = self.client.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                os.path.join(get_log_dir(), f"replay.mp4"),
            )

    def close(self):
        if self.record_video:

            # Stop recording video
            self.client.stopStateLogging(self.video_recorder)

        # Take image of last state of sim
        camera_image, _, _, _, _ = self.get_camera_image_side(
            image_size=(460 * 2, 640 * 2)
        )
        imageio.imsave(os.path.join(get_log_dir(), f"final_frame.png"), camera_image)

    @staticmethod
    def sample_twin(
        real_env: RavenYCBEnv, belief: RavenBelief, task: Task, render: bool = False
    ) -> RavenYCBEnv:
        twin_state = copy.deepcopy(belief)
        twin_env = RavenYCBEnv(task=task, teleport=True, render=render, is_twin=True)
        for obj_name, object in twin_state.objects.items():
            obj_id = create_object(
                object.category, object.color, client=twin_env.client
            )
            pbu.set_pose(obj_id, object.pose.to_pbu(), client=twin_env.client)
            twin_state.objects[obj_name].body = obj_id
        twin_env.internal_state = twin_state
        twin_env.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        twin_env.reset()

        return twin_env

    def xyz_to_pix(self, position):
        """Convert from 3D position to pixel location on heightmap."""
        u = int(np.round((position[1] - TABLE_BOUNDS[1, 0]) / PIXEL_SIZE))
        v = int(np.round((position[0] - TABLE_BOUNDS[0, 0]) / PIXEL_SIZE))
        return (u, v)

    def reset(self):
        self.attachments = []
        self.gripper.reset()

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            self.client.resetJointState(
                self.robot_id, self.joint_ids[i], DEFAULT_JOINT_ANGLES[i]
            )

        if self.internal_state is None:
            self.internal_state = self.task.setup_env(client=self.client)

        # Re-enable rendering.
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        for _, obj in self.internal_state.objects.items():
            self.client.resetBasePositionAndOrientation(
                int(obj.body), obj.pose.point, obj.pose.quat
            )

        # pbu.wait_if_gui(client=self.client)
        return self.get_observation()

    def apply_attachments(self):
        for attachment in self.attachments:
            attachment.assign(client=self.client)

    def servoj(self, joints, teleport: bool):
        """Move to target joint positions with position control."""
        if teleport:
            pbu.set_joint_positions(
                self.robot_id, self.joint_ids, joints, client=self.client
            )
            self.gripper.apply_transform()
            self.apply_attachments()

        self.client.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[0.005] * 6,
        )

    def movep(self, pose: RavenPose, teleport: bool):
        """Move to target end effector position."""

        joints = self.client.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=TIP_LINK_ID,
            targetPosition=pose.point,
            targetOrientation=self.client.getQuaternionFromEuler(pose.euler),
            maxNumIterations=100,
        )
        self.servoj(joints, teleport=teleport)

    def get_env_collisions(self):
        collisions_check = {
            "gripper finger": self.gripper.body,
            "robot arm body": self.robot_id,
        } | {"held object": a.child for a in self.attachments}
        collision_messages = []
        for obj_name, obj in self.internal_state.objects.items():
            if obj.body not in collisions_check.values():
                for cc_name, cc in collisions_check.items():
                    if pbu.pairwise_collision(cc, obj.body, client=self.client):
                        collision_message = f"{self.log_prefix} Collision detected between {obj_name} object {cc_name}"
                        log.info(collision_message)
                        collision_messages.append(collision_message)
                        # pbu.wait_if_gui(client=self.client)
        return collision_messages

    def name_from_id(self, body_id):
        for k, v in self.internal_state.objects.items():
            if v.body == body_id:
                return k
        assert False

    def add_pick_attachments(self):
        # Currently, we decide kinematic attachment by distance between object centroid and tool tip
        for obj_name, obj in self.internal_state.objects.items():
            obj_pose = pbu.get_pose(obj.body, client=self.client)
            world_T_tool = pbu.get_link_pose(
                self.robot_id, TIP_LINK_ID, client=self.client
            )
            dist = np.linalg.norm(np.array(obj_pose[0]) - np.array(world_T_tool[0]))
            tool_T_obj = pbu.multiply(pbu.invert(world_T_tool), obj_pose)
            print("Pick attachment dist: " + str(dist))
            if dist < 0.025:
                self.attachments.append(
                    pbu.Attachment(
                        self.robot_id,
                        TIP_LINK_ID,
                        tool_T_obj,
                        obj.body,
                        client=self.client,
                    )
                )

    def move(self, dest: RavenPose, teleport: bool, max_steps=500):
        ee_pose = RavenPose.from_pbu(
            self.client.getLinkState(self.robot_id, TIP_LINK_ID)
        )
        step = 0

        while dest.dist(ee_pose) > 0.001 and step < max_steps:
            self.movep(dest, teleport=teleport)
            ee_pose = RavenPose.from_pbu(
                self.client.getLinkState(self.robot_id, TIP_LINK_ID)
            )
            self.step_sim_and_render(teleport=teleport)
            ee_pose = RavenPose.from_pbu(
                self.client.getLinkState(self.robot_id, TIP_LINK_ID)
            )
            if teleport:
                break
            step += 1

        return dest.dist(ee_pose) <= 0.001

    def step(self, action: Action):
        """Do pick and place motion primitive."""

        log.info(f"{self.log_prefix} executing action: " + str(action))

        collisions = []
        ik_success = True

        # # Check bounds
        # x, y, z = action.params
        # if x < TABLE_BOUNDS[0][0]:
        #     return None, 0, False, {"constraint_violations": ["x < TABLE_BOUNDS"]}
        # elif x > TABLE_BOUNDS[0][1]:
        #     return None, 0, False, {"constraint_violations": ["x > TABLE_BOUNDS"]}
        # elif y < TABLE_BOUNDS[1][0]:
        #     return None, 0, False, {"constraint_violations": ["y < TABLE_BOUNDS"]}
        # elif y > TABLE_BOUNDS[1][1]:
        #     return None, 0, False, {"constraint_violations": ["y > TABLE_BOUNDS"]}

        # Set fixed primitive z-heights.
        if action.name == "pick":
            obj_name, grasp = action.params
            obj_body = self.internal_state.objects[obj_name].body
            obj_pose = self.internal_state.objects[obj_name].pose
            gripper_target = obj_pose.multiply(grasp)
            hover_distance = 0.15
            hover_pose = RavenPose(z=hover_distance).multiply(
                copy.deepcopy(gripper_target)
            )
            gripper_target = RavenPose(z=-0.015).multiply(gripper_target)

            # Move to prepick
            log.info(f"{self.log_prefix} Moving to hover")
            ik_success &= self.move(hover_pose, teleport=self.teleport)
            collisions += self.get_env_collisions()

            # Move to pick
            log.info(f"{self.log_prefix} Moving to grasp")
            ik_success &= self.move(gripper_target, teleport=self.teleport)
            collisions += self.get_env_collisions()

            if len(collisions) == 0:
                world_T_lpad = self.client.getLinkState(self.gripper.body, 4)
                world_T_rpad = self.client.getLinkState(self.gripper.body, 9)

                # Perform collision checking across the line
                rayResults = p.rayTest(world_T_lpad[0], world_T_rpad[0])

                # Process the collision results
                collided_objects = [objectId for objectId, _, _, _, _ in rayResults]
                # self.client.addUserDebugLine(world_T_lpad[0], world_T_rpad[0], [1, 0, 0], 2)

                if self.teleport:
                    # log.info(len(collided_objects))
                    # log.info(str(self.gripper))
                    # log.info("{}, {}".format(obj_body, collided_objects))
                    if len(collided_objects) != 1 or obj_body not in collided_objects:
                        return (
                            None,
                            0,
                            False,
                            {"constraint_violations": ["Infeasible grasp"]},
                        )

                # Simulate grasps
                # pbu.wait_if_gui(client=self.client)

                # Close the gripper
                log.info(f"{self.log_prefix} Closing gripper")
                pose_before_lift = pbu.get_pose(obj_body, client=self.client)

                self.gripper.activate()
                for _ in range(500):
                    self.step_sim_and_render(teleport=False)

                # pbu.wait_if_gui(client=self.client)

                # Back to prepick
                log.info(f"{self.log_prefix} Moving back to hover")
                self.move(hover_pose, teleport=False)

                pose_after_lift = pbu.get_pose(obj_body, client=self.client)
                height_diff = (
                    RavenPose(z=hover_distance).multiply(
                        RavenPose.from_pbu(pose_before_lift)
                    )
                ).dist(RavenPose.from_pbu(pose_after_lift))

                # pbu.wait_if_gui(client=self.client)
                if self.teleport and height_diff > 0.05:
                    return None, 0, False, {"constraint_violations": ["Unstable grasp"]}

                if self.teleport:
                    self.gripper.reset()
                    pbu.set_pose(obj_body, pose_before_lift, client=self.client)
                    self.move(gripper_target, teleport=self.teleport)
                    self.add_pick_attachments()
                    log.info(
                        f"{self.log_prefix} Pick added {len(self.attachments)} attachments"
                    )

                collisions += self.get_env_collisions()

        elif action.name == "place":
            obj_name, grasp, world_T_target = action.params
            obj_pose = self.internal_state.objects[obj_name].pose
            obj_body = self.internal_state.objects[obj_name].body
            gripper_target = world_T_target.multiply(grasp)
            hover_pose = copy.deepcopy(gripper_target)
            hover_pose = RavenPose(z=0.15).multiply(gripper_target)

            # Move to place location.
            log.info(f"{self.log_prefix} Moving to place location")
            ik_success &= self.move(hover_pose, teleport=self.teleport)
            collisions += self.get_env_collisions()

            # Place down object.
            log.info(f"{self.log_prefix} Placing object")
            ik_success &= self.move(gripper_target, teleport=self.teleport)
            collisions += self.get_env_collisions()

            pose_before_place = pbu.get_pose(obj_body, client=self.client)

            # Open gripper
            # Simulate the object falling
            self.gripper.release()
            for _ in range(250):
                self.step_sim_and_render(teleport=False)
            self.gripper.reset()
            for _ in range(250):
                self.step_sim_and_render(teleport=False)

            pose_after_place = pbu.get_pose(obj_body, client=self.client)
            pose_diff = RavenPose.from_pbu(pose_before_place).dist(
                RavenPose.from_pbu(pose_after_place)
            )

            log.info(pose_diff)
            if self.teleport and pose_diff > 0.1:
                return None, 0, False, {"constraint_violations": ["Unstable placement"]}

            if self.teleport:
                # Release kinematic attachments
                self.attachments = []

            # back to preplace
            log.info(f"{self.log_prefix} Move up a little after placing")
            ik_success &= self.move(hover_pose, teleport=self.teleport)

        log.info(f"{self.log_prefix} Getting observation")

        if not self.teleport:
            observation = self.get_observation()
            reward = self.get_reward()
        else:
            observation = None
            reward = None

        done = False

        info = {"constraint_violations": collisions}

        if not ik_success:
            info["constraint_violations"].append("IK Failure")

        return observation, reward, done, info

    def set_alpha_transparency(self, alpha: float) -> None:
        for id in range(20):
            visual_shape_data = self.client.getVisualShapeData(id)
            for i in range(len(visual_shape_data)):
                object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
                rgba_color = list(rgba_color[0:3]) + [alpha]
                self.client.changeVisualShape(
                    self.robot_id, linkIndex=i, rgbaColor=rgba_color
                )
                self.client.changeVisualShape(
                    self.gripper.body, linkIndex=i, rgbaColor=rgba_color
                )

    def step_sim_and_render(self, teleport: bool):
        if not teleport:
            self.client.stepSimulation()
            self.gripper.update_gripper()
            # time.sleep(0.001)
        self.sim_step += 1

    def get_camera_image_side(
        self,
        image_size=(240, 240),
        focal_length=1000.0,
        position=(0, -1.55, 0.60),
        orientation=(np.pi / 2.5, np.pi, np.pi),
    ):
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # self.set_alpha_transparency(0)
        camera_image = self.render_image(
            image_size, focal_length, position, orientation
        )
        # self.set_alpha_transparency(1)
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return camera_image

    def get_camera_image_top(
        self,
        image_size=(240, 240),
        focal_len=2000.0,
        position=(0, -0.5, 5),
        orientation=(0, np.pi, -np.pi / 2),
        set_alpha=True,
    ):
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # set_alpha and self.set_alpha_transparency(0)
        camera_image = self.render_image(image_size, focal_len, position, orientation)
        # set_alpha and self.set_alpha_transparency(1)
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return camera_image

    def get_reward(self):
        return self.task.get_reward(self)

    def get_observation(self):
        observation = {}

        # Render current image.
        side_camera_image = self.get_camera_image_top()

        # Get heightmaps and colormaps.
        color, depth, position, orientation, intrinsics = side_camera_image
        points = get_pointcloud(depth, intrinsics)
        position = np.float32(position).reshape(3, 1)
        rotation = self.client.getMatrixFromQuaternion(orientation)
        rotation = np.float32(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(points, transform)
        colormap = self.get_heightmap(points, color, TABLE_BOUNDS, PIXEL_SIZE)

        observation["image"] = colormap
        observation["pointcloud"] = points
        observation["image_top"] = self.get_camera_image_top()
        observation["image_side"] = side_camera_image
        observation["seed"] = self.seed
        observation["internal_state"] = self.internal_state
        return observation

    def render_image(
        self,
        image_size=(240, 240),
        focal_len=2000,
        position=(0, -0.5, 5),
        orientation=(0, np.pi, -np.pi / 2),
    ):
        # Camera parameters.
        orientation = self.client.getQuaternionFromEuler(orientation)
        noise = True

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self.client.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        znear, zfar = (0.01, 10.0)
        viewm = self.client.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = self.client.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self.client.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = zfar + znear - (2 * zbuffer - 1) * (zfar - znear)
        depth = (2 * znear * zfar) / depth
        if noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.zeros((3, 3))

        intrinsics[0, 0] = focal_len
        intrinsics[1, 1] = focal_len
        intrinsics[0, 2] = image_size[0] / 2.0
        intrinsics[1, 2] = image_size[0] / 2.0

        return color, depth, position, orientation, intrinsics

    def get_heightmap(self, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D
        pointcloud.

        Args:
          points: HxWx3 float array of 3D points in world coordinates.
          colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
          bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
          pixel_size: float defining size of each pixel in meters.
        Returns:
          heightmap: HxW float array of height (from lower z-bound) in meters.
          colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
          xyzmap: HxWx3 float array of XYZ points in world coordinates.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (
            points[Ellipsis, 0] < bounds[0, 1]
        )
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (
            points[Ellipsis, 1] < bounds[1, 1]
        )
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (
            points[Ellipsis, 2] < bounds[2, 1]
        )
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
        colormap = colormap[::-1, :, :]  # Flip up-down.

        return colormap
