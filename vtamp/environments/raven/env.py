from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import tensorflow.compat.v1 as tf  # type: ignore
import torch

import vtamp.environments.pb_utils as pbu
from vtamp.environments.robots.panda import PandaRobot
from vtamp.environments.robots.robot import Robot
from vtamp.environments.robots.ur5 import UR5Robot
from vtamp.environments.utils import (
    MODELS_PATH,
    Action,
    Environment,
    Pose,
    Task,
    Updater,
)
from vtamp.utils import get_log_dir
import trimesh
import vtamp.perception.grounded_sam as gsam

log = logging.getLogger(__name__)



# Hammer fall params
# LATERAL_FRICTION = 10.0
# ROLLING_FRICTION = 0.001
# SPINNING_FRICTION = 0.001

# Normal params
OBJ_LATERAL_FRICTION = 0.5
OBJ_ROLLING_FRICTION = 0.1
OBJ_SPINNING_FRICTION = 0.1

BLOCK_SIZE = 0.04
# COLORS = {
    # "blue": (78 / 255, 121 / 255, 167 / 255, 255 / 255),
    # "red": (255 / 255, 87 / 255, 89 / 255, 255 / 255),
    # "green": (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    # "yellow": (237 / 255, 201 / 255, 72 / 255, 255 / 255),
    # "orange": (251 / 255, 106 / 255, 74 / 255, 255 / 255),
    # "purple": (123 / 255, 102 / 255, 210 / 255, 255 / 255),
    # "pink": (247 / 255, 104 / 255, 161 / 255, 255 / 255),
    # "teal": (68 / 255, 170 / 255, 153 / 255, 255 / 255),
# }
COLORS = {
    "blue": (70 / 255, 70 / 255, 255 / 255, 255 / 255),
    "red": (255 / 255, 70 / 255, 70 / 255, 255 / 255),
    # "green": (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    # "yellow": (237 / 255, 201 / 255, 72 / 255, 255 / 255),
    # "orange": (251 / 255, 106 / 255, 74 / 255, 255 / 255),
    # "purple": (123 / 255, 102 / 255, 210 / 255, 255 / 255),
    # "pink": (247 / 255, 104 / 255, 161 / 255, 255 / 255),
    # "teal": (68 / 255, 170 / 255, 153 / 255, 255 / 255),
}



PIXEL_SIZE = 0.00267857
TABLE_BOUNDS = np.float32([[-0.4, 0.4], [-0.8, -0.2], [0, 0]])
WORKSPACE_SIZE = 0.3
TABLE_CENTER = [0, -0.5, 0]

# Used as imports for the LLM-generated code
__all__ = [
    "RavenPose",
    "RavenObject",
    "RavenBelief",
    "RavenState",
    "TABLE_BOUNDS",
    "BLOCK_SIZE",
    "TABLE_CENTER",
]


class RavenPose(Pose):
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


# A temporary hack because vision sucks
class RavenGroundTruthBeliefUpdater(Updater):
    def update(self, obs) -> RavenBelief:
        return obs["internal_state"]


class RavenVisionBeliefUpdater(Updater):
    def __init__(self, category_names = {"bowl":"bowl", "block":"block"}, box_threshold=0.2, text_threshold=0.2):
        
        self.last_belief = None
        self.category_names = category_names

        self.grounded_checkpoint = "./checkpoints/gsam/groundingdino_swint_ogc.pth"
        self.sam_checkpoint = "./checkpoints/gsam/sam_vit_h_4b8939.pth"
        self.config = "./checkpoints/gsam/config.py"

        self.device = "cpu"
        # self.model = gsam.load_model(
        #     self.config, self.grounded_checkpoint, device=self.device
        # )
        # self.box_threshold = box_threshold
        # self.text_threshold = text_threshold
        # self.sam_version = "vit_h"

        # self.predictor = gsam.SamPredictor(
        #     gsam.sam_model_registry[self.sam_version](
        #         checkpoint=self.sam_checkpoint
        #     ).to(self.device)
        # )


    def closest_predefined_color(self, pointcloud):
        """Find the predefined color closest to the mean color of the pointcloud."""
        
        # Convert pointcloud to a NumPy array if it is not already
        pointcloud = np.array(pointcloud)
        
        # Validate pointcloud
        if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
            raise ValueError("pointcloud should be a 2D array with at least 3 columns for RGB values")
        
        # Check if the pointcloud is empty
        if pointcloud.size == 0:
            raise ValueError("pointcloud is empty")
        
        # Compute the mean color of the pointcloud
        mean_color = np.mean(pointcloud[:, :3], axis=0)
        print("mean color: "+str(mean_color))
        # Convert predefined colors to a NumPy array
        color_names = []
        color_values = []
        for color_name, color_value in COLORS.items():
            color_names.append(color_name)
            color_values.append(color_value)
        color_values = np.array(color_values)

        # Initialize variables to store the closest color
        closest_color_name = None
        min_distance = float("inf")

        # Iterate over each predefined color
        for color_value, color_name in zip(color_values, color_names):
            # Calculate the squared Euclidean distance from the mean color
            distance = np.linalg.norm(mean_color - color_value[:3])

            # Find the color with the smallest distance
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name

        print(closest_color_name)
        return closest_color_name

    def update(self, obs) -> RavenBelief:
        if self.last_belief is None:
            image_path = os.path.join(get_log_dir(), "tmp.png")
            camera_image: pbu.CameraImage = obs["image_side"]

            imageio.imsave(image_path, camera_image.rgbPixels)

            image_pil, image = gsam.load_image(image_path)
            # boxes_filt, pred_phrases = gsam.get_grounding_output(
            #     self.model,
            #     image,
            #     " . ".join(self.category_names.keys()),
            #     self.box_threshold,
            #     self.text_threshold,
            #     with_logits=False,
            #     device=self.device,
            # )
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # original_image = copy.deepcopy(image)

            # self.predictor.set_image(image)

            # size = image_pil.size
            # H, W = size[1], size[0]
            # for i in range(boxes_filt.size(0)):
            #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            #     boxes_filt[i][2:] += boxes_filt[i][:2]

            # boxes_filt = boxes_filt.cpu()
            # transformed_boxes = self.predictor.transform.apply_boxes_torch(
            #     boxes_filt, image.shape[:2]
            # ).to(self.device)

            # masks, _, _ = self.predictor.predict_torch(
            #     point_coords=None,
            #     point_labels=None,
            #     boxes=transformed_boxes.to(self.device),
            #     multimask_output=False,
            # )

            # # draw output image
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image)
            # for mask in masks:
            #     gsam.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            # for box, label in zip(boxes_filt, pred_phrases):
            #     gsam.show_box(box.numpy(), plt.gca(), label)

            # plt.axis("off")
            # plt.savefig(
            #     os.path.join(get_log_dir(), "grounded_sam_output.jpg"),
            #     bbox_inches="tight",
            #     dpi=300,
            #     pad_inches=0.0,
            # )

            # points = get_pointcloud(
            #     camera_image.depthPixels, camera_image.camera_matrix
            # )

            # position = np.float32(camera_image.camera_pose[0]).reshape(3, 1)
            # rotation = p.getMatrixFromQuaternion(camera_image.camera_pose[1])
            # rotation = np.float32(rotation).reshape(3, 3)
            # transform = np.eye(4)
            # transform[:3, :] = np.hstack((rotation, position))
            # pointcloud = transform_pointcloud(points, transform)

            self.last_belief = RavenBelief(observations=[obs])

            # for i, (box, category) in enumerate(zip(boxes_filt, pred_phrases)):
            #     segmentation = masks[i, ...].squeeze()
            #     seg_xs, seg_ys = np.where(segmentation > 0)
            #     mean_xyz = np.mean(pointcloud[seg_xs, seg_ys, :], axis=0)
            #     crop_rgb = original_image[seg_xs, seg_ys, :] / 256.0
            #     xyz = mean_xyz.tolist()
            #     xyz[2] = BLOCK_SIZE / 2.0
            #     object = RavenObject(
            #         category=self.category_names[category],
            #         color=self.closest_predefined_color(crop_rgb),
            #         pose=RavenPose(*xyz),
            #     )
            #     self.last_belief.objects[f"object_{i}"] = object

        new_belief = copy.deepcopy(self.last_belief)
        new_belief.observations.append(obs)
        return new_belief


def setup_raven_environment(
    robot_type="ur5", real_robot=False, gui=False, teleport=False
) -> Tuple[int, Robot]:
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

    if robot_type == "ur5":
        robot = UR5Robot(client)
    elif robot_type == "panda":
        robot = PandaRobot(client, real_robot=real_robot, teleport=teleport)

    # Add workspace.
    plane_shape = client.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.4, WORKSPACE_SIZE, 0.001]
    )
    plane_visual = client.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.4, WORKSPACE_SIZE, 0.001]
    )
    plane_id = client.createMultiBody(
        0, plane_shape, plane_visual, basePosition=TABLE_CENTER
    )
    client.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])
    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return client, robot


class RavenEnv(Environment):
    def __init__(
        self,
        task: Task,
        render: bool = False,
        teleport: bool = False,
        is_twin: bool = False,
        record_video: bool = False,
        stability_check: bool = False,
        robot_type: str = "ur5",
        real_robot: bool = False,
        real_camera: bool = False,
        image_observations: bool = False,
        **kwargs,
    ):
        super().__init__(task)
        self.teleport = teleport
        self.robot_type = robot_type
        self.stability_check = stability_check
        self.image_observations = image_observations

        if is_twin:
            self.log_prefix = "[Twin]"
        else:
            self.log_prefix = "[Main]"

        self.sim_step = 0
        (self.client, self.robot) = setup_raven_environment(
            robot_type=robot_type, gui=render, teleport=teleport, real_robot=real_robot
        )
        self.real_camera = real_camera
        self.internal_state = None
        self.record_video = record_video
        if self.record_video:
            self.video_recorder = self.client.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                os.path.join(get_log_dir(), f"replay.mp4"),
            )


    def close(self):
        if self.record_video:
            self.client.stopStateLogging(self.video_recorder)

        # Take image of last state of sim
        camera_image = self.get_camera_image_side()
        imageio.imsave(
            os.path.join(get_log_dir(), f"final_frame.png"), camera_image.rgbPixels
        )

    @staticmethod
    def sample_twin(
        real_env: RavenEnv,
        belief: RavenBelief,
        task: Task,
        robot_type: str = "ur5",
        render: bool = False,
        **kwargs,
    ) -> RavenEnv:
        twin_state = copy.deepcopy(belief)
        twin_env = RavenEnv(
            task=task,
            teleport=True,
            render=True,
            is_twin=True,
            robot_type=robot_type,
            stability_check=real_env.stability_check,
        )
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
        self.robot.attachments = []
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

    def get_env_collisions(self):
        collisions_check = self.robot.get_collision_map() | {
            "held object": a.child for a in self.robot.attachments
        }
        collision_messages = []
        for obj_name, obj in self.internal_state.objects.items():
            if obj.body not in collisions_check.values():
                for cc_name, cc in collisions_check.items():
                    if(cc is not None and obj.body is not None):
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
            world_T_tool = self.robot.get_tool_pose()
            dist = np.linalg.norm(np.array(obj_pose[0]) - np.array(world_T_tool[0]))
            tool_T_obj = pbu.multiply(pbu.invert(world_T_tool), obj_pose)
            if dist < 0.025:
                self.robot.add_attachment(
                    pbu.Attachment(
                        self.robot.get_id(),
                        self.robot.get_tool_id(),
                        tool_T_obj,
                        obj.body,
                        client=self.client,
                    )
                )

    def move(self, dest: Pose, max_steps=1000, teleport=False, interp=False):
        ee_pose = Pose.from_pbu(
            self.client.getLinkState(self.robot.get_id(), self.robot.get_tool_id())
        )
        step = 0
        while dest.dist(ee_pose) > 0.005 and step < max_steps:
            terminate = self.robot.move(dest, teleport=teleport, interp=interp)
            if terminate:
                break
            self.step_sim_and_render(teleport=teleport)
            ee_pose = RavenPose.from_pbu(self.robot.get_tool_pose())
            if teleport:
                break
            step += 1

        return dest.dist(ee_pose) <= 0.005

    def step(self, action: Action):
        """Do pick and place motion primitive."""

        log.info(f"{self.log_prefix} executing action: " + str(action))

        collisions = []
        ik_success = True

        # Check bounds
        x, y, z = action.params
        if x < TABLE_BOUNDS[0][0]:
            return None, 0, False, {"constraint_violations": ["x < TABLE_BOUNDS"]}
        elif x > TABLE_BOUNDS[0][1]:
            return None, 0, False, {"constraint_violations": ["x > TABLE_BOUNDS"]}
        elif y < TABLE_BOUNDS[1][0]:
            return None, 0, False, {"constraint_violations": ["y < TABLE_BOUNDS"]}
        elif y > TABLE_BOUNDS[1][1]:
            return None, 0, False, {"constraint_violations": ["y > TABLE_BOUNDS"]}

        # Set fixed primitive z-heights.
        if action.name == "pick":
            pick_pose = RavenPose(*action.params, pitch=np.pi, yaw=np.pi / 2.0)
            hover_pose = copy.deepcopy(pick_pose)
            hover_pose = RavenPose(z=0.15).multiply(pick_pose)
            pick_pose = RavenPose(z=-0.005).multiply(pick_pose)

            # Move to prepick
            log.info(f"{self.log_prefix} Moving to hover")
            print(hover_pose)
            ik_success &= self.move(hover_pose, teleport=self.teleport)
            collisions += self.get_env_collisions()

            # Move to pick
            log.info(f"{self.log_prefix} Moving to grasp")
            ik_success &= self.move(pick_pose, teleport=self.teleport, interp=True)
            # collisions += self.get_env_collisions()

            # pbu.wait_if_gui(client=self.client)
            # Close the gripper
            log.info(f"{self.log_prefix} Closing gripper")
            if not self.teleport:
                self.robot.activate_gripper()
                if not self.teleport:
                    for _ in range(240):
                        self.step_sim_and_render(teleport=self.teleport)
            else:
                self.add_pick_attachments()
                log.info(
                    f"{self.log_prefix} Pick added {len(self.robot.attachments)} attachments"
                )

            # Back to prepick
            log.info(f"{self.log_prefix} Moving back to hover")
            ik_success &= self.move(hover_pose, teleport=self.teleport, interp=True)
            collisions += self.get_env_collisions()

        elif action.name == "place":
            if self.teleport and len(self.robot.attachments) == 0:
                return (
                    None,
                    0,
                    False,
                    {
                        "constraint_violations": [
                            "Tried to place while not holding an object"
                        ]
                    },
                )

            place_pose = RavenPose(*action.params, pitch=np.pi, yaw=np.pi / 2.0)
            hover_pose = RavenPose(z=0.15).multiply(place_pose)
            place_pose = RavenPose(z=0.02).multiply(place_pose)

            # Move to place location.
            log.info(f"{self.log_prefix} Moving to place location")
            ik_success &= self.move(hover_pose, teleport=self.teleport)
            collisions += self.get_env_collisions()

            # Place down object.
            log.info(f"{self.log_prefix} Placing object")
            ik_success &= self.move(place_pose, teleport=self.teleport, interp=True)
            collisions += self.get_env_collisions()

            if self.teleport:
                pose_before_place = pbu.get_pose(
                    self.robot.attachments[0].child, client=self.client
                )
                # Open gripper
                self.robot.release_gripper(teleport=True)

                # Simulate the object falling
                for _ in range(500):
                    self.step_sim_and_render(teleport=False)

                pose_after_place = pbu.get_pose(
                    self.robot.attachments[0].child, client=self.client
                )
                pose_diff = RavenPose.from_pbu(pose_before_place).dist(
                    RavenPose.from_pbu(pose_after_place)
                )
                log.info("pose_diff: " + str(pose_diff))
                if self.teleport and pose_diff > 0.04 and self.stability_check:
                    return (
                        None,
                        0,
                        False,
                        {"constraint_violations": ["Unstable placement"]},
                    )

                # Release kinematic attachments
                self.robot.attachments = []
            else:
                # Open gripper
                self.robot.release_gripper()

                # Simulate the object falling
                for _ in range(500):
                    self.step_sim_and_render(teleport=self.teleport)

            # back to preplace
            log.info(f"{self.log_prefix} Move up a little after placing")
            ik_success &= self.move(hover_pose, teleport=self.teleport, interp=True)

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

        self.client.stepSimulation()
        # log.info(info["constraint_violations"])
        # pbu.wait_if_gui(client=self.client)

        return observation, reward, done, info

    def step_sim_and_render(self, teleport: bool):
        if not teleport:
            self.client.stepSimulation()
            self.robot.maintain_gripper()
            # time.sleep(0.001)
        self.sim_step += 1

    def get_camera_image_side(
        self,
        image_size=(460 * 2, 640 * 2),
        focal_length=1000.0,
        position=(0, -1.55, 0.60),
        orientation=(np.pi / 2.5, np.pi, np.pi),
    ):
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        camera_image = self.render_image(
            image_size, focal_length, position, orientation
        )
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return camera_image

    def get_reward(self):
        return self.task.get_reward(self)

    def get_image(self) -> pbu.CameraImage:
        if self.real_camera:
            rgb, depth, intrinsics = self.robot.sender.capture_realsense()

            # External camera
            # camera_tform = np.load(
            #     "/home/aidan/vlm-tamp/vtamp/environments/robots/calibration/captures/2024-08-12_16-36-54/032622073024/pose_gt.npy"
            # )
            # camera_pose = pbu.pose_from_tform(camera_tform)

            camera_pose = self.robot.get_camera_pose()
            camera_image = pbu.CameraImage(
                rgb, depth / 1000.0, None, camera_pose, intrinsics
            )
            return camera_image

        else:
            return self.get_camera_image_side(
                position=(0, -1, 0.5), orientation=(np.pi / 4.5, np.pi, np.pi)
            )

    def get_observation(self):
        observation = {}
        observation["image_side"] = self.get_image()
        observation["internal_state"] = self.internal_state
        return observation

    def render_image(
        self,
        image_size=(240, 240),
        focal_len=2000,
        position=(0, -0.5, 5),
        orientation=(0, np.pi, -np.pi / 2),
    ) -> pbu.CameraImage:
        # Camera parameters.
        orientation = self.client.getQuaternionFromEuler(orientation)

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
        _, _, color, depth, _ = self.client.getCameraImage(
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

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = zfar + znear - (2 * zbuffer - 1) * (zfar - znear)
        depth = (2 * znear * zfar) / depth

        intrinsics = np.zeros((3, 3))

        intrinsics[0, 0] = focal_len
        intrinsics[1, 1] = focal_len
        intrinsics[0, 2] = image_size[1] / 2.0  # Width divided by 2
        intrinsics[1, 2] = image_size[0] / 2.0  # Height divided by 2

        return pbu.CameraImage(color, depth, None, (position, orientation), intrinsics)

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
        heightmap = np.zeros((height, width), dtype=np.float32)
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
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
        colormap = colormap[::-1, :, :]  # Flip up-down.

        heightmap = heightmap[::-1, :]  # Flip up-down.
        return colormap
