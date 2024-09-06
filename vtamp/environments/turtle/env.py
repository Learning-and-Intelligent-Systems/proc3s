from __future__ import annotations

import copy
import logging
import os
import random
from dataclasses import dataclass, field
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from shapely.geometry import LineString, Point

import vtamp.environments.pb_utils as pbu
from vtamp.environments.raven.env import (
    COLORS,
    HOME_EE_POSE,
    TABLE_BOUNDS,
    TABLE_CENTER,
    TIP_LINK_ID,
    WORKSPACE_SIZE,
    setup_raven_environment,
)
from vtamp.environments.utils import Action, Environment, State, Task
from vtamp.utils import get_log_dir

log = logging.getLogger(__name__)
# Used as imports for the LLM-generated code
__all__ = ["Obstacle", "TurtleState"]


@dataclass
class Obstacle:
    name: str
    x_pos: float
    y_pos: float
    radius: float
    color: str

    @property
    def shape(self):
        return Point([self.x_pos, self.y_pos]).buffer(self.radius)

    @property
    def scaled_shape(self):
        return Point([self.x_pos, self.y_pos]).buffer(self.radius + 1)

    def __str__(self):
        return 'Obstacle(name="{}", x_pos={}, y_pos={}, radius={}, color="{}")'.format(
            self.name,
            round(self.x_pos, 2),
            round(self.y_pos, 2),
            round(self.radius, 2),
            self.color,
        )


@dataclass
class Line:
    p1_x: float
    p1_y: float
    p2_x: float
    p2_y: float

    @property
    def shape(self):
        return LineString([[self.p1_x, self.p1_y], [self.p2_x, self.p2_y]])


@dataclass
class TurtleState(State):
    obstacles: List[Obstacle] = field(default_factory=list)
    drawn_lines: List[Line] = field(default_factory=list)

    def __str__(self):
        return "TurtleState(obstacles=[{}])".format(
            ", ".join([str(o) for o in self.obstacles])
        )


ENV_MIN = 0
ENV_MAX = 100


class TurtleEnv(Environment):
    def __init__(self, task: Task, **kwargs):

        super().__init__(task)

        obstacles, self.boundary_lines = self.task.setup_env()
        self.state = TurtleState(obstacles=obstacles)
        self.initial_state = copy.deepcopy(self.state)
        self.param_scale = 100
        self.reset()

    def step(self, action: Action):
        info = {"constraint_violations": []}

        if action.name == "draw_line":
            new_line = Line(*action.params)
            self.state.drawn_lines.append(new_line)

            obstacle_int = [
                new_line.shape.intersection(obstacle.scaled_shape)
                for obstacle in self.state.obstacles
            ]
            if any(obstacle_int):
                info["constraint_violations"] += [
                    self.state.obstacles[i].name
                    for i in range(len(self.state.obstacles))
                    if obstacle_int[i]
                ]

            for drawn_line in self.state.drawn_lines:
                for line_x, line_y in [
                    [drawn_line.p1_x, drawn_line.p1_y],
                    [drawn_line.p2_x, drawn_line.p2_y],
                ]:
                    if line_x < ENV_MIN:
                        info["constraint_violations"] += [
                            f"x coordinate below environment minimum of {ENV_MIN}"
                        ]
                    if line_y < ENV_MIN:
                        info["constraint_violations"] += [
                            f"y coordinate below environment minimum of {ENV_MIN}"
                        ]
                    if line_x > ENV_MAX:
                        info["constraint_violations"] += [
                            f"x coordinate above environment minimum of {ENV_MAX}"
                        ]
                    if line_y > ENV_MAX:
                        info["constraint_violations"] += [
                            f"y coordinate above environment maximum of {ENV_MAX}"
                        ]
        elif action.name == "remove_obstacle":
            self.state.obstacles = [
                o for o in self.state.obstacles if o.name != action.params[0]
            ]
        else:
            raise NotImplementedError

        self.t = self.t + 1

        return self.state, False, 0, info

    @staticmethod
    def sample_twin(real_env: TurtleEnv, obs, task: Task, **kwargs) -> TurtleEnv:
        twin = TurtleEnv(task)
        twin.state = copy.deepcopy(obs)
        twin.initial_state = copy.deepcopy(obs)
        twin.reset()
        return twin

    def reset(self):
        self.state = copy.deepcopy(self.initial_state)
        self.t = 0
        return self.state

    def render(self):
        # Plot the star
        fig, ax = plt.subplots()
        for boundary in self.boundary_lines:
            x, y = boundary.shape.xy
            ax.plot(
                x,
                y,
                color="black",
                alpha=1.0,
                linewidth=2,
                solid_capstyle="round",
                zorder=2,
            )
            ax.set_aspect("equal")

        for o in self.state.obstacles:
            x, y = o.shape.exterior.xy
            ax.fill(x, y, alpha=0.8, fc=o.color, ec="none")

        for line in self.state.drawn_lines:
            x, y = line.shape.xy
            ax.plot(
                x,
                y,
                color="red",
                alpha=1.0,
                linewidth=2,
                solid_capstyle="round",
                zorder=2,
            )

        plt.draw()

        save_dir = os.path.join(get_log_dir(), "env_t={}.pdf".format(self.t))
        fig.savefig(save_dir, bbox_inches="tight")

        plt.pause(0.5)
        plt.close()


def get_obstacles(num_obstacles=5):
    boundary_points = [
        (ENV_MIN, ENV_MIN),
        (ENV_MIN, ENV_MAX),
        (ENV_MAX, ENV_MAX),
        (ENV_MAX, ENV_MIN),
    ]
    boundary_lines = [
        Line(*(boundary_points[i] + boundary_points[(i + 1) % len(boundary_points)]))
        for i in range(len(boundary_points))
    ]

    obstacles = []
    while len(obstacles) < num_obstacles:
        center = [random.uniform(ENV_MIN, ENV_MAX) for _ in range(2)]
        radius = random.uniform(4, 15)
        obstacle = Obstacle(
            name=f"obstacle_{len(obstacles)}",
            x_pos=center[0],
            y_pos=center[1],
            radius=radius,
            color=random.choice(list(COLORS.keys())),
        )
        if not any(
            [
                obstacle.shape.intersects(bline.shape)
                for bline in boundary_lines + obstacles
            ]
        ):
            obstacles.append(obstacle)
    return obstacles, boundary_lines


class DrawShape(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return get_obstacles(5)

    def get_reward(self, env):
        return 0


class DrawShapeHard(Task):
    def __init__(self, goal: str, **kwargs):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def setup_env(self):
        return get_obstacles(100)

    def get_reward(self, env):
        return 0


def interpolate_arrays(arr1, arr2, threshold):
    diffs = np.abs(arr2 - arr1)
    max_diff = np.max(diffs)
    num_points = int(np.ceil(max_diff / threshold)) + 1
    interpolated = np.array(
        [np.linspace(x, y, num_points) for x, y in zip(arr1, arr2)]
    ).T
    return interpolated


class TurtleEnvEmbodied(TurtleEnv):
    def __init__(
        self,
        task: Task,
        render: bool = True,
        teleport: bool = False,
        record_video=False,
        **kwargs,
    ):
        super().__init__(task)

        self.teleport = teleport

        (
            self.client,
            self.gripper,
            self.robot_id,
            self.joint_ids,
        ) = setup_raven_environment(gui=render, teleport=teleport)
        self.gripper_T_marker = pbu.Pose(point=pbu.Point(z=0.18))
        self.marker = pbu.create_cylinder(
            0.008, 0.08, 0.1, color=pbu.RED, client=self.client
        )
        self.client.changeDynamics(
            self.marker,
            -1,
            lateralFriction=0.05,
            spinningFriction=0.05,
            rollingFriction=0.05,
            frictionAnchor=True,
        )

        gripper_pose = pbu.get_pose(self.gripper.body, client=self.client)
        self.gripper.activate()
        for _ in range(400):
            self.step_sim_and_render()

        pbu.set_pose(
            self.marker,
            pbu.multiply(gripper_pose, self.gripper_T_marker),
            client=self.client,
        )
        for _ in range(400):
            self.step_sim_and_render()

        gripper_pose = pbu.get_pose(self.gripper.body, client=self.client)
        # Updated relative pose after sim
        self.gripper_T_marker = pbu.multiply(
            pbu.invert(gripper_pose), (pbu.get_pose(self.marker, client=self.client))
        )

        obstacle_height = 0.03
        for obstacle in self.state.obstacles:
            obstacle_body = pbu.create_cylinder(
                obstacle.radius / 100.0 * WORKSPACE_SIZE * 2,
                obstacle_height,
                pbu.STATIC_MASS,
                color=COLORS[obstacle.color],
                client=self.client,
            )
            pbu.set_pose(
                obstacle_body,
                pbu.Pose(
                    point=pbu.Point(
                        x=obstacle.x_pos / 100.0 * WORKSPACE_SIZE * 2
                        - WORKSPACE_SIZE
                        + TABLE_CENTER[0],
                        y=obstacle.y_pos / 100.0 * WORKSPACE_SIZE * 2
                        - WORKSPACE_SIZE
                        + TABLE_CENTER[1],
                        z=obstacle_height / 2.0,
                    )
                ),
                client=self.client,
            )

        self.record_video = record_video
        if self.record_video:
            self.video_recorder = self.client.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                os.path.join(get_log_dir(), f"replay.mp4"),
            )

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

    def close(self):
        if self.record_video:
            self.client.stopStateLogging(self.video_recorder)

        # Take image of last state of sim
        camera_image, _, _, _, _ = self.get_camera_image_side(
            image_size=(460 * 2, 640 * 2)
        )
        imageio.imsave(os.path.join(get_log_dir(), f"final_frame.png"), camera_image)

    def apply_attachments(self):
        for attachment in self.attachments:
            attachment.assign(client=self.client)

    def servoj(self, joints):
        """Move to target joint positions with position control."""
        if self.teleport:
            pbu.set_joint_positions(
                self.robot_id, self.joint_ids, joints, client=self.client
            )
            self.gripper.apply_transform()
            self.apply_attachments()
        else:
            self.client.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.joint_ids,
                controlMode=self.client.POSITION_CONTROL,
                targetPositions=joints,
                positionGains=[0.005] * 6,
            )

    def step_sim_and_render(self):
        if not self.teleport:
            self.client.stepSimulation()
            self.gripper.update_gripper()

    def movep(self, position):
        """Move to target end effector position."""
        joints = self.client.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=TIP_LINK_ID,
            targetPosition=position,
            targetOrientation=self.client.getQuaternionFromEuler(HOME_EE_POSE.euler),
            maxNumIterations=100,
        )
        self.servoj(joints)

    def marker_in_hand(self):
        gripper_pose = pbu.get_pose(self.gripper.body, client=self.client)
        intended_marker_pose = pbu.multiply(gripper_pose, self.gripper_T_marker)
        real_marker_pose = pbu.get_pose(self.marker, client=self.client)
        distance = pbu.get_pose_distance(intended_marker_pose, real_marker_pose)
        log.info(distance)
        return distance[0] < 0.1 and distance[1] < 0.6

    def move_to_target(self, target_xyz, add_paint=False):
        log.info("Moving to target")
        ee_xyz = np.float32(self.client.getLinkState(self.robot_id, TIP_LINK_ID)[0])
        step_t = 0
        while np.linalg.norm(target_xyz - ee_xyz) > 0.01:
            if add_paint and step_t % 5 == 0 and self.marker_in_hand():
                paint = pbu.create_cylinder(
                    0.008,
                    0.001,
                    color=pbu.RGBA(1, 0, 0, 0.5),
                    collision=False,
                    client=self.client,
                )
                pbu.set_pose(
                    paint,
                    pbu.Pose(pbu.Point(x=ee_xyz[0], y=ee_xyz[1], z=0.01)),
                    client=self.client,
                )
            self.movep(target_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(self.client.getLinkState(self.robot_id, TIP_LINK_ID)[0])
            step_t += 1

    def step(self, action: Action):

        for line_x, line_y in [
            [action.params[0], action.params[1]],
            [action.params[2], action.params[3]],
        ]:
            if (
                line_x < ENV_MIN
                or line_y < ENV_MIN
                or line_x > ENV_MAX
                or line_y > ENV_MAX
            ):
                return super().step(action)

        scaled_action = (
            np.array(action.params) / 100.0 * WORKSPACE_SIZE * 2 - WORKSPACE_SIZE
        )

        from_xyz = np.array([scaled_action[0], scaled_action[1], 0.06]) + TABLE_CENTER
        self.move_to_target(from_xyz + np.array([0, 0, 0.1]))
        self.move_to_target(from_xyz)

        to_xyz = np.array([scaled_action[2], scaled_action[3], 0.06]) + TABLE_CENTER
        for interm_to_xyz in interpolate_arrays(from_xyz, to_xyz, threshold=0.05):
            self.move_to_target(interm_to_xyz, add_paint=True)
        self.move_to_target(interm_to_xyz + np.array([0, 0, 0.1]))
        return super().step(action)

    def render(self):
        pass
