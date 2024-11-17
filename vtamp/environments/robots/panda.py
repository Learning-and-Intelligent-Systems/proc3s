import logging
import math
import os
import time

import numpy as np
import pybullet as p

import vtamp.environments.pb_utils as pbu
from vtamp.environments.robots.panda_sender import PandaSender
from vtamp.environments.robots.robot import Robot
from vtamp.environments.utils import MODELS_PATH, Pose
from vtamp.environments.pb_utils import Attachment
import itertools
from typing import List
import random
log = logging.getLogger(__name__)


# Original
# DEFAULT_JOINT_ANGLES = [
#     -0.0806406098426434,
#     -1.6722951504174777,
#     0.07069076842695393,
#     -2.7449419709102822,
#     0.08184716251979611,
#     1.7516337599063168,
#     0.7849295270972781,
# ]

# Top down
DEFAULT_JOINT_ANGLES = [-0.08585239216881181, -1.6772606901708562, 0.04586290604326021, -2.5769458161404257, 0.05899769418183296, 1.468362243334452, 0.7847049503061588]
PANDA_IGNORE_COLLISIONS = {(6, 9), (12, 13)}
MAX_TOOL_DISTANCE = np.inf
COLLISION_EPSILON = 1e-3
COLLISION_DISTANCE = 5e-3
PANDA_TOOL_TIP = "panda_tool_tip"
ARM_GROUP = "main_arm"
GRIPPER_GROUP = "main_gripper"
PANDA_GROUPS = {
    "base": [],
    "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
    "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
}

CAMERA_OPTICAL_FRAME = "camera_frame"

class PandaRobot(Robot):
    def __init__(self, client, teleport=False, real_robot=False):
        super(PandaRobot, self).__init__(client)

        self.teleport = teleport
        self.real_robot = real_robot

        self.robot_id = client.loadURDF(
            os.path.join(MODELS_PATH, "franka_panda/panda.urdf"),
            [0, 0.1, 0],
            pbu.quat_from_euler([0, 0, -math.pi / 2.0]),
            useFixedBase=True,
        )
        self.joint_ids = self.get_group_joints(ARM_GROUP)

        if self.real_robot:
            self.sender = PandaSender()
            real_joint_position_dict = self.sender.get_joint_states()
            real_joint_positions = [
                real_joint_position_dict[jn] for jn in PANDA_GROUPS[ARM_GROUP]
            ]
            pbu.set_joint_positions(
                self.robot_id,
                self.get_joint_ids(),
                real_joint_positions,
                client=self.client,
            )

        self.servoj(DEFAULT_JOINT_ANGLES, teleport=True)

        self.release_gripper(teleport=self.teleport)

    def solve_ik(self, link, target_pose, start_q=None, obstacles=[]):
        randomize_seed = start_q is None
        max_attempts = 100
        arm_joints = self.get_group_joints(ARM_GROUP)
        ranges = [
            pbu.get_joint_limits(self.robot_id, joint, client=self.client)
            for joint in arm_joints
        ]

        for i in range(max_attempts):
            # Start with the current joint positions and then randomize within limits after
            if not randomize_seed:
                initialization_sample = start_q
                randomize_seed = True
            else:
                initialization_sample = [random.uniform(r[0], r[1]) for r in ranges]

            pbu.set_joint_positions(
                self.robot_id, arm_joints, initialization_sample, client=self.client
            )

            conf = self.client.calculateInverseKinematics(
                int(self.robot_id),
                link,
                target_pose[0],
                target_pose[1],
                residualThreshold=0.00001,
                maxNumIterations=5000,
            )

            # Need to extract the arm component of the returned joints
            conf = [
                q
                for q, j in zip(conf, pbu.get_movable_joints(self.robot_id, client=self.client))
                if j in arm_joints
            ]

            lower, upper = list(zip(*ranges))

            assert len(arm_joints) == len(conf)
            pbu.set_joint_positions(self.robot_id, arm_joints, conf, client=self.client)

            if not pbu.all_between(lower, conf, upper):
                print("IK solution outside limits")
                continue

            contact_points = []
            for obstacle in obstacles:
                contact_points += self.client.getClosestPoints(
                    bodyA=obstacle, bodyB=self.robot_id, distance=pbu.MAX_DISTANCE
                )

            all_joints = pbu.get_joints(self.robot_id, client=self.client)
            check_link_pairs = pbu.get_self_link_pairs(
                self.robot_id, all_joints, PANDA_IGNORE_COLLISIONS, client=self.client
            )

            self_collision = False
            for link1, link2 in check_link_pairs:
                if pbu.pairwise_link_collision(
                    self.robot_id, link1, self.robot_id, link2, client=self.client
                ):
                    print(link1, link2)
                    self_collision = True

            if self_collision:
                print("Self collision")
                continue

            # Print contact points if there are any
            if contact_points:
                print("Collision!")
                continue

            pose = pbu.get_link_pose(self.robot_id, link, client=self.client)
            trans_diff, rot_diff = pbu.get_pose_distance(target_pose, pose)

            if trans_diff < 0.001 and rot_diff < 0.01:
                return list(conf)
            else:
                print("IK Error: {}, {}".format(trans_diff, rot_diff))

        return None




    def get_collision_fn(
        self,
        joints: List[int],
        obstacles: List[int] = [],
        attachments: List[Attachment] = [],
        self_collisions: bool = True,
        disabled_collisions=set(),
        custom_limits={},
        use_aabb=False,
        cache=False,
        max_distance=pbu.MAX_DISTANCE,
        extra_collisions=None,
    ):
        check_link_pairs = (
            pbu.get_self_link_pairs(
                self.robot_id, joints, disabled_collisions, client=self.client
            )
            if self_collisions
            else []
        )
        moving_links = frozenset(
            link
            for link in pbu.get_moving_links(self.robot_id, joints, client=self.client)
            if pbu.can_collide(self.robot_id, link, client=self.client)
        )
        attached_bodies = [attachment.child for attachment in attachments]
        moving_bodies = [pbu.CollisionPair(self.robot_id, moving_links)] + list(
            map(pbu.parse_body, attached_bodies)
        )

        get_obstacle_aabb = pbu.cached_fn(
            pbu.get_buffered_aabb,
            cache=cache,
            max_distance=max_distance / 2.0,
            client=self.client,
        )
        limits_fn = pbu.get_limits_fn(
            self.robot_id, joints, custom_limits=custom_limits, client=self.client
        )

        def collision_fn(q, verbose=False):
            if limits_fn(q):
                return True

            pbu.set_joint_positions(self.robot_id, joints, q, client=self.client)

            for attachment in attachments:
                world_T_child = pbu.multiply(
                    pbu.get_link_pose(
                        attachment.parent, attachment.parent_link, client=self.client
                    ),
                    attachment.parent_T_child,
                )
                pbu.set_pose(attachment.child, world_T_child, client=self.client)

                # Check if the attachment is in collision with objects in the environment
                if any(
                    pbu.pairwise_collision(attachment.child, body, client=self.client)
                    for body in obstacles
                ):
                    return True

            if extra_collisions is not None and extra_collisions(client=self.client):
                return True

            get_moving_aabb = pbu.cached_fn(
                pbu.get_buffered_aabb,
                cache=True,
                max_distance=max_distance / 2.0,
                client=self.client,
            )

            for link1, link2 in check_link_pairs:
                if (
                    not use_aabb
                    or pbu.aabb_overlap(
                        get_moving_aabb(self.robot_id), get_moving_aabb(self.robot_id)
                    )
                ) and pbu.pairwise_link_collision(
                    self.robot_id, link1, self.robot_id, link2, client=self.client
                ):
                    print("Link on link collision")
                    print(self.robot_id, link1, self.robot_id, link2)
                    return True

            for body1, body2 in itertools.product(moving_bodies, obstacles):
                if (
                    not use_aabb
                    or pbu.aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
                ) and pbu.pairwise_collision(body1, body2, client=self.client):
                    print("Body on body collision")
                    print(body1, body2)
                    return True
            return False

        return collision_fn


    def plan_workspace_motion(
        self,
        robot: int,
        tool_waypoints: List[pbu.Pose],
        attachment: Attachment = None,
        obstacles: List[int] = [],
        max_attempts=2,
        debug=False,
        start_q=None,
        client=None,
    ) -> List[List[float]]:
        """Return a joint path that moves the tool along the tool waypoints.

        This is useful if you want to move the gripper in a straight line
        path.
        """
        assert tool_waypoints

        tool_link = self.get_tool_id()
        parts = [self.robot_id] + ([] if attachment is None else [attachment.child])
        arm_joints = self.get_group_joints(ARM_GROUP)

        collision_fn = self.get_collision_fn(
            arm_joints,
            obstacles=obstacles,
            attachments=[],
            self_collisions=True,
        )

        for attempts in range(max_attempts):
            arm_conf = self.solve_ik(tool_link, tool_waypoints[0], start_q=start_q)

            if arm_conf is None or collision_fn(arm_conf):
                continue

            arm_waypoints = [arm_conf]
            for tool_pose in tool_waypoints[1:]:
                arm_conf = self.solve_ik(tool_link, tool_pose, start_q=arm_waypoints[-1])
                if arm_conf is None or collision_fn(arm_conf):
                    break

                arm_waypoints.append(arm_conf)
            else:
                pbu.set_joint_positions(
                    self.robot_id, arm_joints, arm_waypoints[-1], client=self.client
                )
                if attachment is not None:
                    attachment.assign()
                if any(
                    pbu.pairwise_collisions(
                        part,
                        obstacles,
                        max_distance=(COLLISION_DISTANCE + COLLISION_EPSILON),
                        client=self.client,
                    )
                    for part in parts
                ):
                    if debug:
                        pbu.wait_if_gui(client=self.client)
                    continue
                arm_path = pbu.interpolate_joint_waypoints(
                    self.robot_id, arm_joints, arm_waypoints, client=self.client
                )

                if any(collision_fn(q) for q in arm_path):
                    if debug:
                        pbu.wait_if_gui(client=self.client)
                    continue

                print(
                    "Found path with {} waypoints and {} configurations after {} attempts".format(
                        len(arm_waypoints), len(arm_path), attempts + 1
                    )
                )

                return arm_path
        return None

    def reset_gripper(self):
        self.release_gripper(teleport=True)

    def set_gripper_reset_state(self):
        pass

    def get_gripper_id(self):
        raise NotImplementedError

    def get_id(self):
        return self.robot_id

    def get_group_joints(self, group):
        return pbu.joints_from_names(
            self.robot_id, PANDA_GROUPS[group], client=self.client
        )

    def get_camera_pose(self):
        return pbu.get_link_pose(
            self.robot_id,
            pbu.link_from_name(self.robot_id, CAMERA_OPTICAL_FRAME, client=self.client),
            client=self.client,
        )

    def get_group_limits(self, group):
        return pbu.get_custom_limits(
            self.robot_id, self.get_group_joints(group), client=self.client
        )

    def get_collision_map(self):
        return {
            "robot arm": self.robot_id,
        }

    def get_finger_segment(self):
        world_T_lpad = self.client.getLinkState(
            self.robot_id,
            pbu.link_from_name(self.robot_id, "panda_leftfinger", client=self.client),
        )
        world_T_rpad = self.client.getLinkState(
            self.robot_id,
            pbu.link_from_name(self.robot_id, "panda_rightfinger", client=self.client),
        )
        shifted_l = pbu.multiply(world_T_lpad, pbu.Pose([0,0,0.02]))
        shifted_r = pbu.multiply(world_T_rpad, pbu.Pose([0,0,0.02]))

        return shifted_l[0], shifted_r[0]

    def get_finger_joint_ids(self):
        return pbu.joints_from_names(
            self.get_id(), names=PANDA_GROUPS["main_gripper"], client=self.client
        )

    def release_gripper(self, teleport=False):  # These are mirrored on the pr2
        _, open_conf = self.get_group_limits(GRIPPER_GROUP)

        if teleport:
            pbu.set_joint_positions(
                self.get_id(),
                self.get_finger_joint_ids(),
                open_conf,
                client=self.client,
            )
        
        self.client.setJointMotorControlArray(
            bodyIndex=self.get_id(),
            jointIndices=self.get_finger_joint_ids(),
            controlMode=p.POSITION_CONTROL,
            targetPositions=open_conf,
            positionGains=[0.005, 0.005],
        )

        if self.real_robot:
            self.sender.open_gripper()

    def activate_gripper(self, teleport=False):  # These are mirrored on the pr2
        closed_conf, _ = self.get_group_limits(GRIPPER_GROUP)

        if teleport:
            pbu.set_joint_positions(
                self.get_id(),
                self.get_finger_joint_ids(),
                closed_conf,
                client=self.client,
            )
        
        self.client.setJointMotorControlArray(
            bodyIndex=self.get_id(),
            jointIndices=self.get_finger_joint_ids(),
            controlMode=p.POSITION_CONTROL,
            targetPositions=closed_conf,
            positionGains=[0.005, 0.005],
        )

        if self.real_robot:
            self.sender.close_gripper()

    def execute_trajecotry(self, trajectory, teleport):
        
        joints = self.get_joint_ids()

        if teleport:
            pbu.set_joint_positions(
                self.get_id(), joints, trajectory[-1], client=self.client
            )
            self.apply_gripper()
            self.apply_attachments()
        self.client.setJointMotorControlArray(
            bodyIndex=self.get_id(),
            jointIndices=self.get_joint_ids(),
            controlMode=p.POSITION_CONTROL,
            targetPositions=trajectory[-1],
            positionGains=[0.005] * len(self.get_joint_ids()),
        )
        
        if self.real_robot:
            pdicts = [{name: val for name, val in zip(PANDA_GROUPS[ARM_GROUP], target_positions)} for target_positions in trajectory]
            pdicts = [self.sender.get_joint_states()]+pdicts
            self.sender.execute_position_path(pdicts)
            return True

    def servoj(self, target_positions, teleport):
        joints = self.get_joint_ids()

        if teleport:
            pbu.set_joint_positions(
                self.get_id(), joints, target_positions, client=self.client
            )
            self.apply_gripper()
            self.apply_attachments()
        self.client.setJointMotorControlArray(
            bodyIndex=self.get_id(),
            jointIndices=self.get_joint_ids(),
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            positionGains=[0.005] * len(self.get_joint_ids()),
        )

        if self.real_robot:
            pdict = {name: val for name, val in zip(PANDA_GROUPS[ARM_GROUP], target_positions)}
            self.sender.move_to_joint_positions(pdict)
            return True

    def move(self, target_pose: Pose, teleport: bool, interp=False):
        """Move to target end effector position."""
        
        joints = self.get_group_joints(ARM_GROUP)  

        if(interp and not teleport):
            current_joint_positions = self.sender.get_joint_states()
            current_joint_positions = [
                current_joint_positions[j] for j in PANDA_GROUPS[ARM_GROUP]
            ]
            
            pbu.set_joint_positions(self.robot_id, joints, current_joint_positions, client=self.client)
            current_pose = pbu.get_link_pose(self.robot_id, link=self.get_tool_id(), client=self.client)

            # Start with the current joint positions and then randomize within limits after
            intermediate_poses = list(pbu.interpolate_poses(current_pose, target_pose.to_pbu()))
            trajectory = self.plan_workspace_motion(self.robot_id, tool_waypoints=intermediate_poses, start_q=current_joint_positions, attachment=None, obstacles=[], client=self.client)
            return self.execute_trajecotry(trajectory, teleport=teleport)
        else:
            target_positions = self.client.calculateInverseKinematics(
                bodyUniqueId=self.get_id(),
                endEffectorLinkIndex=self.get_tool_id(),
                targetPosition=target_pose.point,
                targetOrientation=self.client.getQuaternionFromEuler(target_pose.euler),
                maxNumIterations=100,
            )
            target_positions = target_positions[: len(self.get_joint_ids())]
            return self.servoj(target_positions, teleport=teleport)

    # def command_group_trajectory(self, group, positions):
    #     current_joint_positions = self.sender.get_joint_states()
    #     current_joint_positions = [
    #         current_joint_positions[j] for j in PANDA_GROUPS[group]
    #     ]

    #     if not pbu.all_close(np.array(current_joint_positions), np.array(positions[0])):
    #         positions = [current_joint_positions] + positions

    #     pdicts = [
    #         {name: val for name, val in zip(PANDA_GROUPS[group], position)}
    #         for position in positions
    #     ]
    #     self.sender.execute_position_path(pdicts)
    #     new_current_joint_positions = self.sender.get_joint_states()
    #     new_current_joint_positions = [
    #         new_current_joint_positions[j] for j in PANDA_GROUPS[group]
    #     ]
    #     pbu.set_joint_positions(
    #         self.robot_id,
    #         self.get_joint_ids(),
    #         new_current_joint_positions,
    #         client=self.client,
    #     )

    def maintain_gripper(self):
        pass

    def get_tool_pose(self):
        return pbu.get_link_pose(self.robot_id, self.get_tool_id(), client=self.client)

    def get_tool_id(self):
        return pbu.link_from_name(self.robot_id, PANDA_TOOL_TIP, client=self.client)

    def apply_gripper(self):
        pass

    def get_joint_ids(self):
        return self.joint_ids
