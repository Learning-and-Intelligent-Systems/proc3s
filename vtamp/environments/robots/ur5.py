import logging
import os
import time

import numpy as np
import pybullet as p

import vtamp.environments.pb_utils as pbu
from vtamp.environments.robots.robot import Robot
from vtamp.environments.utils import MODELS_PATH, Pose

log = logging.getLogger(__name__)

EE_LINK_ID = 9
TIP_LINK_ID = 10
DEFAULT_JOINT_ANGLES = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0]

FINGER_LATERAL_FRICTION = 5.0
FINGER_ROLLING_FRICTION = 5.0
FINGER_SPINNING_FRICTION = 5.0


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


class UR5Robot(Robot):
    def __init__(self, client, teleport=False):
        super(UR5Robot, self).__init__(client)

        self.client = client
        self.robot_id = client.loadURDF(
            os.path.join(MODELS_PATH, "ur5e/ur5e.urdf"),
            [0, 0, 0],
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        self.joint_ids = [
            self.client.getJointInfo(self.robot_id, i)
            for i in range(self.client.getNumJoints(self.robot_id))
        ]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            self.client.resetJointState(
                self.robot_id, self.joint_ids[i], DEFAULT_JOINT_ANGLES[i]
            )

        # Add gripper.
        self.gripper = Robotiq2F85(
            self.robot_id, EE_LINK_ID, teleport=teleport, client=self.client
        )
        self.gripper.release()

    def reset_gripper(self):
        self.gripper.reset()

    def get_collision_map(self):
        return {
            "gripper finger": self.gripper.body,
            "robot arm body": self.get_id(),
        }

    def get_finger_segment(self):
        world_T_lpad = self.client.getLinkState(self.gripper.body, 4)
        world_T_rpad = self.client.getLinkState(self.gripper.body, 9)
        return (world_T_lpad[0], world_T_rpad[0])

    def set_gripper_reset_state(self):
        self.gripper.set_reset_state()

    def get_id(self):
        return self.robot_id

    def activate_gripper(self, teleport=False):
        return self.gripper.activate()

    def release_gripper(self, teleport=False):
        return self.gripper.release()

    def maintain_gripper(self):
        return self.gripper.update_gripper()

    def get_tool_pose(self):
        return pbu.get_link_pose(self.robot_id, TIP_LINK_ID, client=self.client)

    def get_tool_id(self):
        return TIP_LINK_ID

    def apply_gripper(self):
        self.gripper.apply_transform()

    def get_joint_ids(self):
        return self.joint_ids

    def servoj(self, joints, teleport: bool):
        """Move to target joint positions with position control."""
        if teleport:
            pbu.set_joint_positions(
                self.get_id(), self.get_joint_ids(), joints, client=self.client
            )
            self.apply_gripper()
            self.apply_attachments()

        self.client.setJointMotorControlArray(
            bodyIndex=self.get_id(),
            jointIndices=self.get_joint_ids(),
            controlMode=p.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[0.005] * len(self.get_joint_ids()),
        )

    def move(self, pose: Pose, teleport: bool):
        """Move to target end effector position."""

        joints = self.client.calculateInverseKinematics(
            bodyUniqueId=self.get_id(),
            endEffectorLinkIndex=self.get_tool_id(),
            targetPosition=pose.point,
            targetOrientation=self.client.getQuaternionFromEuler(pose.euler),
            maxNumIterations=100,
        )
        self.servoj(joints[: len(self.get_joint_ids())], teleport=teleport)
