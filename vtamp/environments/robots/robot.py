from abc import ABC, abstractmethod

import pybullet as p

import vtamp.environments.pb_utils as pbu
from vtamp.environments.utils import Pose


class Robot(ABC):
    @abstractmethod
    def __init__(self, client, **kwargs):
        self.client = client
        self.attachments = []

    @abstractmethod
    def move(self, gripper_pose):
        pass

    @abstractmethod
    def get_id(self):
        pass

    @abstractmethod
    def get_finger_segment(self):
        pass

    @abstractmethod
    def activate_gripper(self, teleport=False):
        pass

    @abstractmethod
    def release_gripper(self, teleport=False):
        pass

    @abstractmethod
    def maintain_gripper(self):
        pass

    @abstractmethod
    def get_tool_pose(self):
        pass

    @abstractmethod
    def servoj(self, target_positions, teleport):
        pass

    @abstractmethod
    def get_tool_id(self):
        pass

    @abstractmethod
    def reset_gripper(self):
        pass

    @abstractmethod
    def set_gripper_reset_state(self):
        pass

    def add_attachment(self, attachment):
        self.attachments.append(attachment)

    def apply_attachments(self):
        for attachment in self.attachments:
            attachment.assign(client=self.client)

    @abstractmethod
    def apply_gripper(self):
        pass

    @abstractmethod
    def get_joint_ids(self):
        pass

    @abstractmethod
    def get_collision_map(self):
        pass

    @abstractmethod
    def move(self, pose: Pose, teleport: bool):
        pass
