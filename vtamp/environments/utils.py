from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import vtamp.environments.pb_utils as pbu

MODELS_PATH = os.path.join(os.path.dirname(__file__), "../models/")


@dataclass
class Pose:
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

    def set_euler(self, euler):
        self.roll, self.pitch, self.yaw = euler

    @staticmethod
    def from_pbu(pose):
        euler = pbu.euler_from_quat(pose[1])
        return Pose(*pose[0], *euler)

    def dist(self, pose: Pose, rot_scale: float = 1e-2) -> float:
        pos_distance, ori_distance = pbu.get_pose_distance(self.to_pbu(), pose.to_pbu())
        return pos_distance + ori_distance * rot_scale

    def multiply(self, pose: Pose) -> Pose:
        return Pose.from_pbu(pbu.multiply(self.to_pbu(), pose.to_pbu()))


@dataclass
class Action:
    name: str = "default"
    params: List[float] = field(default_factory=list)


def parse_lisp(action_str: str) -> Action:
    assert action_str[0] == "(" and action_str[-1] == ")"
    parts = action_str[1:-1].split(" ")
    return Action(parts[0], [float(p) for p in parts[1:]])


@dataclass
class State:
    pass


class Task(ABC):
    @abstractmethod
    def get_goal(self):
        pass

    @abstractmethod
    def get_reward(self, env):
        pass

    @abstractmethod
    def setup_env(self, **kwargs):
        pass


class Updater(ABC):
    def __init__(self):
        pass

    def update(self, obs):
        raise NotImplementedError


class DefaultUpdater(Updater):
    def __init__(self):
        pass

    def update(self, obs):
        return obs


class Environment(ABC):
    @abstractmethod
    def __init__(self, task: Task = None, robot_type="ur5", **kwargs):
        self.task = task
        self.robot_type=robot_type
        self.param_scale = 1

    @abstractmethod
    def step(self, action: Action, return_belief: bool = False, profile_stats={}):
        raise NotImplementedError

    @abstractmethod
    def sample_twin(env, obs, task, **kwargs) -> Environment:
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass
