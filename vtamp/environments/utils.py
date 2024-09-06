from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


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
    def __init__(self, task: Task = None, **kwargs):
        self.task = task
        self.param_scale = 1

    @abstractmethod
    def step(self, action: Action, return_belief: bool = False, profile_stats={}):
        raise NotImplementedError

    @abstractmethod
    def sample_twin(env, obs, task) -> Environment:
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass
