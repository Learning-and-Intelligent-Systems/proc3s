import logging

import numpy as np
import pybullet

from vtamp.environments.raven.env import PICK_TARGETS, PLACE_TARGETS

log = logging.getLogger(__name__)


class ScriptedPolicy:
    def __init__(self, env):
        self.env = env

    def step(self, text, obs):
        log.info(f"Input: {text}")

        # Parse pick and place targets.
        pick_text, place_text = text.split("and")
        pick_target, place_target = None, None
        for name in PICK_TARGETS.keys():
            if name in pick_text:
                pick_target = name
                break
        for name in PLACE_TARGETS.keys():
            if name in place_text:
                place_target = name
                break

        # Admissable targets only.
        assert pick_target is not None
        assert place_target is not None

        pick_id = self.env.obj_name_to_id[pick_target]
        pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
        pick_position = np.float32(pick_pose[0])

        if place_target in self.env.obj_name_to_id:
            place_id = self.env.obj_name_to_id[place_target]
            place_pose = pybullet.getBasePositionAndOrientation(place_id)
            place_position = np.float32(place_pose[0])
        else:
            place_position = np.float32(PLACE_TARGETS[place_target])

        # Add some noise to pick and place positions.
        # pick_position[:2] += np.random.normal(scale=0.01)
        place_position[:2] += np.random.normal(scale=0.01)

        act = {"pick": pick_position, "place": place_position}
        return act
