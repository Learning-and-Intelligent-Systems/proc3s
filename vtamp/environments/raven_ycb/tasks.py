import copy
import logging
import random
import sys

import numpy as np
import pybullet as p

import vtamp.environments.pb_utils as pbu
from vtamp.environments.raven.tasks import RavenTask
from vtamp.environments.raven_ycb.env import (
    TABLE_BOUNDS,
    TABLE_CENTER,
    RavenObject,
    RavenPose,
    RavenState,
    RavenYCBEnv,
    create_object,
)
from vtamp.environments.utils import Task

TABLE_AABB = pbu.AABB(
    lower=[TABLE_BOUNDS[0][0], TABLE_BOUNDS[1][0], TABLE_BOUNDS[2][0]],
    upper=[TABLE_BOUNDS[0][1], TABLE_BOUNDS[1][1], TABLE_BOUNDS[2][1]],
)
log = logging.getLogger(__name__)


def get_random_position(object_coords):
    """Get random position 15cm+ from other objects."""
    while True:
        rand_x = np.random.uniform(TABLE_BOUNDS[0, 0] + 0.1, TABLE_BOUNDS[0, 1] - 0.1)
        rand_y = np.random.uniform(TABLE_BOUNDS[1, 0] + 0.1, TABLE_BOUNDS[1, 1] - 0.1)
        rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
        if len(object_coords) == 0:
            # object_coords = np.concatenate((object_coords, rand_xyz), axis=0)
            # break
            return rand_xyz
        else:
            nn_dist = np.min(np.linalg.norm(object_coords - rand_xyz, axis=1)).squeeze()
            if nn_dist > 0.15:
                return rand_xyz


def fixed_objects_stack(obj_names, client=None):
    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    obj_xyz = np.zeros((0, 3))
    object_ids = []
    bottom_blocks = []
    shuffled_obj_names = copy.deepcopy(obj_names)
    random.shuffle(shuffled_obj_names)
    clear_idxs = []
    # edit this loop to allow blocks to be stacked on top of each other
    for i, (color, category) in enumerate(shuffled_obj_names):
        object_id = create_object(category, color, client=client)
        if category == "block":
            # decide whether to stack block. Always stack at least one block
            stack = len(bottom_blocks) == 0 or np.random.choice([True, False])
            if stack and len(clear_idxs) > 0:
                # choose random block to stack
                base_block_idx = np.random.choice(clear_idxs)
                clear_idxs = [c for c in clear_idxs if c != base_block_idx]
                base_block_xyz = obj_xyz[base_block_idx]
                bottom_blocks.append(shuffled_obj_names[base_block_idx])
                # set new block on top of base block
                curr_block_xyz = base_block_xyz + np.array([0, 0, 0.02])
                obj_xyz = np.concatenate(
                    (obj_xyz, curr_block_xyz.reshape(1, 3)), axis=0
                )
            else:
                # place block on table away from other objects
                curr_block_xyz = get_random_position(obj_xyz)
                obj_xyz = np.concatenate((obj_xyz, curr_block_xyz), axis=0)

            object_position = curr_block_xyz.squeeze()

            clear_idxs.append(i)

        elif category == "bowl":
            rand_xyz = get_random_position(obj_xyz)
            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)

            object_position = rand_xyz.squeeze()

        object_ids.append(object_id)
        pbu.set_pose(object_id, pbu.Pose(pbu.Point(*object_position)), client=client)

    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    for _ in range(200):
        client.stepSimulation()

    objects = []
    for object_id, (color, category) in zip(object_ids, shuffled_obj_names):
        pose = pbu.get_pose(object_id, client=client)
        objects.append(
            RavenObject(category, color, RavenPose.from_pbu(pose), object_id)
        )

    state = RavenState(objects={f"object_{i}": obj for i, obj in enumerate(objects)})
    return bottom_blocks, state


def fixed_objects_random_location(
    obj_names, aabb=TABLE_AABB, client=None
) -> RavenState:
    object_ids = []
    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    log.info("Placing ycb objects...")
    for color, category in obj_names:
        # Get random position 15cm+ from other objects.
        new_obj = create_object(category, color, client=client)
        object_ids.append(new_obj)

    max_attempts = 1000
    for i in range(max_attempts):
        print("Placing attempt {}".format(i))
        for object_id in object_ids:
            pose = pbu.sample_placement_on_aabb(
                object_id, aabb, percent=0.8, client=client
            )
            pbu.set_pose(object_id, pose, client=client)

        if all(
            [
                not pbu.pairwise_collisions(oid, object_ids, client=client)
                for oid in object_ids
            ]
        ):
            break
    else:
        print("Placing failed")
        sys.exit()

    for _ in range(200):
        client.stepSimulation()

    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    objects = []
    for object_id, (color, category) in zip(object_ids, obj_names):
        pose = pbu.get_pose(object_id, client=client)
        objects.append(
            RavenObject(category, color, RavenPose.from_pbu(pose), object_id)
        )
    return RavenState(objects={f"object_{i}": obj for i, obj in enumerate(objects)})


class BlocksInRavenYCB(Task):
    def setup_env(self, **kwargs) -> RavenState:
        objects = [
            ("yellow", "block"),
            ("green", "block"),
            ("blue", "block"),
            ("green", "bowl"),
        ]
        return fixed_objects_random_location(objects, **kwargs)


class PackingYCB(RavenTask):
    def __init__(self, goal_str="", **kwargs):
        self.goal_str = goal_str

    def setup_env(self, client=None):
        objects = [
            ("yellow", "banana"),
            ("red", "strawberry"),
            ("red", "strawberry"),
        ]

        object_pose_map = fixed_objects_random_location(
            objects, aabb=pbu.scale_aabb(TABLE_AABB, 0.8), client=client
        )

        plane_shape = client.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.24 / 2.0, 0.24 / 2.0, 0.0014]
        )
        plane_visual = client.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.24 / 2.0, 0.24 / 2.0, 0.0015]
        )
        plane_id = client.createMultiBody(
            0, plane_shape, plane_visual, basePosition=TABLE_CENTER
        )
        client.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

        # region_aabb = pbu.get_aabb(plane_id, client=client)

        return object_pose_map

    def get_reward(self, env: RavenYCBEnv):
        return 1

    def get_goal(self):
        return self.goal_str


class StackingYCB(RavenTask):
    def __init__(self, goal_str="", **kwargs):
        self.goal_str = goal_str

    def setup_env(self, client=None):
        objects = [
            ("red", "strawberry"),
            ("blue", "potted_meat_can"),
            ("red", "apple"),
            ("green", "pear"),
            ("orange", "power_drill"),
            ("yellow", "banana"),
        ]

        object_pose_map = fixed_objects_random_location(objects, client=client)

        return object_pose_map

    def get_reward(self, env: RavenYCBEnv):
        return 1

    def get_goal(self):
        return self.goal_str


class HammerYCB(RavenTask):
    def __init__(self, goal_str="", **kwargs):
        self.goal_str = goal_str

    def setup_env(self, **kwargs):
        objects = [("black", "banana")]
        raven_state = fixed_objects_random_location(objects, **kwargs)
        return raven_state

    def get_reward(self, env: RavenYCBEnv):
        return 1

    def get_goal(self):
        return self.goal_str
