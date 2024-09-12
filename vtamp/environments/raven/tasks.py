import copy
import logging
import random

import numpy as np
import pybullet as p

import vtamp.environments.pb_utils as pbu
from vtamp.environments.raven.env import (
    BLOCK_SIZE,
    COLORS,
    TABLE_BOUNDS,
    TABLE_CENTER,
    RavenEnv,
    RavenObject,
    RavenPose,
    RavenState,
    create_object,
)
from vtamp.environments.utils import Task

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

                # Stacks of 4 are too high
                if base_block_xyz[2] > BLOCK_SIZE * 2:
                    # place block on table away from other objects
                    curr_block_xyz = get_random_position(obj_xyz)
                    obj_xyz = np.concatenate((obj_xyz, curr_block_xyz), axis=0)
                else:

                    curr_block_xyz = base_block_xyz + np.array([0, 0, BLOCK_SIZE])
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
    bottom_block_objs = []
    for object_id, (color, category) in zip(object_ids, shuffled_obj_names):
        pose = pbu.get_pose(object_id, client=client)
        ro = RavenObject(category, color, RavenPose.from_pbu(pose), object_id)
        objects.append(ro)
        if (color, category) in bottom_blocks:
            bottom_block_objs.append(ro)

    state = RavenState(objects={f"object_{obj.body}": obj for obj in objects})
    return bottom_block_objs, state


def fixed_objects_random_location(obj_names, client=None) -> RavenState:
    object_ids = []
    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    obj_xyz = np.zeros((0, 3))
    for color, category in obj_names:
        # Get random position 15cm+ from other objects.
        while True:
            rand_x = np.random.uniform(
                TABLE_BOUNDS[0, 0] + 0.1, TABLE_BOUNDS[0, 1] - 0.1
            )
            rand_y = np.random.uniform(
                TABLE_BOUNDS[1, 0] + 0.1, TABLE_BOUNDS[1, 1] - 0.1
            )
            rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
            if len(obj_xyz) == 0:
                obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                break
            else:
                nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                if nn_dist > 0.125:
                    obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                    break

        object_position = rand_xyz.squeeze()
        object_ids.append(create_object(category, color, client=client))
        pbu.set_pose(
            object_ids[-1], pbu.Pose(pbu.Point(*object_position)), client=client
        )

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


class RavenTask(Task):
    def setup_env(self, **kwargs) -> RavenState:
        objects = [
            ("yellow", "block"),
            ("green", "block"),
            ("blue", "block"),
            ("green", "bowl"),
        ]
        return fixed_objects_random_location(objects, **kwargs)


class OnePlace(RavenTask):
    def __init__(self, **kwargs):
        pass

    def get_reward(self, env: RavenEnv):
        # TODO
        # # Assuming bowls have a predetermined radius or size for simplicity
        # bowl_radius = 0.1  # Example radius of the bowl's effective area

        # bowl_poses = [
        #     pbu.get_pose(oid, client=self.env.client)
        #     for name, oid in env.obj_name_to_id.items()
        #     if "bowl" in name
        # ]
        # block_poses = [
        #     pbu.get_pose(oid, client=self.env.client)
        #     for name, oid in env.obj_name_to_id.items()
        #     if "block" in name
        # ]

        # for block_pose in block_poses:
        #     for bowl_pose in bowl_poses:
        #         # Calculate the distance between the block and the bowl in 2D
        #         distance = (
        #             (block_pose[0] - bowl_pose[0]) ** 2
        #             + (block_pose[1] - bowl_pose[1]) ** 2
        #         ) ** 0.5
        #         if distance < bowl_radius:
        #             return 1  # Return 1 if any block is inside any bowl's radius

        return 0

    def get_goal(self):
        return "Place one block in any bowl"


class UnstackTask(RavenTask):
    def __init__(self, goal_str=None, **kwargs):
        self.goal_str = goal_str

    def setup_env(self, **kwargs):
        objects = [
            ("yellow", "block"),
            ("blue", "block"),
            ("pink", "block"),
            ("orange", "block"),
            ("brown", "block"),
            ("teal", "block"),
            ("green", "bowl"),
        ]
        bottom_blocks, raven_state = fixed_objects_stack(objects, **kwargs)
        print(bottom_blocks)
        assert len(bottom_blocks) > 0

        target_block = random.choice(bottom_blocks)

        # Change color to green
        raven_state.objects[f"object_{target_block.body}"].color = "green"
        kwargs["client"].changeVisualShape(
            target_block.body, -1, rgbaColor=COLORS["green"]
        )

        return raven_state

    def get_reward(self, env: RavenEnv):
        return 1

    def get_goal(self):
        return self.goal_str


class Empty(RavenTask):
    def __init__(self, goal_str: str = "", **kwargs):
        self.goal_str = goal_str

    def setup_env(self, **kwargs):
        return RavenState()

    def get_reward(self, env: RavenEnv):
        return 1

    def get_goal(self):
        return self.goal_str


class LineHard(RavenTask):
    def __init__(self, goal_str: str = "", always_stacked=True, **kwargs):
        self.goal_str = goal_str
        self.always_stacked = always_stacked

    def setup_env(self, **kwargs):
        objects = [
            ("green", "bowl"),
            ("blue", "bowl"),
            ("red", "bowl"),
            ("yellow", "block"),
            ("green", "block"),
            ("blue", "block"),
            ("red", "block"),
            ("green", "block"),
            ("blue", "block"),
        ]
        return fixed_objects_random_location(objects, **kwargs)

    def get_reward(self, env: RavenEnv):
        return 1

    def get_goal(self):
        return self.goal_str


class FruitInBowl(RavenTask):
    def __init__(self, always_stacked=True, **kwargs):
        self.always_stacked = always_stacked

    def setup_env(self, **kwargs):
        objects = [
            ("red", "strawberry"),
            ("blue", "sponge"),
            ("orange", "power_drill"),
            ("green", "bowl"),
        ]

        return fixed_objects_random_location(objects, **kwargs)

    def get_reward(self, env: RavenEnv):
        return 1

    def get_goal(self):
        return "Put a fruit in a bowl"


class CustomTask(RavenTask):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_reward(self, env: RavenEnv):
        return 1

    def get_goal(self):
        return self.goal_str


class PackingTask(RavenTask):
    """Pack blocks in a circle region of radius 0.1."""

    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.object_ids = None

    def setup_env(self, client=None, **kwargs):
        objects = [
            ("red", "block"),
            ("red", "block"),
            ("red", "block"),
            ("red", "block"),
            ("red", "block"),
        ]
        state = fixed_objects_random_location(objects, client=client)
        object_pose_map_ids = [
            int(object.split("_")[1]) for object in state.objects.keys()
        ]
        self.object_ids = object_pose_map_ids
        # draw circle of radius 0.1

        plane_shape = client.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.16 / 2.0, 0.16 / 2.0, 0.0015]
        )
        plane_visual = client.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.16 / 2.0, 0.16 / 2.0, 0.0015]
        )
        plane_id = client.createMultiBody(
            0, plane_shape, plane_visual, basePosition=TABLE_CENTER
        )
        client.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

        log.info("Goal str: " + str(self.goal_str))

        return state

    def get_reward(self, env: RavenEnv):
        circle_center = [0, -0.5, 0]
        object_poses = [
            p.getBasePositionAndOrientation(oid)[0] for oid in self.object_ids
        ]
        # distance between objects and circle center should be less than 0.08
        for pose in object_poses:
            if np.linalg.norm(np.array(pose[:2]) - np.array(circle_center[:2])) > 0.08:
                return 0

        return 1

    def get_goal(self):
        return self.goal_str
