import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from show_calibration import load_calibration

from vtamp.environments.pb_utils import set_joint_positions, tform_from_pose
from vtamp.environments.robots.panda import ARM_GROUP, PANDA_GROUPS, PandaRobot


def convert_to_robot_frame(robot: PandaRobot, capture_dir: str, client):
    # Get ref2world transformation
    q = robot.sender.get_joint_states()
    arm_joint_names = PANDA_GROUPS[ARM_GROUP]
    q = [q[jn] for jn in arm_joint_names]
    set_joint_positions(
        robot.get_id(), robot.get_group_joints(ARM_GROUP), q, client=client
    )

    pose = tform_from_pose(robot.get_camera_pose())

    # Load calibration, apply ref2world to the poses
    calibration = load_calibration(capture_dir, pose_fname="pose.npy")
    coords = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)]

    for camera_serial, values in calibration.items():
        c2w = values["c2w"]
        composed = pose @ c2w
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord.transform(composed)
        coords.append(coord)
        cam_dir = os.path.join(capture_dir, camera_serial)
        savefile = os.path.join(cam_dir, "pose_gt.npy")
        np.save(savefile, composed)

    # o3d.visualization.draw_geometries(coords)
    print("converted to robot frame ok")


if __name__ == "__main__":
    _ref_cam_sn = "231122071284"
    convert_to_robot_frame("captures/2023-10-25_18-20-11", _ref_cam_sn)
