import json
import os

import cv2
import numpy as np
import open3d as o3d


def load_calibration(calibration_dir: str, pose_fname: str = "pose_gt.npy") -> dict:
    # list all dirs in calibration_dir to get device serials
    device_serials = os.listdir(calibration_dir)
    # remove things not directories
    device_serials = [
        e for e in device_serials if os.path.isdir(os.path.join(calibration_dir, e))
    ]

    results = {}
    # color is in color/00000.png, depth is in depth/00000.png
    # load images
    for serial in device_serials:
        device_dir = os.path.join(calibration_dir, serial)
        color_dir = os.path.join(device_dir, "color")
        depth_dir = os.path.join(device_dir, "depth")

        color_image = cv2.imread(os.path.join(color_dir, "00000.png"))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        depth_image = cv2.imread(
            os.path.join(depth_dir, "00000.png"), cv2.IMREAD_ANYDEPTH
        )

        # intrinsics in intrinsics_color.json
        with open(os.path.join(device_dir, "intrinsics_color.json")) as f:
            intrinsics = json.load(f)

        # read pose.npy if it exists
        if os.path.exists(os.path.join(device_dir, pose_fname)):
            pose = np.load(os.path.join(device_dir, pose_fname))
        else:
            pose = None

        results[serial] = {
            "rgb": color_image,
            "depth": depth_image,
            "intrinsics": intrinsics,
            "pose": pose,
            "c2w": pose,
            "w2c": np.linalg.inv(pose),
        }
    return results
