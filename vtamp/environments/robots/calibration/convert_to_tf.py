import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from show_calibration import load_calibration

serial_to_cam_name = {
    "231122071284": "wrist_cam",
    "819612071034": "front_cam",
    "231122071283": "front_right_cam",
    "032622073024": "front_left_cam",
    "818312070030": "right_cam",
    "819612071978": "left_cam",
}

base_template = """
<?xml version="1.0" ?>
<launch>
{cams}
</launch>
""".strip()

wrist_cam = """
    <!-- Wrist cam -->
    <node pkg="tf2_ros" type="static_transform_publisher" name="wrist_cam_link_broadcaster"
          args="0.036499 -0.034889 0.0574 0.00252743 0.0065769 0.70345566 0.71070423 panda_hand wrist_cam_color_optical_frame"/>
"""

cam_template = """
    <!-- {cam_name} -->
    <node pkg="tf2_ros" type="static_transform_publisher" name="{cam_name}_link_broadcaster"
            args="{x} {y} {z} {qx} {qy} {qz} {qw} panda_link0 {cam_name}_color_optical_frame"/>
"""

additional_save_paths = [
    "/home/aidan/control_ws/src/tampura_ros/launch/extrinsics.launch"
]


def convert_to_tf(capture_dir: str):
    calibration = load_calibration(capture_dir)
    cam_xmls = []

    for serial, values in calibration.items():
        assert serial in serial_to_cam_name, f"Unknown serial {serial}"
        cam_name = serial_to_cam_name[serial]
        if cam_name == "wrist_cam":
            cam_xmls.append(wrist_cam)
            continue

        c2w = values["c2w"]
        x, y, z = c2w[:3, 3]
        rot_mat = c2w[:3, :3]
        quat = Rotation.from_matrix(rot_mat).as_quat()
        qx, qy, qz, qw = quat

        cam_xml = cam_template.format(
            cam_name=cam_name,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )
        cam_xmls.append(cam_xml)

    # remove leading new lines as there's too many things going on
    cam_xmls = [xml.lstrip("\n") for xml in cam_xmls]
    tf_xml = base_template.format(cams="\n".join(cam_xmls))

    tf_path = f"{capture_dir}/extrinsics.xml"
    save_paths = [tf_path] + additional_save_paths
    for save_path in save_paths:
        with open(save_path, "w") as f:
            f.write(tf_xml)
        print(f"Saved tf xml to {save_path}")


if __name__ == "__main__":
    convert_to_tf("captures/2023-11-29_12-19-04")
