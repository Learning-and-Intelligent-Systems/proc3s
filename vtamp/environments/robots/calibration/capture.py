import json
import os
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from device_manager import DeviceManager
from params_proto import ParamsProto, Proto


class CaptureArgs(ParamsProto):
    base_log_dir: str = Proto("captures", help="Base directory for capture files")
    capture_pattern: str = Proto(
        "%Y-%m-%d_%H-%M-%S",
        help="strftime pattern for capture directory inside base_log_dir",
    )


align_to = rs.stream.color
align = rs.align(align_to)


def start_rgbd_stream(device):
    pass


def capture_devices():
    # Enumerate devices available
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} RealSense devices")

    capture_now = datetime.now().strftime(CaptureArgs.capture_pattern)
    capture_dir = os.path.join(CaptureArgs.base_log_dir, capture_now)
    os.makedirs(capture_dir, exist_ok=True)
    print(f"Capturing to {capture_dir}")

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Use device manager
    device_manager = DeviceManager(ctx, config)
    device_manager.enable_all_devices()

    # Allow some frames for the auto-exposure controller to stablise
    dispose_frames_for_stablisation = 15  # frames
    for frame in range(dispose_frames_for_stablisation):
        frames = device_manager.poll_frames()

    for (serial, model), device_frames in frames.items():
        device_dir = os.path.join(capture_dir, serial)
        depth_dir = os.path.join(device_dir, "depth")
        color_dir = os.path.join(device_dir, "color")
        for d in [device_dir, depth_dir, color_dir]:
            os.makedirs(d, exist_ok=True)

        depth_frame = device_frames[rs.stream.depth]
        depth_path = os.path.join(depth_dir, "00000.png")
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imwrite(depth_path, depth_image)

        color_frame = device_frames[rs.stream.color]
        color_path = os.path.join(color_dir, "00000.png")
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(color_path, color_image)

        # Get intrinsics
        intrinsics_color = color_frame.profile.as_video_stream_profile().intrinsics
        intrisics_dict = {
            "fx": intrinsics_color.fx,
            "fy": intrinsics_color.fy,
            "cx": intrinsics_color.ppx,
            "cy": intrinsics_color.ppy,
        }
        with open(os.path.join(device_dir, "intrinsics_color.json"), "w") as f:
            json.dump(intrisics_dict, f, indent=4)
        np.savez(
            os.path.join(device_dir, "intrinsics_color.npz"),
            **intrisics_dict,
        )

    print(f"Done! Check {capture_dir}")

    return capture_dir
