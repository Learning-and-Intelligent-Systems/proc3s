import argparse
import glob
import json
import os
import random

import apriltag
import cv2
import numpy as np
import open3d as o3d
from capture import capture_devices
from convert_to_robot_frame import convert_to_robot_frame
from convert_to_tf import convert_to_tf

from vtamp.environments.raven.env import setup_raven_environment
from vtamp.environments.robots.panda import PandaRobot

parser = argparse.ArgumentParser()
parser.add_argument(
    "--logdir",
    type=str,
    help="Path to log directory containing captured rgb-d images and extrinsics",
)
parser.add_argument(
    "--save", action="store_true", help="Save extrinsic calib information to logdir"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Visualize each registered pair of pointclouds",
)

parser.add_argument(
    "--ref_cam",
    type=str,
    default="231122071284",
    help="Serial number of reference camera",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")

q_capture = {
    "panda_joint1": -0.27497146181102855,
    "panda_joint2": -0.5074832722571648,
    "panda_joint3": 0.08557974424069388,
    "panda_joint4": -1.7755993051362118,
    "panda_joint5": 0.09217465460300446,
    "panda_joint6": 1.3729298438194049,
    "panda_joint7": 0.5980444280329391,
}


CALIB_OBJECTS_DICT = {
    "plane": {
        "metadata": {
            "0": "ground plane (tabletop)",
        },
        "tags_on_face": {
            "0": [i for i in range(1, 146)],
        },
    },
    "big_box": {
        "metadata": {
            "0": "big face towards kinect 0",
            "1": "top face",
            "2": "side face towards kinect 1",
            "3": "side face towards kinect 2",
            "4": "big face towards kinect 3",
        },
        "tags_on_face": {
            "0": [148, 154, 160, 166, 147, 153, 159, 165, 146, 152, 158, 164],
            "1": [145, 151, 157, 163],
            "2": [170, 169, 168],
            "3": [155, 161, 167],
            "4": [174, 175, 176, 177, 180, 181, 182, 183, 186, 187, 188, 189],
        },
    },
    "small_box": {
        "metadata": {
            "0": "big face towards kinect 0",
            "1": "top face",
            "2": "side face towards kinect 1",
            "3": "side face towards kinect 2",
            "4": "big face towards kinect 3",
        },
        "tags_on_face": {
            "0": [204, 205, 206, 210, 211, 212],
            "1": [209, 208, 207],
            "2": [149, 171],
            "3": [191, 190],
            "4": [184, 178, 172, 185, 179, 173],
        },
    },
}


def detection_to_metric_pose(detector, detection, intrinsics, tagsize):
    """
    detector: apriltag.Detector object
    detection: apriltag detection result objects
    intrinsics: fx, fy, cx, cy (in units of focal length)
    tagsize: size (side length) of the tag (in metres)
    """
    metric_pose = detector.detection_pose(detection, intrinsics)[0]
    metric_pose[:3, 3] = metric_pose[:3, 3] * tagsize
    return metric_pose


def get_valid_points_in_both(
    pts1, depth_img_1, pts2, depth_img_2, depth_scale_factor=1000.0
):
    # pts1: 2D tag corners in first image
    # depth_img_1: first depth image
    # pts2: 2D tag corners in second image
    # depth_img_2: second depth image

    # Find points that are common (have valid, non-zero depths) in both images

    rounded_pts_1 = np.round(pts1).astype(np.uint16)
    depths1 = depth_img_1[rounded_pts_1[:, 1], rounded_pts_1[:, 0]]
    depths1 = depths1.astype(np.float32) / depth_scale_factor
    valid_inds_1 = depths1 > 0

    rounded_pts_2 = np.round(pts2).astype(np.uint16)
    depths2 = depth_img_2[rounded_pts_2[:, 1], rounded_pts_2[:, 0]]
    depths2 = depths2.astype(np.float32) / depth_scale_factor
    valid_inds_2 = depths2 > 0

    return np.logical_and(valid_inds_1, valid_inds_2)


def backproject_to_3d(
    pts, depth_img, intrinsics, valid_inds, depth_scale_factor=1000.0
):
    # pts: np.ndarray of shape N x 2 (pts may be float)
    # depth_img: np.ndarray of shape H x W
    # intrinsics: list of length 4; format: [fx, fy, cx, cy]
    rounded_pts = np.round(pts).astype(np.uint16)
    depths = depth_img[rounded_pts[:, 1], rounded_pts[:, 0]]
    depths = depths.astype(np.float32) / depth_scale_factor
    # valid_inds = depths > 0
    rounded_pts = rounded_pts[valid_inds]
    depths = depths[valid_inds]
    pts = pts[valid_inds].astype(np.float32)
    # ones = np.ones_like(pts[:, 1]).reshape(-1, 1)
    # pts_homo = np.concatenate([pts, ones], axis=-1)
    fx, fy, cx, cy = intrinsics
    _x = pts[:, 0]
    _y = pts[:, 1]
    Z = depths
    X = Z * (_x - cx) / fx
    Y = Z * (_y - cy) / fy
    backproj_pts = np.stack([X, Y, Z], axis=-1)
    return backproj_pts


def orthogonal_procrustes(src, tgt):
    # src: pointcloud (np.ndarray) N, 3
    # tgt: pointcloud (np.ndarray) N, 3
    # !! ASSUMES src and tgt ARE CORRESPONDING POINTS

    # Adapted from pytorch3d
    # https://github.com/facebookresearch/pytorch3d/blob/3388d3f0aa6bc44fe704fca78d11743a0fcac38c/pytorch3d/ops/points_alignment.py#L226

    # Compute mean of each pointset
    src_mu = src.mean(axis=0)
    tgt_mu = tgt.mean(axis=0)

    # Zero-center each pointset
    src_centered = src - src_mu
    tgt_centered = tgt - tgt_mu

    # Compute 3 x 3 covariance matrix
    src_tgt_cov = np.matmul(src_centered.T, tgt_centered)

    # SVD of cov mat gives us components of rotation
    U, D, V = np.linalg.svd(src_tgt_cov)

    # identity mat used for fixing reflections
    E = np.eye(3)
    # reflection test:
    #   checks whether the estimated rotation has det == 1
    #   if not, finds nearest rotation with det == 1 by
    #   flipping the sign of the last column of U
    R_test = np.matmul(U, V.T)
    det = np.linalg.det(R_test)
    if det == -1:
        print("REFLECTION DETECTED! FLIPPING SIGN OF LAST COL OF U!")
        E[-1, -1] = -1

    # rotation component
    R = np.matmul(np.matmul(U, E), V.T)
    # translation component
    t = tgt_mu - np.matmul(src_mu, R)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T


def procrustes(X, Y, scaling=False, reflection=False):
    """
    https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform


def load_intrinsics(intrinsics_file):
    with open(intrinsics_file, "r") as _f:
        intrinsics_dict = json.load(_f)

    return intrinsics_dict


def calibrate_all(args):
    # Seed RNG
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Index of reference camera (w.r.t. which all other cameras are calibrated)

    # args.logdir = "/home/krishna/code/digitize-objects/image-acquisition/logs/calib-dec-28-kinect-00/*"

    glob_str = os.path.join(args.logdir, "*")
    devids = [x for x in sorted(glob.glob(glob_str)) if os.path.isdir(x)]
    logdir_clean = args.logdir.rstrip("/")
    devids = [d.replace(logdir_clean + "/", "") for d in devids]
    num_cameras = len(devids)
    print(f"Found {num_cameras} cameras...")
    print(f"Device IDs: {devids}")

    assert (
        args.ref_cam in devids
    ), f"Reference camera {args.ref_cam} not found in logdir {args.logdir}"
    REF_CAM_IDX = devids.index(args.ref_cam)

    tagsize = 0.03  # 3 cm tags
    num_tags = 216

    detector_options = apriltag.DetectorOptions(
        quad_decimate=1.0,
        refine_decode=True,
        refine_pose=True,
    )

    detector = apriltag.Detector(options=detector_options)
    # detector_dt = dt_apriltags.Detector(
    #     quad_decimate=1.0
    # )

    imgfiles = [
        os.path.join(args.logdir, devids[i], "color", str("0").zfill(5) + ".png")
        for i in range(num_cameras)
    ]
    depthfiles = [
        os.path.join(args.logdir, devids[i], "depth", str("0").zfill(5) + ".png")
        for i in range(num_cameras)
    ]
    depths = [cv2.imread(depthfile, cv2.IMREAD_ANYDEPTH) for depthfile in depthfiles]

    intrinsics = [
        load_intrinsics(os.path.join(args.logdir, devids[i], "intrinsics_color.json"))
        for i in range(num_cameras)
    ]
    imgs = [cv2.imread(imgfile) for imgfile in imgfiles]
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    results = [detector.detect(gray) for gray in grays]
    # rendered_detections = [draw_tag_detections(results[i], imgs[i]) for i in range(num_cameras)]
    # import pdb; pdb.set_trace()

    camera_params = [
        [
            intrinsics[i]["fx"],
            intrinsics[i]["fy"],
            intrinsics[i]["cx"],
            intrinsics[i]["cy"],
        ]
        for i in range(num_cameras)
    ]

    viz_clouds = []  # This list will be set only if args.visualize is True

    # Poll all images for tags of interest
    img_tag_viewgraph = np.zeros((num_cameras, num_tags)).astype(bool)
    img_tag_confidences = np.zeros((num_cameras, num_tags)).astype(np.float32)
    img_tag_detection_map = np.zeros((num_cameras, num_tags)).astype(np.uint32)
    for cam_idx in range(num_cameras):
        for detection_idx, detection in enumerate(results[cam_idx]):
            tag_idx = detection.tag_id
            img_tag_viewgraph[cam_idx, tag_idx] = True
            img_tag_confidences[cam_idx, tag_idx] = detection.decision_margin
            img_tag_detection_map[cam_idx, tag_idx] = detection_idx

    # Compute the transform for each camera w.r.t. the reference camera
    tags_in_img_0 = img_tag_viewgraph[REF_CAM_IDX] == 1.0
    tags_in_img_0_inds = np.nonzero(tags_in_img_0)[
        0
    ]  # 0-th element cause np wraps additional dim
    tag_poses_cam_0 = np.zeros((num_tags, 4, 4))
    print(tags_in_img_0_inds)
    for tag_idx in tags_in_img_0_inds:
        detection_idx = img_tag_detection_map[REF_CAM_IDX, tag_idx]
        tag_poses_cam_0[tag_idx] = detection_to_metric_pose(
            detector,
            results[REF_CAM_IDX][detection_idx],
            camera_params[REF_CAM_IDX],
            tagsize,
        )
        # tag_poses_cam_0[tag_idx][:3, :3] = results_dt[0][detection_idx].pose_R
        # tag_poses_cam_0[tag_idx][:3, 3] = results_dt[0][detection_idx].pose_t[:3, 0] # pose_t is (3, 1)

    color_0 = o3d.geometry.Image(imgs[REF_CAM_IDX])
    depth_0 = o3d.geometry.Image(depths[REF_CAM_IDX])
    rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_0, depth_0, convert_rgb_to_intensity=False
    )
    intrinsic_0 = o3d.camera.PinholeCameraIntrinsic(
        imgs[REF_CAM_IDX].shape[1],
        imgs[REF_CAM_IDX].shape[0],
        camera_params[REF_CAM_IDX][0],
        camera_params[REF_CAM_IDX][1],
        camera_params[REF_CAM_IDX][2],
        camera_params[REF_CAM_IDX][3],
    )

    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd_0, intrinsic=intrinsic_0
    )
    pcd0_colors = np.asarray(pcd0.colors)
    # pcd0_colors[:, 1:3] /= 2  # decrease opacity of G, B channels
    pcd0.colors = o3d.utility.Vector3dVector(pcd0_colors)

    if args.visualize:
        viz_clouds.append(pcd0)

    if args.save:
        # Store pose of ref cam as the 4 x 4 identity matrix
        savefile = os.path.join(logdir_clean, devids[REF_CAM_IDX], "pose.npy")
        np.save(savefile, np.eye(4))

    all_transforms = []

    for cam_idx in range(num_cameras):
        if cam_idx == REF_CAM_IDX:
            continue
        tags_in_img_cur = img_tag_viewgraph[cam_idx] == 1.0
        common_tag_inds = np.nonzero(np.logical_and(tags_in_img_0, tags_in_img_cur))[
            0
        ]  # 0-th element cause np wraps additional dim
        print("Common tag indices shape:", common_tag_inds.shape)
        retained_common_tag_inds = []
        # Retain a maximum of 16 tags on each calibration object
        for calib_obj_key, calib_obj_val in CALIB_OBJECTS_DICT.items():
            tags_on_calib_object = []
            for k, v in calib_obj_val["tags_on_face"].items():
                for _v_tag in v:
                    if _v_tag in common_tag_inds:
                        tags_on_calib_object.append(_v_tag)
            _sampled_tag_ids = random.sample(
                tags_on_calib_object, min(16, len(tags_on_calib_object))
            )
            retained_common_tag_inds += _sampled_tag_ids

        common_tag_inds = retained_common_tag_inds
        # print(common_tag_inds)

        cam_cur_to_cam_0_tf_measurements_rmat = []
        measurement_confidences = []
        _pts_0 = []
        _pts_cur = []
        for common_tag_idx in common_tag_inds:
            detection_idx_cur = img_tag_detection_map[cam_idx, common_tag_idx]
            tag_to_cam_cur_tf = detection_to_metric_pose(
                detector,
                results[cam_idx][detection_idx_cur],
                camera_params[cam_idx],
                tagsize,
            )
            detection_idx_0 = img_tag_detection_map[REF_CAM_IDX][common_tag_idx]
            _detection_0 = results[REF_CAM_IDX][detection_idx_0]
            _detection_cur = results[cam_idx][detection_idx_cur]
            # # (not working) swap x, y coordinates of detection.corners (apriltag detections have x axis along cols)
            # _pts_0.append(np.stack([_detection_0.corners[:, 1], _detection_0.corners[:, 0]]).T)
            # _pts_cur.append(np.stack([_detection_cur.corners[:, 1], _detection_0.corners[:, 0]]).T)
            _pts_0.append(np.asarray(_detection_0.center))
            _pts_cur.append(np.asarray(_detection_cur.center))
            cam_cur_to_cam_0_tf = np.matmul(
                tag_poses_cam_0[common_tag_idx], np.linalg.inv(tag_to_cam_cur_tf)
            )
            cam_cur_to_cam_0_tf_measurements_rmat.append(cam_cur_to_cam_0_tf)
            measurement_confidences.append(img_tag_confidences[cam_idx, detection_idx])

        _pts_0 = np.stack(_pts_0, axis=0)
        _pts_cur = np.stack(_pts_cur, axis=0)

        valid_inds = get_valid_points_in_both(
            _pts_0, depths[REF_CAM_IDX], _pts_cur, depths[cam_idx]
        )

        _corner_pcd_0 = backproject_to_3d(
            _pts_0, depths[REF_CAM_IDX], camera_params[REF_CAM_IDX], valid_inds
        )
        _corner_pcd_cur = backproject_to_3d(
            _pts_cur, depths[cam_idx], camera_params[cam_idx], valid_inds
        )

        # _corner_pcd_0 = np.random.rand(100, 3)
        # tf = np.eye(4)
        # c = np.cos(np.pi / 4)
        # s = np.sin(np.pi / 4)
        # tf[1, 1] = c
        # tf[1, 2] = -s
        # tf[2, 1] = s
        # tf[2, 2] = c
        # _corner_pcd_cur = np.matmul(_corner_pcd_0, tf[:3, :3])
        # print(_corner_pcd_cur.shape, _corner_pcd_0.shape)

        # _transform = orthogonal_procrustes(_corner_pcd_cur, _corner_pcd_0)
        # _transform[:3, 3] = 0

        print(devids[cam_idx], _corner_pcd_0.shape)
        rand_indices = np.random.choice(_corner_pcd_0.shape[0], size=10, replace=False)
        # _corner_pcd_0

        src_cloud = _corner_pcd_cur.copy()

        print(_corner_pcd_cur.shape, _corner_pcd_0.shape)
        # this uses the reverse convention
        _, _, tform = procrustes(_corner_pcd_cur, _corner_pcd_0, scaling=True)
        _transform = np.eye(4)
        _transform[:3, :3] = tform["rotation"].T
        _src_rotated = np.matmul(_corner_pcd_cur, tform["rotation"].T)
        _translation = _corner_pcd_0.mean(0) - _src_rotated.mean(0)
        _transform[:3, 3] = _translation
        _corner_pcd_cur = _src_rotated + _translation

        color_cur = o3d.geometry.Image(imgs[cam_idx])
        depth_cur = o3d.geometry.Image(depths[cam_idx])
        rgbd_cur = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_cur, depth_cur, convert_rgb_to_intensity=False
        )
        intrinsic_cur = o3d.camera.PinholeCameraIntrinsic(
            imgs[cam_idx].shape[1],
            imgs[cam_idx].shape[0],
            camera_params[cam_idx][0],
            camera_params[cam_idx][1],
            camera_params[cam_idx][2],
            camera_params[cam_idx][3],
        )
        pcd_cur = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_cur, intrinsic_cur
        )

        pcd0_corners = o3d.geometry.PointCloud()
        pcd0_corners.points = o3d.utility.Vector3dVector(_corner_pcd_0)
        pcd0_corners.paint_uniform_color([1, 0, 0])

        pcd_cur_corners = o3d.geometry.PointCloud()
        pcd_cur_corners.points = o3d.utility.Vector3dVector(_corner_pcd_cur)
        pcd_cur_corners.paint_uniform_color([0, 0, 1])

        # This seems to work well
        _rot = _transform[:3, :3]
        _translation = _transform[:3, 3]
        _transform[:3, :3] = _rot.T
        _transform[:3, 3] = _translation
        pcd_cur.transform(_transform)

        # pcd_cur_corners.transform(_transform)

        # o3d.visualization.draw_geometries([pcd0_corners, pcd_cur_corners])

        # pcd_cur_points = np.asarray(pcd_cur.points)
        # pcd_cur_points = (
        #     np.matmul(pcd_cur_points, _transform[:3, :3]) + _transform[:3, 3]
        # )
        # pcd_cur.points = o3d.utility.Vector3dVector(pcd_cur_points)
        # pcd0_colors[:, :2] /= 2  # decrease opacity of R, G channels
        pcd0.colors = o3d.utility.Vector3dVector(pcd0_colors)

        if args.visualize:
            # o3d.visualization.draw_geometries(
            #     [pcd0, pcd_cur], window_name=f"{devids[cam_idx]}"
            # )
            # o3d.visualization.draw_geometries([pcd0, pcd_cur], window_name=f"{devids[cam_idx]}")
            viz_clouds.append(pcd_cur)

        if args.save:
            savefile = os.path.join(logdir_clean, devids[cam_idx], "pose.npy")
            np.save(savefile, _transform)
            print(f"Saved transform to {savefile}")

        all_transforms.append(_transform)

    coords = []
    for transform in all_transforms:
        # transform is a 4x4 np
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        mesh.transform(transform)
        coords.append(mesh)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    coords.append(mesh)

    if args.visualize:
        # visualize all clouds
        merged = viz_clouds + coords
        for thing in merged:
            thing.transform(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
        o3d.visualization.draw_geometries(merged)


if __name__ == "__main__":
    args = parser.parse_args()

    client, robot = setup_raven_environment(robot_type="panda", real_robot=True)
    robot.servoj(list(q_capture.values()), teleport=True)

    args.logdir = capture_dir = capture_devices()
    args.save = True
    args.visualize = True

    calibrate_all(args)

    convert_to_robot_frame(robot, capture_dir, client)
    convert_to_tf(capture_dir)
