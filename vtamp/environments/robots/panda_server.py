import pickle
import struct
import time
import zlib

import numpy as np
import rospy
import zmq
import zmq.ssh
from cv_bridge import CvBridge, CvBridgeError
from franka_interface import ArmInterface, GripperInterface
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo as msg_CameraInfo
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

# Gripper testing
# import rospy
# from franka_interface import GripperInterface
# rospy.init_node("panda_data_collection_node")
# gripper_interface = GripperInterface()
# gripper_interface.open()
# gripper_interface.close()

rospy.init_node("panda_data_collection_node2")
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5554")

SAMPLE_FREQ = 0.1


class ImageListener:
    def __init__(self, cam_name):
        self.cam_name = cam_name
        self.depth_topic = "/" + str(cam_name) + "/aligned_depth_to_color/image_raw"
        self.rgb_topic = "/" + str(cam_name) + "/color/image_raw"

        self.ex_sub = rospy.Subscriber("/tf_static", TFMessage, self.extrinsicsCallback)
        self.extrinsics = None
        self.parent_frame = None

        self.camera_info = rospy.wait_for_message(
            "/" + str(cam_name) + "/aligned_depth_to_color/camera_info", msg_CameraInfo
        )

        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, msg_Image, self.imageDepthCallback
        )
        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic, msg_Image, self.imageRGBCallback
        )

        # self.visibility_sub = rospy.Subscriber(self.pose_topic, msg_Visibility, self.occupancyCallback)
        self.last_rgb_time = -np.inf
        self.last_depth_time = -np.inf

        self.current_depth = None
        self.current_rgb = None

    def extrinsicsCallback(self, data):
        for transform in data.transforms:
            if (
                self.cam_name not in transform.header.frame_id
                and transform.child_frame_id.startswith(self.cam_name)
            ):
                t = transform.transform.translation
                r = transform.transform.rotation
                self.parent_frame = transform.header.frame_id
                self.extrinsics = ([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])

    def imageDepthCallback(self, data):
        if time.time() - self.last_depth_time > SAMPLE_FREQ:
            if self.extrinsics is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
                    self.last_depth_time = time.time()
                    self.current_depth = cv_image

                except CvBridgeError as e:
                    print(e)
                    return

    def imageRGBCallback(self, data):
        if time.time() - self.last_rgb_time > SAMPLE_FREQ:
            if self.extrinsics is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
                    self.last_rgb_time = time.time()
                    self.current_rgb = cv_image

                except CvBridgeError as e:
                    print(e)
                    return


# image_listener = ImageListener("front_left_cam")
image_listener = ImageListener("wrist_cam")

arm_interface = ArmInterface()
gripper_interface = GripperInterface()


def pose_to_ros(pose):
    pose_message = Pose()
    pose_message.position.x = pose[0][0]
    pose_message.position.y = pose[0][1]
    pose_message.position.z = pose[0][2]
    pose_message.orientation.x = pose[1][0]
    pose_message.orientation.y = pose[1][1]
    pose_message.orientation.z = pose[1][2]
    pose_message.orientation.w = pose[1][3]
    return pose_message


def capture_realsense(message):
    intrinsics = np.reshape(image_listener.camera_info.K, (3, 3))
    message = {
        "rgb": image_listener.current_rgb,
        "depth": image_listener.current_depth,
        "intrinsics": intrinsics,
    }
    socket.send(zlib.compress(pickle.dumps(message)))


def command_arm(message):
    arm_interface.move_to_joint_positions(message["positions"])
    message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(message)))


def get_joint_states(message):
    joints = arm_interface.joint_angles()
    message = {"joint_states": joints}
    socket.send(zlib.compress(pickle.dumps(message)))


def open_gripper(message):
    gripper_interface.open()
    message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(message)))


def close_gripper(message):
    gripper_interface.close()
    message = {"success": True}
    socket.send(zlib.compress(pickle.dumps(message)))


def execute_position_path(message):
    arm_interface.execute_position_path(message["pdicts"])
    socket.send(zlib.compress(pickle.dumps({"success": True})))
    
    
def move_to_joint_positions(message):
    arm_interface.move_to_joint_positions(message["pdict"])
    socket.send(zlib.compress(pickle.dumps({"success": True})))


while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle.loads(zlib.decompress(socket.recv()))

    print("Received request: {}".format(message))

    #  Send reply back to client
    globals()[message["message_name"]](message)
