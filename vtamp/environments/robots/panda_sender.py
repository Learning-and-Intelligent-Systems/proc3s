import pickle
import zlib

import zmq
import zmq.ssh


class PandaSender:
    def __init__(self):
        # Local comms
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5554")

    def publish_pose_dict(self, pose_tuple):
        print("[Controller] Publishing Object Poses")
        self.socket.send(
            zlib.compress(
                pickle.dumps(
                    {"message_name": "publish_pose_dict", "pose_dict": pose_tuple}
                )
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def publish_weighted_pointcloud(self, points, values):
        print("[Controller] Publishing Pointcloud")
        self.socket.send(
            zlib.compress(
                pickle.dumps(
                    {
                        "message_name": "publish_weighted_pointcloud",
                        "points": points,
                        "values": values,
                    }
                )
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def publish_attachment(self, object_id, relative_pose, attach=True):
        print("[Controller] Publishing Attachment/Detachment")
        self.socket.send(
            zlib.compress(
                pickle.dumps(
                    {
                        "message_name": "publish_attachment",
                        "object_id": object_id,
                        "relative_pose": relative_pose,
                        "attach": attach,
                    }
                )
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def capture_realsense(self):
        print("[Controller] Capturing realsense")
        self.socket.send(
            zlib.compress(pickle.dumps({"message_name": "capture_realsense"}))
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["rgb"], message["depth"], message["intrinsics"]

    def command_arm(self, positions):
        print("[Controller] Commanding arm to target config")
        self.socket.send(
            zlib.compress(
                pickle.dumps({"message_name": "command_arm", "positions": positions})
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def get_joint_states(self):
        print("[Controller] Getting joint states")
        self.socket.send(
            zlib.compress(pickle.dumps({"message_name": "get_joint_states"}))
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["joint_states"]

    def open_gripper(self):
        print("[Controller] Opening gripper")
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "open_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def close_gripper(self):
        print("[Controller] Closing gripper")
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "close_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def execute_position_path(self, pdicts):
        print("[Controller] Executing position path")
        self.socket.send(
            zlib.compress(
                pickle.dumps(
                    {"message_name": "execute_position_path", "pdicts": pdicts}
                )
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message


    def move_to_joint_positions(self, pdict):
        print("[Controller] move_to_joint_positions")
        print(pdict)
        self.socket.send(
            zlib.compress(
                pickle.dumps(
                    {"message_name": "move_to_joint_positions", "pdict": pdict}
                )
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

