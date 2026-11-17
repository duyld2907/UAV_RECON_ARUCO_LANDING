#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ArucoDebugSaver(Node):
    def __init__(self):
        super().__init__('aruco_debug_saver')

        self.declare_parameter('image_topic', '/aruco/debug_image')
        self.declare_parameter('save_dir', '/ws/src/drone_control/aruco_debug')
        self.declare_parameter('save_every_n', 3)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.save_dir = Path(str(self.get_parameter('save_dir').value))
        self.save_every_n = int(self.get_parameter('save_every_n').value)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0

        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.get_logger().info(f'Saving debug images from: {self.image_topic} -> {self.save_dir}')

    def msg_to_bgr(self, msg: Image):
        if msg.encoding != 'bgr8':
            raise RuntimeError(f'Only bgr8 is supported, current: {msg.encoding}')
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        return frame.copy()

    def image_cb(self, msg: Image):
        try:
            frame = self.msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f'Image decode failed: {e}')
            return

        self.frame_count += 1
        if self.frame_count % max(1, self.save_every_n) != 0:
            return

        latest_path = self.save_dir / 'latest.jpg'
        cv2.imwrite(str(latest_path), frame)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDebugSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
