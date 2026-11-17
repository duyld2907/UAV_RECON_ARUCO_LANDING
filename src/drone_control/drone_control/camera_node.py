#!/usr/bin/env python3
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage


class BottomCameraNode(Node):
    """
    Camera node đọc camera 1 lần rồi publish 2 luồng:

    1) /camera/bottom/image_raw
       - Ảnh raw BGR8 đầy đủ.
       - Dành cho aruco_node / landing detect.
       - Không nên xem topic này qua mạng yếu nếu không cần.

    2) /camera/bottom/raw_preview/compressed
       - Ảnh camera thô nhưng đã nén JPEG, giới hạn FPS.
       - Dành để xem live drone đang thấy gì.
       - Không ghi video, không ghi JPG, không vẽ detect.
    """

    def __init__(self):
        super().__init__('bottom_camera_node')

        # ================= Camera raw output =================
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('topic_name', '/camera/bottom/image_raw')
        self.declare_parameter('frame_id', 'bottom_camera')
        self.declare_parameter('flip_code', -2)  # -2 = no flip, 0/1/-1 = cv2.flip code
        self.declare_parameter('fourcc', 'MJPG')

        # ================= Lightweight live preview =================
        # enable_preview=True: publish thêm ảnh JPEG compressed để xem live.
        # Không ghi file video/JPG ra ổ đĩa.
        self.declare_parameter('enable_preview', False)
        self.declare_parameter('preview_topic_name', '/camera/bottom/raw_preview/compressed')
        self.declare_parameter('preview_fps', 5.0)
        self.declare_parameter('jpeg_quality', 65)

        self.camera_index = int(self.get_parameter('camera_index').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.topic_name = str(self.get_parameter('topic_name').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.flip_code = int(self.get_parameter('flip_code').value)
        self.fourcc = str(self.get_parameter('fourcc').value)

        self.enable_preview = bool(self.get_parameter('enable_preview').value)
        self.preview_topic_name = str(self.get_parameter('preview_topic_name').value)
        self.preview_fps = float(self.get_parameter('preview_fps').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.last_preview_time = 0.0

        self.pub = self.create_publisher(Image, self.topic_name, 10)

        self.preview_pub = None
        if self.enable_preview:
            self.preview_pub = self.create_publisher(
                CompressedImage,
                self.preview_topic_name,
                3,
            )
            self.get_logger().info(
                f'Raw camera preview enabled: topic={self.preview_topic_name}, '
                f'fps={self.preview_fps:.1f}, jpeg_quality={self.jpeg_quality}'
            )

        self.cap = None
        self.open_camera()

        period = max(1.0 / max(self.fps, 1.0), 0.01)
        self.timer = self.create_timer(period, self.publish_frame)

    def open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.get_logger().info(f'Opening camera index={self.camera_index}')
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {self.camera_index}')

        if self.fourcc and len(self.fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(
            f'Camera ready requested={self.width}x{self.height}@{self.fps:.1f}, '
            f'actual={actual_w:.0f}x{actual_h:.0f}@{actual_fps:.1f}, fourcc={self.fourcc}'
        )

    def frame_to_msg(self, frame):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * frame.shape[2]
        msg.data = frame.tobytes()
        return msg

    def frame_to_compressed_msg(self, frame):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'

        quality = max(10, min(95, int(self.jpeg_quality)))
        ok, encoded = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return None

        msg.data = encoded.tobytes()
        return msg

    def maybe_publish_preview(self, frame):
        if not self.enable_preview or self.preview_pub is None:
            return

        now = time.time()
        period = 1.0 / max(self.preview_fps, 0.5)

        if now - self.last_preview_time < period:
            return

        self.last_preview_time = now
        preview_msg = self.frame_to_compressed_msg(frame)
        if preview_msg is not None:
            self.preview_pub.publish(preview_msg)

    def publish_frame(self):
        if self.cap is None or not self.cap.isOpened():
            try:
                self.open_camera()
            except Exception as e:
                self.get_logger().error(f'Camera reopen failed: {e}')
                return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Camera read failed, retrying...')
            time.sleep(0.05)
            return

        if self.flip_code in (-1, 0, 1):
            frame = cv2.flip(frame, self.flip_code)

        # Publish raw full image for ArUco/landing.
        self.pub.publish(self.frame_to_msg(frame))

        # Publish lightweight live preview for user viewing.
        # This does not affect ArUco detection and does not write video/JPG files.
        self.maybe_publish_preview(frame)

    def destroy_node(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BottomCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
