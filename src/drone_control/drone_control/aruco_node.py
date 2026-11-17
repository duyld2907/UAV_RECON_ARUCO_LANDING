#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_node.py

Bản hoàn chỉnh đã chỉnh cho D435/camera đang lắp ngược:
- Detect ArUco/PnP/JSON/Kalman vẫn dùng ảnh raw như trước để không ảnh hưởng landing.
- Chỉ xoay debug video/debug image/latest_debug_jpg để quan sát cho dễ.
- Nếu debug_rotate_180=True, chữ overlay được vẽ lại sau khi xoay nên không bị ngược.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32, String


class PixelErrKalmanCV:
    """
    Kalman 2D constant-velocity cho sai số ảnh:
        state = [err_x, err_y, err_x_rate, err_y_rate]
    """

    def __init__(
        self,
        accel_noise_px_s2: float = 350.0,
        meas_noise_px: float = 7.0,
        p0_pos_px: float = 80.0,
        p0_vel_px_s: float = 500.0,
        max_dt_s: float = 0.20,
        reset_timeout_s: float = 0.80,
        gate_nis: float = 25.0,
        max_rate_px_s: float = 1200.0,
    ) -> None:
        self.accel_noise_px_s2 = float(accel_noise_px_s2)
        self.meas_noise_px = float(meas_noise_px)
        self.p0_pos_px = float(p0_pos_px)
        self.p0_vel_px_s = float(p0_vel_px_s)
        self.max_dt_s = float(max_dt_s)
        self.reset_timeout_s = float(reset_timeout_s)
        self.gate_nis = float(gate_nis)
        self.max_rate_px_s = float(max_rate_px_s)

        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float)
        self.initialized = False
        self.t: Optional[float] = None
        self.last_update_t: Optional[float] = None
        self.samples = 0
        self.last_nis = 0.0
        self.last_rejected = False

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P = np.eye(4, dtype=float)
        self.initialized = False
        self.t = None
        self.last_update_t = None
        self.samples = 0
        self.last_nis = 0.0
        self.last_rejected = False

    def _init_from_measurement(self, err_x: float, err_y: float, t: float) -> None:
        self.x[:] = 0.0
        self.x[0, 0] = float(err_x)
        self.x[1, 0] = float(err_y)

        self.P = np.diag([
            self.p0_pos_px ** 2,
            self.p0_pos_px ** 2,
            self.p0_vel_px_s ** 2,
            self.p0_vel_px_s ** 2,
        ]).astype(float)

        self.initialized = True
        self.t = float(t)
        self.last_update_t = float(t)
        self.samples = 1
        self.last_nis = 0.0
        self.last_rejected = False

    def predict_to(self, t: float) -> None:
        t = float(t)

        if not self.initialized:
            self.t = t
            return

        if self.t is None:
            self.t = t
            return

        dt = t - float(self.t)
        if dt <= 1e-4:
            return

        dt = min(dt, self.max_dt_s)

        F = np.eye(4, dtype=float)
        F[0, 2] = dt
        F[1, 3] = dt

        sa2 = self.accel_noise_px_s2 ** 2
        q11 = 0.25 * dt ** 4 * sa2
        q13 = 0.5 * dt ** 3 * sa2
        q33 = dt ** 2 * sa2

        Q = np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33],
        ], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        n = math.hypot(vx, vy)
        if n > self.max_rate_px_s and n > 1e-6:
            scale = self.max_rate_px_s / n
            self.x[2, 0] = vx * scale
            self.x[3, 0] = vy * scale

        self.t = t

    def update(self, err_x: float, err_y: float, t: float, meas_noise_px: Optional[float] = None) -> bool:
        t = float(t)
        err_x = float(err_x)
        err_y = float(err_y)

        if not math.isfinite(err_x) or not math.isfinite(err_y):
            return False

        if self.initialized and self.last_update_t is not None:
            if t - float(self.last_update_t) > self.reset_timeout_s:
                self.reset()

        if not self.initialized:
            self._init_from_measurement(err_x, err_y, t)
            return True

        self.predict_to(t)

        H = np.zeros((2, 4), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        r = float(meas_noise_px if meas_noise_px is not None else self.meas_noise_px)
        r = max(1.0, r)
        R = np.eye(2, dtype=float) * (r ** 2)

        z = np.array([[err_x], [err_y]], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.reset()
            self._init_from_measurement(err_x, err_y, t)
            return True

        nis = float((y.T @ S_inv @ y)[0, 0])
        self.last_nis = nis

        if nis > self.gate_nis and self.samples >= 3:
            self.last_rejected = True
            return False

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y

        I = np.eye(4, dtype=float)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self.last_update_t = t
        self.samples += 1
        self.last_rejected = False
        return True

    def age(self, now: float) -> float:
        if self.last_update_t is None:
            return 1e9
        return max(0.0, float(now) - float(self.last_update_t))

    def state(self):
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0]),
        )

    def pos_std_px(self) -> float:
        px = max(0.0, float(self.P[0, 0]))
        py = max(0.0, float(self.P[1, 1]))
        return math.sqrt(max(px, py))


class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.declare_parameter("image_topic", "/camera/bottom/image_raw")
        self.declare_parameter("debug_image_topic", "/aruco/debug_image")
        self.declare_parameter("found_topic", "/aruco/found")
        self.declare_parameter("center_topic", "/aruco/center_px")
        self.declare_parameter("error_topic", "/aruco/error_px")
        self.declare_parameter("tvec_topic", "/aruco/tvec")
        self.declare_parameter("ids_topic", "/aruco/detected_ids")
        self.declare_parameter("selected_id_topic", "/aruco/selected_id")
        self.declare_parameter("selected_mode_topic", "/aruco/selected_mode")
        self.declare_parameter("desired_mode_topic", "/aruco/desired_mode")
        self.declare_parameter("detections_json_topic", "/vision/aruco_detections_json")

        self.declare_parameter("landing_cmd_topic", "/landing_cmd")
        self.declare_parameter("sync_desired_mode_from_landing_cmd", True)

        self.declare_parameter("large_id", 150)
        self.declare_parameter("small_id", 40)
        self.declare_parameter("dict_name", "DICT_4X4_250")
        self.declare_parameter("marker_size_large_m", 0.70)
        self.declare_parameter("marker_size_small_m", 0.08)
        self.declare_parameter("calib_path", "/ws/src/drone_control/config/camera_calib.npz")
        self.declare_parameter("use_calib", True)
        self.declare_parameter("publish_debug_image", True)

        self.declare_parameter("use_clahe", True)
        self.declare_parameter("clahe_clip_limit", 2.0)
        self.declare_parameter("clahe_grid_size", 8)
        self.declare_parameter("use_small_blur", True)
        self.declare_parameter("blur_kernel", 3)
        self.declare_parameter("detect_on_raw_gray", True)
        self.declare_parameter("detect_on_processed", True)
        self.declare_parameter("duplicate_center_dist_px", 12.0)
        self.declare_parameter("json_filter_alpha", 0.30)
        self.declare_parameter("json_err_source", "filtered")
        self.declare_parameter("marker_state_timeout_s", 0.35)
        self.declare_parameter("ema_alpha", 0.25)
        self.declare_parameter("min_avg_side_px_large", 14.0)
        self.declare_parameter("min_avg_side_px_small", 6.0)

        self.declare_parameter("use_kalman_err", True)
        self.declare_parameter("kf_accel_noise_px_s2", 350.0)
        self.declare_parameter("kf_meas_noise_px_large", 7.0)
        self.declare_parameter("kf_meas_noise_px_small", 11.0)
        self.declare_parameter("kf_p0_pos_px", 80.0)
        self.declare_parameter("kf_p0_vel_px_s", 500.0)
        self.declare_parameter("kf_max_dt_s", 0.20)
        self.declare_parameter("kf_reset_timeout_s", 0.80)
        self.declare_parameter("kf_gate_nis", 25.0)
        self.declare_parameter("kf_max_rate_px_s", 1200.0)

        self.declare_parameter("publish_predict_only_json", False)
        self.declare_parameter("kf_predict_timeout_s", 0.16)

        self.declare_parameter("small_priority_in_large", False)
        self.declare_parameter("small_priority_min_side_px", 28.0)
        self.declare_parameter("small_min_side_for_both_px", 22.0)
        self.declare_parameter("allow_small_override_in_large_mode", False)
        self.declare_parameter("small_override_min_side_px", 24.0)

        self.declare_parameter("log_root_dir", "/ws/src/drone_control/aruco_logs")
        self.declare_parameter("record_debug_video", False)
        self.declare_parameter("record_raw_video", False)
        self.declare_parameter("video_fps", 10.0)
        self.declare_parameter("video_fourcc", "XVID")
        self.declare_parameter("draw_landing_boxes", True)
        self.declare_parameter("large_box_half_x_px", 220.0)
        self.declare_parameter("large_box_half_y_px", 165.0)
        self.declare_parameter("small_box_half_x_px", 140.0)
        self.declare_parameter("small_box_half_y_px", 110.0)
        self.declare_parameter("record_video_stride", 3)
        self.declare_parameter("write_latest_debug_jpg", False)
        self.declare_parameter("latest_debug_jpg_period_s", 1.0)
        self.declare_parameter("csv_flush_every_n", 10)

        # Mới: chỉ xoay phần debug để xem cho đúng chiều.
        # Không xoay dữ liệu detect/JSON/PID/Kalman/tvec.
        self.declare_parameter("debug_rotate_180", False)

        self.use_kalman_err = bool(self.get_parameter("use_kalman_err").value)
        self.kf_accel_noise_px_s2 = float(self.get_parameter("kf_accel_noise_px_s2").value)
        self.kf_meas_noise_px_large = float(self.get_parameter("kf_meas_noise_px_large").value)
        self.kf_meas_noise_px_small = float(self.get_parameter("kf_meas_noise_px_small").value)
        self.kf_p0_pos_px = float(self.get_parameter("kf_p0_pos_px").value)
        self.kf_p0_vel_px_s = float(self.get_parameter("kf_p0_vel_px_s").value)
        self.kf_max_dt_s = float(self.get_parameter("kf_max_dt_s").value)
        self.kf_reset_timeout_s = float(self.get_parameter("kf_reset_timeout_s").value)
        self.kf_gate_nis = float(self.get_parameter("kf_gate_nis").value)
        self.kf_max_rate_px_s = float(self.get_parameter("kf_max_rate_px_s").value)
        self.publish_predict_only_json = bool(self.get_parameter("publish_predict_only_json").value)
        self.kf_predict_timeout_s = float(self.get_parameter("kf_predict_timeout_s").value)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.found_topic = str(self.get_parameter("found_topic").value)
        self.center_topic = str(self.get_parameter("center_topic").value)
        self.error_topic = str(self.get_parameter("error_topic").value)
        self.tvec_topic = str(self.get_parameter("tvec_topic").value)
        self.ids_topic = str(self.get_parameter("ids_topic").value)
        self.selected_id_topic = str(self.get_parameter("selected_id_topic").value)
        self.selected_mode_topic = str(self.get_parameter("selected_mode_topic").value)
        self.desired_mode_topic = str(self.get_parameter("desired_mode_topic").value)
        self.detections_json_topic = str(self.get_parameter("detections_json_topic").value)
        self.landing_cmd_topic = str(self.get_parameter("landing_cmd_topic").value)
        self.sync_desired_mode_from_landing_cmd = bool(self.get_parameter("sync_desired_mode_from_landing_cmd").value)

        self.large_id = int(self.get_parameter("large_id").value)
        self.small_id = int(self.get_parameter("small_id").value)
        self.allowed_ids = {self.large_id, self.small_id}
        self.desired_mode = "LARGE"

        self.dict_name = str(self.get_parameter("dict_name").value)
        self.marker_size_large_m = float(self.get_parameter("marker_size_large_m").value)
        self.marker_size_small_m = float(self.get_parameter("marker_size_small_m").value)
        self.calib_path = str(self.get_parameter("calib_path").value)
        self.use_calib = bool(self.get_parameter("use_calib").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)

        self.use_clahe = bool(self.get_parameter("use_clahe").value)
        self.clahe_clip_limit = float(self.get_parameter("clahe_clip_limit").value)
        self.clahe_grid_size = int(self.get_parameter("clahe_grid_size").value)
        self.use_small_blur = bool(self.get_parameter("use_small_blur").value)
        self.blur_kernel = int(self.get_parameter("blur_kernel").value)
        self.detect_on_raw_gray = bool(self.get_parameter("detect_on_raw_gray").value)
        self.detect_on_processed = bool(self.get_parameter("detect_on_processed").value)
        self.duplicate_center_dist_px = float(self.get_parameter("duplicate_center_dist_px").value)
        self.json_filter_alpha = float(self.get_parameter("json_filter_alpha").value)
        self.json_err_source = str(self.get_parameter("json_err_source").value).strip().lower()
        if self.json_err_source not in {"raw", "filtered"}:
            self.get_logger().warn(f"json_err_source={self.json_err_source} invalid -> raw")
            self.json_err_source = "raw"
        self.marker_state_timeout_s = max(0.05, float(self.get_parameter("marker_state_timeout_s").value))
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.min_avg_side_px_large = float(self.get_parameter("min_avg_side_px_large").value)
        self.min_avg_side_px_small = float(self.get_parameter("min_avg_side_px_small").value)
        self.small_priority_in_large = bool(self.get_parameter("small_priority_in_large").value)
        self.small_priority_min_side_px = float(self.get_parameter("small_priority_min_side_px").value)
        self.small_min_side_for_both_px = float(self.get_parameter("small_min_side_for_both_px").value)
        self.allow_small_override_in_large_mode = bool(self.get_parameter("allow_small_override_in_large_mode").value)
        self.small_override_min_side_px = float(self.get_parameter("small_override_min_side_px").value)

        self.log_root_dir = str(self.get_parameter("log_root_dir").value)
        self.record_debug_video = bool(self.get_parameter("record_debug_video").value)
        self.record_raw_video = bool(self.get_parameter("record_raw_video").value)
        self.video_fps = float(self.get_parameter("video_fps").value)
        self.video_fourcc = str(self.get_parameter("video_fourcc").value)
        self.draw_landing_boxes = bool(self.get_parameter("draw_landing_boxes").value)
        self.large_box_half_x_px = float(self.get_parameter("large_box_half_x_px").value)
        self.large_box_half_y_px = float(self.get_parameter("large_box_half_y_px").value)
        self.small_box_half_x_px = float(self.get_parameter("small_box_half_x_px").value)
        self.small_box_half_y_px = float(self.get_parameter("small_box_half_y_px").value)
        self.record_video_stride = max(1, int(self.get_parameter("record_video_stride").value))
        self.write_latest_debug_jpg = bool(self.get_parameter("write_latest_debug_jpg").value)
        self.latest_debug_jpg_period_s = max(0.05, float(self.get_parameter("latest_debug_jpg_period_s").value))
        self.csv_flush_every_n = max(1, int(self.get_parameter("csv_flush_every_n").value))
        self.debug_rotate_180 = bool(self.get_parameter("debug_rotate_180").value)
        self._last_latest_debug_jpg_sec = 0.0

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV thiếu module aruco. Cần opencv-contrib-python")
        if not hasattr(cv2.aruco, self.dict_name):
            raise RuntimeError(f"Dictionary không tồn tại: {self.dict_name}")

        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.015
        params.maxMarkerPerimeterRate = 4.0
        params.minDistanceToBorder = 3
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        self.camera_matrix = None
        self.dist_coeffs = None
        if self.use_calib:
            self.load_calib(self.calib_path)

        self.found_pub = self.create_publisher(Bool, self.found_topic, 10)
        self.center_pub = self.create_publisher(PointStamped, self.center_topic, 10)
        self.error_pub = self.create_publisher(PointStamped, self.error_topic, 10)
        self.tvec_pub = self.create_publisher(PointStamped, self.tvec_topic, 10)
        self.ids_pub = self.create_publisher(String, self.ids_topic, 10)
        self.selected_id_pub = self.create_publisher(Int32, self.selected_id_topic, 10)
        self.selected_mode_pub = self.create_publisher(String, self.selected_mode_topic, 10)
        self.detections_json_pub = self.create_publisher(String, self.detections_json_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10) if self.publish_debug_image else None

        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(String, self.desired_mode_topic, self._cb_desired_mode, 10)
        if self.sync_desired_mode_from_landing_cmd:
            self.create_subscription(String, self.landing_cmd_topic, self._cb_landing_cmd, 10)

        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_tx = None
        self.smooth_ty = None
        self.smooth_tz = None
        self.marker_states = {}

        self.marker_filters = {
            self.large_id: PixelErrKalmanCV(
                accel_noise_px_s2=self.kf_accel_noise_px_s2,
                meas_noise_px=self.kf_meas_noise_px_large,
                p0_pos_px=self.kf_p0_pos_px,
                p0_vel_px_s=self.kf_p0_vel_px_s,
                max_dt_s=self.kf_max_dt_s,
                reset_timeout_s=self.kf_reset_timeout_s,
                gate_nis=self.kf_gate_nis,
                max_rate_px_s=self.kf_max_rate_px_s,
            ),
            self.small_id: PixelErrKalmanCV(
                accel_noise_px_s2=self.kf_accel_noise_px_s2,
                meas_noise_px=self.kf_meas_noise_px_small,
                p0_pos_px=self.kf_p0_pos_px,
                p0_vel_px_s=self.kf_p0_vel_px_s,
                max_dt_s=self.kf_max_dt_s,
                reset_timeout_s=self.kf_reset_timeout_s,
                gate_nis=self.kf_gate_nis,
                max_rate_px_s=self.kf_max_rate_px_s,
            ),
        }
        self.last_primary_id = None
        self.latest_landing_cmd = {}
        self.frame_count = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(self.log_root_dir) / f"run_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.session_dir / "detections.csv"
        self.latest_debug_jpg = self.session_dir / "latest_debug.jpg"
        self.debug_video_path = self.session_dir / "debug.avi"
        self.raw_video_path = self.session_dir / "raw.avi"
        self.debug_writer = None
        self.raw_writer = None
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "stamp_sec", "frame", "found", "desired_mode", "id", "mode",
            "err_x", "err_y", "err_x_rate", "err_y_rate", "tvec_z", "ids",
        ])
        self.csv_file.flush()

        self.get_logger().info(
            f"ArucoNode ready | desired_mode={self.desired_mode} | "
            f"dict={self.dict_name} | use_calib={self.use_calib} | "
            f"debug_rotate_180={self.debug_rotate_180} | "
            f"use_kalman_err={self.use_kalman_err} | json_err_source={self.json_err_source} | "
            f"record_debug_video={self.record_debug_video} stride={self.record_video_stride}"
        )

    def flush_csv_maybe(self):
        if self.frame_count % self.csv_flush_every_n == 0:
            self.csv_file.flush()

    def set_desired_mode(self, mode: str, source: str = ""):
        mode = str(mode).strip().upper()
        old = self.desired_mode
        if mode in ("LARGE", "BIG", "2", "BOTH", "ANY"):
            self.desired_mode = "LARGE"
        elif mode in ("SMALL", "1"):
            self.desired_mode = "SMALL"
        else:
            self.get_logger().warn(f"Unknown desired_mode from {source}: {mode}")
            return
        if old != self.desired_mode:
            self.get_logger().info(f"desired_mode: {old} -> {self.desired_mode} by {source}")
            self.smooth_cx = self.smooth_cy = None
            self.smooth_tx = self.smooth_ty = self.smooth_tz = None
            self.last_primary_id = None

    def _cb_desired_mode(self, msg: String):
        self.set_desired_mode(msg.data, "/aruco/desired_mode")

    def _cb_landing_cmd(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.latest_landing_cmd = data
        phase = str(data.get("phase", "")).upper()
        alt_band = str(data.get("alt_band", "")).upper()
        reason = str(data.get("reason", "")).upper()

        if "SMALL" in phase or alt_band == "SMALL_ONLY" or "SMALL_CENTERED" in reason:
            self.set_desired_mode("SMALL", "/landing_cmd")
        elif "BIG" in phase or "LARGE" in phase or alt_band in ("BIG_ONLY", "MIXED"):
            self.set_desired_mode("LARGE", "/landing_cmd")

    def load_calib(self, path_str: str):
        candidates = [Path(path_str), Path("/ws/src/drone_control/config/camera_calib.npz")]
        chosen = None
        for p in candidates:
            if p.exists():
                chosen = p
                break
        if chosen is None:
            self.get_logger().warn("Không tìm thấy camera_calib.npz, vẫn detect ID nhưng không có tvec")
            return
        data = np.load(str(chosen))
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]
        self.get_logger().info(f"Loaded calib: {chosen}")

    def msg_to_bgr(self, msg: Image):
        if msg.encoding == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()
        if msg.encoding == "rgb8":
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if msg.encoding == "mono8":
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        raise RuntimeError(f"Chưa hỗ trợ encoding: {msg.encoding}")

    def bgr_to_msg(self, frame, stamp, frame_id):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * frame.shape[2]
        msg.data = frame.tobytes()
        return msg

    def preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            clahe = cv2.createCLAHE(self.clahe_clip_limit, (self.clahe_grid_size, self.clahe_grid_size))
            gray = clahe.apply(gray)
        if self.use_small_blur:
            k = max(3, self.blur_kernel)
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return gray

    def marker_avg_side_px(self, pts):
        return float(sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)) / 4.0)

    def detect_markers_once(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        markers = []
        if ids is None:
            return markers
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            marker_id = int(marker_id)
            if marker_id not in self.allowed_ids:
                continue
            pts = corners[i].reshape((4, 2)).astype(np.float32)
            avg_side = self.marker_avg_side_px(pts)
            if marker_id == self.small_id and avg_side < self.min_avg_side_px_small:
                continue
            if marker_id == self.large_id and avg_side < self.min_avg_side_px_large:
                continue
            markers.append({"id": marker_id, "corners": pts, "avg_side_px": avg_side})
        return markers

    def merge_duplicate_markers(self, marker_lists):
        merged = []
        for markers in marker_lists:
            for m in markers:
                pts = m["corners"]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                dup_idx = None
                for j, old in enumerate(merged):
                    if int(old["id"]) != int(m["id"]):
                        continue
                    old_pts = old["corners"]
                    old_cx = float(np.mean(old_pts[:, 0]))
                    old_cy = float(np.mean(old_pts[:, 1]))
                    if np.hypot(cx - old_cx, cy - old_cy) <= self.duplicate_center_dist_px:
                        dup_idx = j
                        break
                if dup_idx is None:
                    merged.append(m)
                elif float(m["avg_side_px"]) > float(merged[dup_idx]["avg_side_px"]):
                    merged[dup_idx] = m
        return merged

    def detect_all_markers(self, frame_bgr):
        gray_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        marker_lists = []
        if self.detect_on_raw_gray:
            marker_lists.append(self.detect_markers_once(gray_raw))
        if self.detect_on_processed:
            marker_lists.append(self.detect_markers_once(self.preprocess(frame_bgr)))
        if not marker_lists:
            marker_lists.append(self.detect_markers_once(gray_raw))
        return self.merge_duplicate_markers(marker_lists)

    def choose_primary_marker(self, markers):
        if not markers:
            return None, "NONE"

        if self.desired_mode == "SMALL":
            candidates = [m for m in markers if m["id"] == self.small_id]
            if not candidates:
                return None, "NONE"
            candidates.sort(key=lambda m: m["avg_side_px"], reverse=True)
            return candidates[0], "SMALL"

        if self.small_priority_in_large:
            smalls = [m for m in markers if m["id"] == self.small_id and m["avg_side_px"] >= self.small_priority_min_side_px]
            if smalls:
                smalls.sort(key=lambda m: m["avg_side_px"], reverse=True)
                return smalls[0], "SMALL"

        larges = [m for m in markers if m["id"] == self.large_id]
        if larges:
            larges.sort(key=lambda m: m["avg_side_px"], reverse=True)
            return larges[0], "LARGE"

        if self.allow_small_override_in_large_mode:
            smalls = [m for m in markers if m["id"] == self.small_id and m["avg_side_px"] >= self.small_override_min_side_px]
            if smalls:
                smalls.sort(key=lambda m: m["avg_side_px"], reverse=True)
                return smalls[0], "SMALL"

        return None, "NONE"

    def ema(self, prev, cur):
        if prev is None:
            return cur
        return self.ema_alpha * cur + (1.0 - self.ema_alpha) * prev

    def marker_size_for_id(self, marker_id):
        return self.marker_size_small_m if int(marker_id) == self.small_id else self.marker_size_large_m

    def solve_tvec(self, corners_4x2, marker_size_m):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return None
        marker_points = np.array([
            [-marker_size_m / 2,  marker_size_m / 2, 0],
            [ marker_size_m / 2,  marker_size_m / 2, 0],
            [ marker_size_m / 2, -marker_size_m / 2, 0],
            [-marker_size_m / 2, -marker_size_m / 2, 0],
        ], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            marker_points,
            corners_4x2.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        return rvec, tvec

    def publish_point(self, pub, stamp, frame_id, x, y, z=0.0):
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)
        pub.publish(msg)

    def publish_not_found(self, ids_text=""):
        self.last_primary_id = None
        self.smooth_cx = self.smooth_cy = None
        self.smooth_tx = self.smooth_ty = self.smooth_tz = None
        self.found_pub.publish(Bool(data=False))
        self.selected_id_pub.publish(Int32(data=-1))
        self.selected_mode_pub.publish(String(data="NONE"))
        self.csv_writer.writerow([
            self.get_clock().now().nanoseconds / 1e9,
            self.frame_count,
            False,
            self.desired_mode,
            "",
            "NONE",
            "",
            "",
            "",
            "",
            "",
            ids_text,
        ])
        self.flush_csv_maybe()

    def update_marker_state(
        self,
        marker_id: int,
        center_x: float,
        center_y: float,
        err_x: float,
        err_y: float,
        stamp_sec: float,
        avg_side_px: float = 0.0,
        image_cx: float = 0.0,
        image_cy: float = 0.0,
        detected: bool = True,
    ):
        marker_id = int(marker_id)
        stamp_sec = float(stamp_sec)

        if marker_id == self.small_id:
            base_r = self.kf_meas_noise_px_small
        else:
            base_r = self.kf_meas_noise_px_large

        side = max(1.0, float(avg_side_px))
        if marker_id == self.small_id:
            ref_side = max(10.0, self.small_override_min_side_px)
        else:
            ref_side = max(20.0, self.min_avg_side_px_large)

        size_scale = max(1.0, ref_side / side)
        meas_noise_px = base_r * min(2.5, size_scale)

        filt = self.marker_filters.get(marker_id)
        if filt is None:
            filt = PixelErrKalmanCV(
                accel_noise_px_s2=self.kf_accel_noise_px_s2,
                meas_noise_px=base_r,
                p0_pos_px=self.kf_p0_pos_px,
                p0_vel_px_s=self.kf_p0_vel_px_s,
                max_dt_s=self.kf_max_dt_s,
                reset_timeout_s=self.kf_reset_timeout_s,
                gate_nis=self.kf_gate_nis,
                max_rate_px_s=self.kf_max_rate_px_s,
            )
            self.marker_filters[marker_id] = filt

        accepted = False

        if self.use_kalman_err:
            if detected:
                accepted = filt.update(err_x, err_y, stamp_sec, meas_noise_px=meas_noise_px)
            else:
                filt.predict_to(stamp_sec)

            if filt.initialized:
                ex_f, ey_f, ex_rate, ey_rate = filt.state()
                cx_f = float(image_cx) + ex_f
                cy_f = float(image_cy) + ey_f
            else:
                ex_f = float(err_x)
                ey_f = float(err_y)
                ex_rate = 0.0
                ey_rate = 0.0
                cx_f = float(center_x)
                cy_f = float(center_y)
        else:
            old = self.marker_states.get(marker_id, {})
            alpha = self.json_filter_alpha

            def _ema(prev, cur):
                if prev is None:
                    return float(cur)
                return alpha * float(cur) + (1.0 - alpha) * float(prev)

            ex_f = _ema(old.get("err_x_f"), err_x)
            ey_f = _ema(old.get("err_y_f"), err_y)
            cx_f = _ema(old.get("center_x_f"), center_x)
            cy_f = _ema(old.get("center_y_f"), center_y)

            prev_t = old.get("stamp")
            if prev_t is not None:
                dt = max(1e-3, stamp_sec - float(prev_t))
                ex_rate = (ex_f - float(old.get("err_x_f", ex_f))) / dt
                ey_rate = (ey_f - float(old.get("err_y_f", ey_f))) / dt
            else:
                ex_rate = 0.0
                ey_rate = 0.0
            accepted = True

        age_s = filt.age(stamp_sec) if self.use_kalman_err else 0.0
        kf_pos_std_px = filt.pos_std_px() if self.use_kalman_err and filt.initialized else 999.0
        kf_nis = float(filt.last_nis) if self.use_kalman_err else 0.0
        kf_rejected = bool(filt.last_rejected) if self.use_kalman_err else False

        new_state = {
            "stamp": float(stamp_sec),
            "detected": bool(detected),
            "predict_only": bool(not detected),
            "accepted": bool(accepted),
            "err_x_f": float(ex_f),
            "err_y_f": float(ey_f),
            "center_x_f": float(cx_f),
            "center_y_f": float(cy_f),
            "err_x_rate_px_s": float(ex_rate),
            "err_y_rate_px_s": float(ey_rate),
            "dt": 0.0,
            "kf_age_s": float(age_s),
            "kf_pos_std_px": float(kf_pos_std_px),
            "kf_nis": float(kf_nis),
            "kf_rejected": bool(kf_rejected),
            "meas_noise_px": float(meas_noise_px),
            "samples": int(filt.samples) if self.use_kalman_err else 0,
        }

        self.marker_states[marker_id] = new_state
        return new_state

    def publish_detections_json(self, markers, w: int, h: int):
        img_cx = w * 0.5
        img_cy = h * 0.5
        detections = []
        stamp_sec = self.get_clock().now().nanoseconds / 1e9

        for m in markers:
            marker_id = int(m["id"])
            pts = m["corners"]
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            raw_err_x = float(cx - img_cx)
            raw_err_y = float(cy - img_cy)
            state = self.update_marker_state(
                marker_id,
                cx,
                cy,
                raw_err_x,
                raw_err_y,
                stamp_sec,
                avg_side_px=float(m.get("avg_side_px", 0.0)),
                image_cx=img_cx,
                image_cy=img_cy,
                detected=True,
            )

            if self.json_err_source == "filtered":
                center_x = float(state["center_x_f"])
                center_y = float(state["center_y_f"])
                err_x = float(state["err_x_f"])
                err_y = float(state["err_y_f"])
            else:
                center_x = cx
                center_y = cy
                err_x = raw_err_x
                err_y = raw_err_y

            det = {
                "detected": True,
                "predict_only": False,
                "kf_age_s": float(state["kf_age_s"]),
                "kf_pos_std_px": float(state["kf_pos_std_px"]),
                "kf_nis": float(state["kf_nis"]),
                "kf_rejected": bool(state["kf_rejected"]),
                "kf_samples": int(state["samples"]),
                "meas_noise_px": float(state["meas_noise_px"]),
                "marker_id": marker_id,
                "id": marker_id,
                "role": "large" if marker_id == self.large_id else "small",
                "center_x": center_x,
                "center_y": center_y,
                "err_x": err_x,
                "err_y": err_y,
                "raw_center_x": cx,
                "raw_center_y": cy,
                "raw_err_x": raw_err_x,
                "raw_err_y": raw_err_y,
                "filtered_center_x": float(state["center_x_f"]),
                "filtered_center_y": float(state["center_y_f"]),
                "filtered_err_x": float(state["err_x_f"]),
                "filtered_err_y": float(state["err_y_f"]),
                "err_x_rate_px_s": float(state["err_x_rate_px_s"]),
                "err_y_rate_px_s": float(state["err_y_rate_px_s"]),
                "dt": float(state["dt"]),
                "stamp": float(stamp_sec),
                "frame": int(self.frame_count),
                "size_px": float(m.get("avg_side_px", 0.0)),
                "side_px": float(m.get("avg_side_px", 0.0)),
                "avg_side_px": float(m.get("avg_side_px", 0.0)),
            }
            pose = self.solve_tvec(pts, self.marker_size_for_id(marker_id))
            if pose is not None:
                _, tvec = pose
                tx, ty, tz = [float(v) for v in tvec.flatten()]
                det["x_m"] = tx
                det["y_m"] = ty
                det["z_m"] = tz
                det["tvec_z"] = tz
                det["distance_m"] = float(np.linalg.norm(tvec))
            detections.append(det)

        if self.publish_predict_only_json:
            seen_ids = {int(d["marker_id"]) for d in detections}
            now_sec = self.get_clock().now().nanoseconds / 1e9

            for marker_id in (self.large_id, self.small_id):
                if marker_id in seen_ids:
                    continue

                filt = self.marker_filters.get(marker_id)
                if filt is None or not filt.initialized:
                    continue

                filt.predict_to(now_sec)
                age_s = filt.age(now_sec)
                if age_s > self.kf_predict_timeout_s:
                    continue

                ex_f, ey_f, ex_rate, ey_rate = filt.state()
                center_x = img_cx + ex_f
                center_y = img_cy + ey_f

                state = self.update_marker_state(
                    marker_id,
                    center_x,
                    center_y,
                    ex_f,
                    ey_f,
                    now_sec,
                    avg_side_px=0.0,
                    image_cx=img_cx,
                    image_cy=img_cy,
                    detected=False,
                )

                detections.append({
                    "marker_id": marker_id,
                    "id": marker_id,
                    "role": "large" if marker_id == self.large_id else "small",
                    "detected": False,
                    "predict_only": True,
                    "center_x": float(state["center_x_f"]),
                    "center_y": float(state["center_y_f"]),
                    "err_x": float(state["err_x_f"]),
                    "err_y": float(state["err_y_f"]),
                    "raw_center_x": None,
                    "raw_center_y": None,
                    "raw_err_x": None,
                    "raw_err_y": None,
                    "filtered_center_x": float(state["center_x_f"]),
                    "filtered_center_y": float(state["center_y_f"]),
                    "filtered_err_x": float(state["err_x_f"]),
                    "filtered_err_y": float(state["err_y_f"]),
                    "err_x_rate_px_s": float(state["err_x_rate_px_s"]),
                    "err_y_rate_px_s": float(state["err_y_rate_px_s"]),
                    "kf_age_s": float(state["kf_age_s"]),
                    "kf_pos_std_px": float(state["kf_pos_std_px"]),
                    "kf_nis": float(state["kf_nis"]),
                    "kf_samples": int(state["samples"]),
                    "stamp": float(now_sec),
                    "frame": int(self.frame_count),
                    "size_px": 0.0,
                    "side_px": 0.0,
                    "avg_side_px": 0.0,
                })

        msg = String()
        msg.data = json.dumps({
            "stamp": self.get_clock().now().nanoseconds / 1e9,
            "frame": int(self.frame_count),
            "count": len(detections),
            "desired_mode": self.desired_mode,
            "json_err_source": self.json_err_source,
            "image_width": int(w),
            "image_height": int(h),
            "image_cx": float(img_cx),
            "image_cy": float(img_cy),
            "detections": detections,
        }, ensure_ascii=False, separators=(",", ":"))
        self.detections_json_pub.publish(msg)

    def ensure_video_writers(self, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)
        if self.record_debug_video and self.debug_writer is None:
            self.debug_writer = cv2.VideoWriter(str(self.debug_video_path), fourcc, self.video_fps, (width, height))
            if not self.debug_writer.isOpened():
                self.get_logger().warn(f"Cannot open debug video writer: {self.debug_video_path}")
                self.debug_writer = None
            else:
                self.get_logger().info(f"Recording debug video: {self.debug_video_path}")

        if self.record_raw_video and self.raw_writer is None:
            self.raw_writer = cv2.VideoWriter(str(self.raw_video_path), fourcc, self.video_fps, (width, height))
            if not self.raw_writer.isOpened():
                self.get_logger().warn(f"Cannot open raw video writer: {self.raw_video_path}")
                self.raw_writer = None
            else:
                self.get_logger().info(f"Recording raw video: {self.raw_video_path}")

    def draw_landing_boxes_overlay(self, debug, w: int, h: int):
        if not self.draw_landing_boxes:
            return

        cx = int(w * 0.5)
        cy = int(h * 0.5)

        def _rect(half_x: float, half_y: float, color, thick: int, label: str):
            x1 = int(round(cx - half_x))
            y1 = int(round(cy - half_y))
            x2 = int(round(cx + half_x))
            y2 = int(round(cy + half_y))
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, thick)
            cv2.putText(debug, label, (x1 + 5, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        _rect(
            self.large_box_half_x_px,
            self.large_box_half_y_px,
            (0, 220, 255),
            2,
            f"ID{self.large_id} BOX ±{self.large_box_half_x_px:.0f},{self.large_box_half_y_px:.0f}px",
        )
        _rect(
            self.small_box_half_x_px,
            self.small_box_half_y_px,
            (255, 255, 0),
            2,
            f"ID{self.small_id} BOX ±{self.small_box_half_x_px:.0f},{self.small_box_half_y_px:.0f}px",
        )

        cmd = self.latest_landing_cmd if isinstance(self.latest_landing_cmd, dict) else {}
        phase = str(cmd.get("phase", ""))
        inside = cmd.get("inside_box", None)
        latch = cmd.get("latch_ready", None)
        chx = cmd.get("center_hold_vx", None)
        chy = cmd.get("center_hold_vy", None)
        text = []
        if phase:
            text.append(f"phase={phase}")
        if inside is not None:
            text.append(f"inside={inside}")
        if latch is not None:
            text.append(f"latch={latch}")
        if chx is not None and chy is not None:
            try:
                text.append(f"center_hold=({float(chx):.2f},{float(chy):.2f})")
            except Exception:
                pass
        if text:
            cv2.putText(debug, " | ".join(text), (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

    def make_debug_view(self, frame):
        """
        Xoay ảnh debug để quan sát. Chỉ dùng cho debug output.
        Không ảnh hưởng frame raw dùng để detect và publish JSON.
        """
        if self.debug_rotate_180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def draw_upright_debug_text(self, debug_out, lines, no_marker=False):
        """
        Khi debug_rotate_180=True:
        - Ảnh debug được xoay sau khi đã vẽ marker/line.
        - Text cũ nếu vẽ trước xoay sẽ bị ngược, nên ta không vẽ text cũ nữa.
        - Hàm này vẽ lại text sau khi xoay để chữ luôn xuôi chiều.
        """
        if not self.debug_rotate_180:
            return

        if not lines:
            return

        h, w = debug_out.shape[:2]

        # Nền mờ phía trên để chữ dễ đọc.
        overlay = debug_out.copy()
        bar_h = min(86, max(54, 26 * len(lines) + 18))
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, debug_out, 0.45, 0, debug_out)

        color_main = (0, 0, 255) if no_marker else (0, 255, 0)
        y = 28
        for i, line in enumerate(lines[:3]):
            color = color_main if i == 0 else (255, 255, 255)
            thick = 2 if i == 0 else 1
            cv2.putText(
                debug_out,
                str(line)[:130],
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                thick,
            )
            y += 24

    def image_cb(self, msg: Image):
        try:
            frame = self.msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"Image convert error: {e}")
            return

        self.frame_count += 1
        h, w = frame.shape[:2]
        self.ensure_video_writers(w, h)
        if self.raw_writer is not None:
            self.raw_writer.write(frame)

        debug = frame.copy()
        debug_lines = []
        debug_no_marker = False

        cx_img = w * 0.5
        cy_img = h * 0.5
        cv2.circle(debug, (int(cx_img), int(cy_img)), 4, (255, 0, 0), -1)
        self.draw_landing_boxes_overlay(debug, w, h)

        markers = self.detect_all_markers(frame)
        ids_text = ", ".join([f"{m['id']}:{m['avg_side_px']:.1f}px" for m in markers])
        self.ids_pub.publish(String(data=ids_text))
        self.publish_detections_json(markers, w, h)

        # Vẽ tất cả marker hợp lệ mảnh.
        for m in markers:
            pts_i_all = m["corners"].astype(int)
            color = (0, 180, 255) if int(m["id"]) == self.small_id else (255, 180, 0)
            for j in range(4):
                cv2.line(debug, tuple(pts_i_all[j]), tuple(pts_i_all[(j + 1) % 4]), color, 1)
            cx_all = int(np.mean(pts_i_all[:, 0]))
            cy_all = int(np.mean(pts_i_all[:, 1]))
            cv2.putText(debug, f"ID{int(m['id'])} {float(m['avg_side_px']):.0f}px", (cx_all + 5, cy_all - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        primary, mode = self.choose_primary_marker(markers)
        if primary is None:
            line1 = f"NO desired {self.desired_mode} | {ids_text}"
            debug_lines = [line1]
            debug_no_marker = True

            if not self.debug_rotate_180:
                cv2.putText(
                    debug,
                    line1,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
            self.publish_not_found(ids_text)
        else:
            primary_id = int(primary["id"])
            pts = primary["corners"]
            pts_i = pts.astype(int)
            for j in range(4):
                cv2.line(debug, tuple(pts_i[j]), tuple(pts_i[(j + 1) % 4]), (0, 255, 0), 3)

            target_cx = float(np.mean(pts[:, 0]))
            target_cy = float(np.mean(pts[:, 1]))
            raw_err_x = target_cx - cx_img
            raw_err_y = target_cy - cy_img

            state = self.marker_states.get(int(primary["id"]))
            if state is not None and self.use_kalman_err:
                self.smooth_cx = float(state["center_x_f"])
                self.smooth_cy = float(state["center_y_f"])
                err_x = float(state["err_x_f"])
                err_y = float(state["err_y_f"])
            else:
                self.smooth_cx = self.ema(self.smooth_cx, target_cx)
                self.smooth_cy = self.ema(self.smooth_cy, target_cy)
                err_x = self.smooth_cx - cx_img
                err_y = self.smooth_cy - cy_img
            self.publish_point(self.center_pub, msg.header.stamp, msg.header.frame_id, self.smooth_cx, self.smooth_cy, 0.0)
            self.publish_point(self.error_pub, msg.header.stamp, msg.header.frame_id, err_x, err_y, 0.0)

            tvec_z_out = ""
            pose = self.solve_tvec(pts, self.marker_size_for_id(primary_id))
            if pose is not None:
                rvec, tvec = pose
                tx, ty, tz = tvec.flatten()
                self.smooth_tx = self.ema(self.smooth_tx, tx)
                self.smooth_ty = self.ema(self.smooth_ty, ty)
                self.smooth_tz = self.ema(self.smooth_tz, tz)
                tvec_z_out = self.smooth_tz
                self.publish_point(self.tvec_pub, msg.header.stamp, msg.header.frame_id, self.smooth_tx, self.smooth_ty, self.smooth_tz)
                cv2.drawFrameAxes(debug, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size_for_id(primary_id) * 0.5)

            self.found_pub.publish(Bool(data=True))
            self.selected_id_pub.publish(Int32(data=primary_id))
            self.selected_mode_pub.publish(String(data=mode))

            state = self.marker_states.get(primary_id, {})
            err_x_rate = float(state.get("err_x_rate_px_s", 0.0))
            err_y_rate = float(state.get("err_y_rate_px_s", 0.0))

            cv2.circle(debug, (int(self.smooth_cx), int(self.smooth_cy)), 6, (0, 255, 0), -1)
            cv2.line(debug, (int(cx_img), int(cy_img)), (int(self.smooth_cx), int(self.smooth_cy)), (0, 255, 0), 2)

            line1 = (
                f"DESIRED {self.desired_mode} | {mode} id={primary_id} "
                f"err=({int(err_x)},{int(err_y)}) "
                f"rate=({err_x_rate:.0f},{err_y_rate:.0f}) z={tvec_z_out}"
            )
            line2 = f"JSON {self.json_err_source} all IDs: {ids_text}"
            debug_lines = [line1, line2]
            debug_no_marker = False

            if not self.debug_rotate_180:
                cv2.putText(
                    debug,
                    line1,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    debug,
                    line2,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                )

            self.csv_writer.writerow([
                self.get_clock().now().nanoseconds / 1e9,
                self.frame_count,
                True,
                self.desired_mode,
                primary_id,
                mode,
                err_x,
                err_y,
                err_x_rate,
                err_y_rate,
                tvec_z_out,
                ids_text,
            ])
            self.flush_csv_maybe()

        now_sec = self.get_clock().now().nanoseconds / 1e9

        # Chỉ xoay output debug, không xoay dữ liệu xử lý.
        debug_out = self.make_debug_view(debug)

        # Sau khi xoay debug thì vẽ lại chữ để chữ luôn xuôi chiều.
        self.draw_upright_debug_text(
            debug_out,
            debug_lines,
            no_marker=debug_no_marker,
        )

        if self.write_latest_debug_jpg and (now_sec - self._last_latest_debug_jpg_sec) >= self.latest_debug_jpg_period_s:
            cv2.imwrite(str(self.latest_debug_jpg), debug_out)
            self._last_latest_debug_jpg_sec = now_sec
        if self.debug_writer is not None and (self.frame_count % self.record_video_stride == 0):
            self.debug_writer.write(debug_out)
        if self.publish_debug_image and self.debug_pub is not None:
            self.debug_pub.publish(self.bgr_to_msg(debug_out, msg.header.stamp, msg.header.frame_id))

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        try:
            if self.debug_writer is not None:
                self.debug_writer.release()
        except Exception:
            pass
        try:
            if self.raw_writer is not None:
                self.raw_writer.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
