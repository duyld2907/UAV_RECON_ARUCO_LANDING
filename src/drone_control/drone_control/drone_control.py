#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
drone_control.py - ROS2 + pymavlink

Bản sửa trọng tâm:
- Bay waypoint GPS như cũ, trong lúc bay WP KHÔNG cho ArUco cắt ngang mission.
- Nếu AL_node báo candidate/person nghi ngờ trong lúc bay WP:
    1) Drone phanh/hover tại chỗ.
    2) Publish /detection_hold_active=True để AL_node chuyển sang HOLD verify nghiêm ngặt.
    3) Nếu AL_node confirmed thì AL_node tự publish GPS người lên map.
    4) Hết thời gian verify thì drone tiếp tục bay WP hiện tại.
- Bay lên/ổn định 2s rồi lưu lat/lon hiện tại làm hover-home.
- Bay hết toàn bộ WP thì quay về hover-home.
- Chỉ trên đường về home hoặc lúc search quanh home mới cho ArUco ID150/ID40 cắt ngang.
- Khi thấy ArUco trong return/search:
    1) Latch marker.
    2) Ngừng gửi return/search position.
    3) Phanh ngắn bằng velocity = 0.
    4) Khóa yaw theo heading tại thời điểm thấy marker.
    5) Bật /landing_enable=True.
    6) Nhận /landing_cmd từ landing_node và gửi velocity xuống Pixhawk để kéo tâm.

Sửa lỗi quan trọng so với bản cũ:
- Không phụ thuộc cứng vào self.airborne nữa.
- Nếu cất cánh tay rồi gạt GUIDED, code vẫn cho phép vào landing guidance.
- Nếu đang bay nhưng self.airborne=False, code không gọi takeoff lại.
- Có log LANDING SEND BODY/LOCAL để biết chắc velocity đã gửi xuống Pixhawk.
- Tắt CONDITION_YAW trong ArUco landing.
- Sửa type_mask để yaw_rate=0 thật sự được Pixhawk dùng, giúp khóa yaw-rate khi kéo tâm.

Yêu cầu landing_node:
- Publish /landing_cmd JSON có mode=hold/track/land.
- Có frame=body_ned hoặc local_ned.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from datetime import datetime
from math import atan2, cos, degrees, radians, sin, sqrt
from typing import Optional

import rclpy
from pymavlink import mavutil
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32, Int32, String


SERIAL_PORT = os.environ.get(
    "MAVLINK_PORT",
    "/dev/serial/by-id/usb-Holybro_Pixhawk6C_290035001751333337363133-if00",
)
BAUD_RATE = int(os.environ.get("MAVLINK_BAUD", "115200"))

MODE_GUIDED = 4
MODE_LAND = 9

VALID_ARUCO_IDS = (150, 40)

AUTO_PARAMS = {}
ORIGINAL_PARAMS = {}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = radians(lat1)
    p2 = radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2.0) ** 2 + cos(p1) * cos(p2) * sin(dl / 2.0) ** 2
    return r * 2.0 * atan2(sqrt(a), sqrt(1.0 - a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = radians(lat1)
    p2 = radians(lat2)
    dl = radians(lon2 - lon1)
    y = sin(dl) * cos(p2)
    x = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)
    return (degrees(atan2(y, x)) + 360.0) % 360.0


def offset_gps(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Tạo điểm GPS lệch north/east vài mét quanh home để search ArUco."""
    dlat = north_m / 111111.0
    denom = 111111.0 * max(0.2, cos(radians(lat)))
    dlon = east_m / denom
    return lat + dlat, lon + dlon


class DroneControl(Node):
    def __init__(self):
        super().__init__("drone_control")

        self.master = None

        # ==================== STATE ====================
        self.ready = False
        self.gps_ok = False
        self.armed = False
        self.airborne = False
        self.mission_running = False
        self.auto_mode = False
        self.current_mode = ""
        self.prev_mode = ""
        self.landing = False

        # Current position / heading
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_alt: Optional[float] = None
        self.current_heading: Optional[float] = None

        # Saved hover-home
        self.home_lat: Optional[float] = None
        self.home_lon: Optional[float] = None
        self.home_alt: Optional[float] = None
        self.home_valid = False

        # Lidar / rangefinder
        self.lidar_height: Optional[float] = None
        self.lidar_ground_level = 0.18
        self.airborne_lidar_margin_m = 0.50

        # ==================== MISSION CONFIG ====================
        self.takeoff_alt = 4.5
        self.cruise_alt = 4.5
        self.stabilize_time = 2.0
        self.waypoint_tolerance = 2.0
        self.waypoint_timeout = 60.0
        self.hover_time = 1.0
        self.land_threshold = 0.25
        self.land_noise_margin = 0.05

        # Sau khi bay hết WP: quay về home đã lưu sau takeoff hover.
        self.return_home_tolerance_m = 2.0
        self.return_home_timeout = 90.0

        # Chống lướt qua ArUco khi quay về home:
        # - >5m: còn được gửi GPS position như cũ.
        # - <=5m: chuyển sang velocity chậm về home.
        # - <=2m: không bay thẳng nữa, vào SEARCH_ARUCO pattern quanh home.
        self.return_slow_enter_radius_m = 5.0
        self.search_aruco_enter_radius_m = 2.0
        self.return_slow_speed_mps = 0.30

        # Nếu về home chưa thấy ArUco thì search quanh home trong bán kính ~2m.
        # Search dùng local velocity chậm để không lướt qua marker.
        self.search_radius_m = 2.0
        self.search_point_tolerance_m = 0.60
        self.search_point_hold_s = 2.0
        self.search_waypoint_timeout_s = 10.0
        self.search_total_timeout = 90.0
        self.search_aruco_speed_mps = 0.25

        # Điều khiển yaw khi bay WP: chỉ chạy trong vòng lặp WP, không chạy khi return/search/landing.
        self.wp_yaw_update_period_s = 1.0
        self.last_wp_yaw_t = 0.0

        # Stage hiện tại: IDLE / WP / RETURN_HOME / SEARCH_ARUCO / ARUCO_LANDING
        self.mission_stage = "IDLE"

        # ==================== YAW CONFIG ====================
        self.enable_yaw_to_wp = True
        self.yaw_deadband_deg = 8.0
        self.yaw_speed_deg_s = 18.0

        # Khi vào landing, KHÔNG dùng current_heading thay đổi liên tục.
        # Chốt heading tại thời điểm thấy ArUco rồi ép giữ heading đó.
        self.landing_locked_yaw_deg: Optional[float] = None
        self.landing_yaw_hold_period = 1.00
        self.last_landing_yaw_hold_t = 0.0

        # ==================== LANDING GUIDANCE ====================
        self.landing_guidance_active = False
        self.latest_landing_cmd = None
        self.last_landing_cmd_t = 0.0
        self.landing_cmd_timeout_s = 1.0

        # Phanh ngắn khi thấy ArUco lúc đang bay WP.
        # Nếu để quá dài sẽ có cảm giác detect rồi mà chưa kéo tâm.
        # Tăng thời gian phanh để xóa quán tính khi đang bay về home rồi lướt qua ArUco.
        self.aruco_brake_sec = 0.20
        self.aruco_brake_rate_hz = 30.0
        self.aruco_brake_until_t = 0.0
        self.brake_thread_active = False

        # Debug log gửi velocity xuống Pixhawk.
        self.landing_send_log_period = 0.25
        self.last_landing_send_log_t = 0.0

        # ==================== ARUCO PRIORITY ====================
        self.aruco_seen = False
        self.aruco_latched = False
        self.last_aruco_seen_t = 0.0
        self.last_aruco_id = -1
        self.aruco_seen_timeout = 1.0

        # Chỉ bật True sau khi bay hết WP và bắt đầu return/search.
        # Trong giai đoạn bay WP, thấy ArUco chỉ cập nhật trạng thái, KHÔNG abort WP.
        self.aruco_interrupt_enabled = False

        self.aruco_abort_lock = threading.Lock()
        self.aruco_abort_done = False

        # ==================== WAYPOINTS ====================
        self.waypoints = []
        self.saved_waypoints = []
        self.mission_thread = None
        self.mission_thread_active = False
        self.mission_thread_lock = threading.Lock()

        # ==================== WP ACCURACY LOG ====================
        # Ghi lại tọa độ WP mong muốn và tọa độ GPS thực tế khi UAV báo đã tới WP.
        # File CSV dùng để lập bảng so sánh sai số qua nhiều lần bay thực nghiệm.
        self.wp_accuracy_log_dir = os.environ.get(
            "WP_ACCURACY_LOG_DIR",
            os.path.expanduser("~/UAV_RECON_ARUCO_LANDING/hinh_quy_dao"),
        )
        self.wp_accuracy_log_path = os.path.join(
            self.wp_accuracy_log_dir,
            "wp_accuracy_log.csv",
        )
        self.current_mission_id = ""
        self._init_wp_accuracy_log()

        # ==================== PERSON / AL_NODE HOLD VERIFY ====================
        self.person_hold_active = False
        self.person_hold_until_t = 0.0
        self.person_verify_hold_sec = 2.8
        self.person_confirmed_seen = False
        self.person_last_event_t = 0.0
        self.person_hold_log_period = 0.5
        self.last_person_hold_log_t = 0.0

        # ==================== ROS PUB/SUB ====================
        self.gps_pub = self.create_publisher(NavSatFix, "/drone_gps", 10)
        self.mode_pub = self.create_publisher(String, "/flight_mode", 10)
        self.auto_pub = self.create_publisher(Bool, "/auto_mode", 10)
        self.landing_enable_pub = self.create_publisher(Bool, "/landing_enable", 10)
        self.lidar_pub = self.create_publisher(Float32, "/lidar_height_m", 10)

        # Gửi cho AL_node: False=CRUISE nhẹ, True=HOLD verify nghiêm ngặt.
        self.detection_hold_pub = self.create_publisher(Bool, "/detection_hold_active", 10)

        # Gửi trạng thái/log cho web map.
        # Web node sẽ subscribe 3 topic này để hiển thị flight log bên phải.
        self.mission_state_pub = self.create_publisher(String, "/mission_state", 10)
        self.web_log_pub = self.create_publisher(String, "/web_log", 10)
        self.aruco_status_pub = self.create_publisher(String, "/aruco_status", 10)

        self.last_web_progress_t = 0.0
        self.web_progress_period_s = 0.7
        self.last_aruco_status_pub_t = 0.0
        self.aruco_status_period_s = 0.25

        self.create_subscription(NavSatFix, "/target_gps", self._cb_waypoint, 10)
        self.create_subscription(Bool, "/clear_waypoints", self._cb_clear_waypoints, 10)
        self.create_subscription(Bool, "/drone_takeoff", self._cb_takeoff, 10)
        self.create_subscription(Bool, "/start_trajectory", self._cb_start, 10)
        self.create_subscription(Bool, "/drone_land", self._cb_land, 10)
        self.create_subscription(String, "/landing_cmd", self._cb_landing_cmd, 10)
        self.create_subscription(String, "/person_event", self._cb_person_event, 10)

        # ArUco interrupt sources.
        self.create_subscription(Bool, "/aruco/found", self._cb_aruco_found, 10)
        self.create_subscription(Bool, "/vision/target_found", self._cb_aruco_found, 10)
        self.create_subscription(Bool, "/target_found", self._cb_aruco_found, 10)
        self.create_subscription(Int32, "/aruco/selected_id", self._cb_aruco_selected_id, 10)
        self.create_subscription(Int32, "/aruco/id", self._cb_aruco_selected_id, 10)
        self.create_subscription(String, "/aruco/ids", self._cb_aruco_ids_text, 10)
        self.create_subscription(String, "/aruco/detected_ids", self._cb_aruco_ids_text, 10)
        self.create_subscription(String, "/aruco/status", self._cb_aruco_status_text, 10)
        self.create_subscription(String, "/vision/aruco_detections_json", self._cb_aruco_detections_json, 10)

        self.create_timer(0.5, self._log_lidar)
        self.create_timer(0.1, self._landing_guidance_tick)
        self.create_timer(0.1, self._person_hold_tick)

        self._mission_state("IDLE")
        self._web_log("drone_control started; web map logging enabled")

        threading.Thread(target=self._connect_and_read, daemon=True).start()

    # ==================== PARAMS ====================

    def _set_params(self, params: dict, label: str):
        if self.master is None:
            return
        if not params:
            self.get_logger().info(f"{label} params empty -> skip")
            return
        self.get_logger().info(f"Setting {label} params...")
        for name, value in params.items():
            param_id = name if isinstance(name, bytes) else str(name).encode()
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                param_id,
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            self.get_logger().info(f"  {param_id.decode(errors='ignore')}: {value}")
            time.sleep(0.1)

    def _set_auto_params(self):
        self._set_params(AUTO_PARAMS, "AUTO")

    def _restore_original_params(self):
        self._set_params(ORIGINAL_PARAMS, "ORIGINAL")

    # ==================== MAVLINK CONNECT ====================

    def _connect_and_read(self):
        self._web_log(f"Connecting MAVLink: {SERIAL_PORT} baud={BAUD_RATE}")
        try:
            self.master = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
            self.master.wait_heartbeat()
            self._web_log(f"MAVLink connected! System: {self.master.target_system}")

            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )

            self.ready = True
            time.sleep(1.0)

            self._web_log("Calibrating lidar/rangefinder ground level (3s)...")
            samples = []
            t_cal = time.time()
            while time.time() - t_cal < 3.0:
                if self.lidar_height is not None:
                    samples.append(self.lidar_height)
                time.sleep(0.1)

            if samples:
                self.lidar_ground_level = sum(samples) / len(samples)
                self._web_log(f"Lidar ground: {self.lidar_ground_level:.2f}m")
            else:
                self._web_warn(f"No lidar data, default ground: {self.lidar_ground_level:.2f}m")

            self._web_log("Ready! ARM + GUIDED to fly!")

            while rclpy.ok():
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                if msg is not None:
                    self._handle_message(msg)

        except Exception as e:
            self._web_warn(f"MAVLink connection failed: {e}")
            time.sleep(3.0)
            if rclpy.ok():
                self._connect_and_read()

    def _handle_message(self, msg):
        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT" and self.master is not None:
            src_sys = msg.get_srcSystem()
            if src_sys != self.master.target_system:
                return

        if msg_type == "GLOBAL_POSITION_INT":
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7

            # msg.alt là global altitude nên ở mặt đất vẫn có thể 50-60m.
            # Dùng relative_alt để web map hiển thị đúng độ cao tương đối.
            raw_alt = msg.alt / 1000.0
            rel_alt = getattr(msg, "relative_alt", 0) / 1000.0
            alt = rel_alt

            if lat == 0.0 and lon == 0.0:
                return

            if not self.gps_ok:
                self.gps_ok = True
                self._web_log(
                    f"GPS OK! lat={lat:.7f}, lon={lon:.7f}, rel_alt={rel_alt:.1f}m, raw_alt={raw_alt:.1f}m"
                )

            self.current_lat = lat
            self.current_lon = lon
            self.current_alt = alt

            if hasattr(msg, "hdg") and msg.hdg not in (None, 65535):
                self.current_heading = msg.hdg / 100.0

            out = NavSatFix()
            out.header.stamp = self.get_clock().now().to_msg()
            out.latitude = lat
            out.longitude = lon
            out.altitude = alt
            self.gps_pub.publish(out)

        elif msg_type == "VFR_HUD":
            heading = getattr(msg, "heading", None)
            if heading is not None:
                self.current_heading = float(heading) % 360.0

        elif msg_type == "ATTITUDE":
            yaw_rad = getattr(msg, "yaw", None)
            if yaw_rad is not None and self.current_heading is None:
                self.current_heading = (degrees(yaw_rad) + 360.0) % 360.0

        elif msg_type == "DISTANCE_SENSOR":
            self.lidar_height = msg.current_distance / 100.0
            self._publish_lidar_height()

        elif msg_type == "RANGEFINDER":
            self.lidar_height = msg.distance
            self._publish_lidar_height()

        elif msg_type == "HEARTBEAT":
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if armed != self.armed:
                self.armed = armed
                self._web_log("ARMED" if armed else "DISARMED")
                if not armed and self.landing:
                    self._web_log("Disarmed -> Landed!")
                    self._on_landed()
                if not armed:
                    self.airborne = False

            mode_map = {
                0: "STABILIZE",
                1: "ACRO",
                2: "ALT_HOLD",
                3: "AUTO",
                4: "GUIDED",
                5: "LOITER",
                6: "RTL",
                9: "LAND",
                16: "POSHOLD",
            }
            mode = mode_map.get(msg.custom_mode, f"UNKNOWN({msg.custom_mode})")

            if mode != self.current_mode:
                old_mode = self.current_mode
                self.current_mode = mode
                self.prev_mode = old_mode
                self._web_log(f"Flight mode: {mode}")

                mode_msg = String()
                mode_msg.data = mode
                self.mode_pub.publish(mode_msg)

                is_guided = mode == "GUIDED"
                auto_msg = Bool()
                auto_msg.data = is_guided and self.auto_mode
                self.auto_pub.publish(auto_msg)

                if is_guided and not self.auto_mode:
                    if not self.armed:
                        self._web_warn("AUTO REJECTED: chua ARM. Hay ARM truoc roi gat GUIDED.")
                    else:
                        self._set_auto_params()
                        self.auto_mode = True
                        self._mission_state("GUIDED_AUTO")
                        self._web_log("AUTO MODE ON")
                        threading.Thread(target=self._auto_start, daemon=True).start()

                elif not is_guided and self.auto_mode:
                    self.auto_mode = False
                    self.mission_running = False
                    self.landing_guidance_active = False
                    self.aruco_seen = False
                    self.aruco_latched = False
                    self.aruco_abort_done = False
                    self.aruco_interrupt_enabled = False
                    self.landing_locked_yaw_deg = None
                    self.mission_stage = "IDLE"
                    self._publish_landing_enable(False)
                    self._publish_detection_hold_active(False)
                    self.person_hold_active = False
                    threading.Thread(target=self._restore_original_params, daemon=True).start()
                    self._mission_state("IDLE")
                    self._web_warn(f"AUTO OFF -> {mode}")

        elif msg_type == "COMMAND_ACK":
            if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                self.get_logger().info(f"Arm: {'OK' if msg.result == 0 else f'FAILED({msg.result})'}")

    # ==================== HELPERS ====================

    def _save_current_position_as_home(self, reason: str) -> bool:
        if self.current_lat is None or self.current_lon is None:
            self.home_valid = False
            self._web_warn(f"Could not save home: no GPS position | {reason}")
            return False

        self.home_lat = self.current_lat
        self.home_lon = self.current_lon
        self.home_alt = self.current_alt
        self.home_valid = True
        alt_text = self.home_alt if self.home_alt is not None else 0.0
        self._web_log(
            f"Home SAVED {reason}: lat={self.home_lat:.7f}, lon={self.home_lon:.7f}, alt={alt_text:.1f}m"
        )
        self._mission_state("HOME_SAVED")
        return True

    def _airborne_by_lidar(self) -> bool:
        if self.lidar_height is None:
            return False
        return self.lidar_height > self.lidar_ground_level + self.airborne_lidar_margin_m

    def _airborne_ok(self) -> bool:
        """
        Dùng cho trường hợp cất cánh tay:
        - self.airborne có thể False vì không đi qua _takeoff().
        - Nếu lidar cho thấy đang cao hơn mặt đất, coi như airborne.
        """
        return bool(self.airborne or self._airborne_by_lidar())

    def _mark_airborne_if_needed(self, reason: str):
        if not self.airborne and self._airborne_by_lidar():
            self.airborne = True
            self.get_logger().warn(
                f"Airborne inferred by lidar={self._lidar_info()} | ground={self.lidar_ground_level:.2f}m | {reason}"
            )
            if not self.home_valid:
                self._save_current_position_as_home(f"inferred airborne: {reason}")

    def _on_landed(self):
        self.airborne = False
        self.landing = False
        self.auto_mode = False
        self.mission_running = False
        self.landing_guidance_active = False
        self.aruco_seen = False
        self.aruco_latched = False
        self.aruco_abort_done = False
        self.aruco_interrupt_enabled = False
        self.landing_locked_yaw_deg = None
        self.mission_stage = "IDLE"
        self.person_hold_active = False
        self._publish_landing_enable(False)
        self._publish_detection_hold_active(False)
        threading.Thread(target=self._restore_original_params, daemon=True).start()
        self._mission_state("LANDED")
        self._web_log("Landed!")

    def _publish_lidar_height(self):
        if self.lidar_height is None:
            return
        msg = Float32()
        msg.data = float(self.lidar_height)
        self.lidar_pub.publish(msg)

    def _log_lidar(self):
        if self.lidar_height is not None:
            self._publish_lidar_height()
            airborne_txt = "AIRBORNE" if self._airborne_ok() else "GROUND/LOW"
            self.get_logger().info(
                f"Lidar: {self.lidar_height:.2f}m | ground={self.lidar_ground_level:.2f}m | {airborne_txt}"
            )
        else:
            self.get_logger().info("Lidar: no data")

    def _lidar_info(self) -> str:
        return f"{self.lidar_height:.2f}m" if self.lidar_height is not None else "N/A"

    def _is_landed(self) -> bool:
        if self.lidar_height is None:
            return False
        near_ground = abs(self.lidar_height - self.lidar_ground_level) <= self.land_noise_margin
        below_threshold = self.lidar_height < self.land_threshold
        return below_threshold or near_ground

    def _angle_diff_deg(self, target_deg: float, current_deg: float) -> float:
        return ((target_deg - current_deg + 540.0) % 360.0) - 180.0

    def _publish_landing_enable(self, enabled: bool):
        msg = Bool()
        msg.data = bool(enabled)
        self.landing_enable_pub.publish(msg)

    def _publish_detection_hold_active(self, enabled: bool):
        msg = Bool()
        msg.data = bool(enabled)
        self.detection_hold_pub.publish(msg)

    def _publish_text(self, pub, text: str):
        msg = String()
        msg.data = str(text)
        pub.publish(msg)

    def _web_log(self, text: str):
        text = str(text)
        self.get_logger().info(text)
        self._publish_text(self.web_log_pub, text)

    def _web_warn(self, text: str):
        text = str(text)
        self.get_logger().warn(text)
        self._publish_text(self.web_log_pub, text)

    def _mission_state(self, text: str):
        self._publish_text(self.mission_state_pub, str(text))

    def _aruco_status(self, text: str):
        self._publish_text(self.aruco_status_pub, str(text))

    def _web_progress(self, text: str, period_s: Optional[float] = None):
        now = time.time()
        period = self.web_progress_period_s if period_s is None else float(period_s)
        if now - self.last_web_progress_t < period:
            return
        self.last_web_progress_t = now
        self._publish_text(self.web_log_pub, str(text))

    def _init_wp_accuracy_log(self):
        """
        Tạo file CSV lưu sai số waypoint.
        File này dùng để lập bảng so sánh lat/lon target và lat/lon reached sau các lần bay.
        """
        try:
            os.makedirs(self.wp_accuracy_log_dir, exist_ok=True)

            file_exists = os.path.exists(self.wp_accuracy_log_path)
            if not file_exists:
                with open(self.wp_accuracy_log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "time",
                        "mission_id",
                        "wp_index",
                        "target_lat",
                        "target_lon",
                        "target_alt_m",
                        "reached_lat",
                        "reached_lon",
                        "reached_alt_m",
                        "error_north_m",
                        "error_east_m",
                        "error_total_m",
                    ])

            self.get_logger().info(f"WP accuracy log file: {self.wp_accuracy_log_path}")

        except Exception as e:
            self.get_logger().warn(f"Cannot init WP accuracy log: {e}")

    def _log_wp_accuracy(
        self,
        wp_index: int,
        target_lat: float,
        target_lon: float,
        target_alt: float,
        reached_lat: float,
        reached_lon: float,
        reached_alt: float,
    ) -> float:
        """
        Lưu sai số waypoint:
        - target_lat/lon: vị trí WP người dùng chấm trên map.
        - reached_lat/lon: vị trí GPS thực tế tại thời điểm drone báo REACHED_WP.
        - error_total_m: sai số tổng theo Haversine.
        - error_north_m, error_east_m: sai số xấp xỉ theo phương Bắc/Đông.
        """
        try:
            error_total_m = haversine(target_lat, target_lon, reached_lat, reached_lon)
            error_north_m = (reached_lat - target_lat) * 111111.0
            error_east_m = (
                (reached_lon - target_lon)
                * 111111.0
                * max(0.2, cos(radians(target_lat)))
            )

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.wp_accuracy_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now_str,
                    self.current_mission_id,
                    int(wp_index),
                    f"{target_lat:.8f}",
                    f"{target_lon:.8f}",
                    f"{target_alt:.2f}",
                    f"{reached_lat:.8f}",
                    f"{reached_lon:.8f}",
                    f"{reached_alt:.2f}",
                    f"{error_north_m:.3f}",
                    f"{error_east_m:.3f}",
                    f"{error_total_m:.3f}",
                ])

            self._web_log(
                f"WP{wp_index} accuracy saved | "
                f"target=({target_lat:.7f},{target_lon:.7f}) | "
                f"reached=({reached_lat:.7f},{reached_lon:.7f}) | "
                f"err={error_total_m:.2f}m"
            )

            return error_total_m

        except Exception as e:
            self._web_warn(f"Cannot save WP accuracy: {e}")
            return -1.0

    def _start_person_verify_hold(self, reason: str):
        """AL_node candidate -> drone phanh để AL_node chuyển HOLD và xác minh kỹ hơn."""
        if self.current_mode != "GUIDED" or not self.armed or not self.auto_mode:
            return
        if self.mission_stage != "WP":
            return
        if self.landing_guidance_active or self.landing:
            return

        now = time.time()
        self.person_hold_active = True
        self.person_hold_until_t = max(self.person_hold_until_t, now + self.person_verify_hold_sec)
        self._publish_detection_hold_active(True)

        # Phanh ngay nhưng không bật landing.
        self._send_local_velocity(0.0, 0.0, 0.0)
        self._send_body_velocity(0.0, 0.0, 0.0)

        text = f"PERSON CANDIDATE -> BRAKE/HOLD {self.person_verify_hold_sec:.1f}s -> AL HOLD VERIFY | {reason}"
        self._mission_state("PERSON_HOLD_VERIFY")
        self._web_warn(text)

    def _person_hold_tick(self):
        if not self.person_hold_active:
            return

        now = time.time()
        if now >= self.person_hold_until_t:
            self.person_hold_active = False
            self._publish_detection_hold_active(False)
            self._mission_state("PERSON_HOLD_DONE")
            self._web_log("PERSON HOLD done -> resume current WP")
            return

        # Trong lúc AL verify thì giữ hover/zero velocity, không gửi WP position.
        self._send_local_velocity(0.0, 0.0, 0.0)
        self._send_body_velocity(0.0, 0.0, 0.0)

        if now - self.last_person_hold_log_t >= self.person_hold_log_period:
            self.last_person_hold_log_t = now
            self._web_progress(f"PERSON HOLD active, remaining={self.person_hold_until_t - now:.1f}s", 0.5)

    # ==================== ARUCO INTERRUPT ====================

    def _is_valid_aruco_id(self, marker_id: int) -> bool:
        return marker_id in VALID_ARUCO_IDS

    def _aruco_interrupt_allowed(self) -> bool:
        """
        Sửa chính:
        - Bản cũ yêu cầu self.airborne=True nên nếu cất cánh tay sẽ bị chặn.
        - Bản mới cho phép khi GUIDED + ARMED.
        - Nếu lidar xác nhận đang bay thì set self.airborne=True luôn.
        """
        if self.current_mode != "GUIDED":
            return False
        if not self.armed:
            return False
        if self.landing_guidance_active or self.landing:
            return False

        # Logic mới: ArUco chỉ được cắt ngang khi đã bay hết WP và đang RETURN_HOME/SEARCH_ARUCO.
        # Trong giai đoạn WP, ArUco có thể được detect nhưng không abort mission.
        if not self.aruco_interrupt_enabled or self.mission_stage not in {"RETURN_HOME", "SEARCH_ARUCO"}:
            return False

        self._mark_airborne_if_needed("aruco_interrupt_allowed")

        # Nếu có lidar thì cần cao hơn mặt đất một chút.
        if self.lidar_height is not None and not self._airborne_ok():
            self.get_logger().warn(
                f"ARUCO ignored: GUIDED+ARMED but lidar low {self._lidar_info()} ground={self.lidar_ground_level:.2f}m"
            )
            return False

        # Cho phép kể cả auto_mode chưa kịp True, miễn là GUIDED + ARMED.
        if not self.auto_mode:
            self.auto_mode = True
            self.get_logger().warn("auto_mode forced ON by ArUco interrupt because vehicle is GUIDED+ARMED")

        return True

    def _aruco_recent(self) -> bool:
        if not self._aruco_interrupt_allowed():
            return False
        if self.aruco_latched:
            return True
        return self.aruco_seen and (time.time() - self.last_aruco_seen_t <= self.aruco_seen_timeout)

    def _latch_aruco_and_start_landing(self, source: str, marker_id: int = -1):
        if marker_id not in VALID_ARUCO_IDS:
            marker_id = self.last_aruco_id

        # Nếu vẫn không biết ID, chỉ cho topic found_bool latch.
        if marker_id not in VALID_ARUCO_IDS and source != "found_bool":
            self.get_logger().warn(f"Ignore ArUco latch: invalid id={marker_id} source={source}")
            return

        if not self._aruco_interrupt_allowed():
            return

        self.aruco_seen = True
        self.aruco_latched = True
        self.last_aruco_seen_t = time.time()
        self.last_aruco_id = int(marker_id)

        self._abort_wp_and_start_landing(f"ArUco seen from {source} id={self.last_aruco_id}")

    def _abort_wp_and_start_landing(self, reason: str):
        with self.aruco_abort_lock:
            if self.aruco_abort_done:
                return
            self.aruco_abort_done = True

            # Chốt yaw tại thời điểm thấy marker.
            if self.current_heading is not None:
                self.landing_locked_yaw_deg = float(self.current_heading) % 360.0
            else:
                self.landing_locked_yaw_deg = None

            # Cắt return/search ngay lập tức.
            self.mission_running = False
            self.mission_stage = "ARUCO_LANDING"
            self.aruco_interrupt_enabled = False
            self.person_hold_active = False
            self._publish_detection_hold_active(False)
            self.landing_guidance_active = True
            self.latest_landing_cmd = None
            self.last_landing_cmd_t = 0.0
            self._mark_airborne_if_needed("start_landing_guidance")

            # Phanh ngắn: velocity 0 liên tục, rồi landing_node được kéo XY.
            self.aruco_brake_until_t = time.time() + self.aruco_brake_sec
            self._hard_brake_now()
            self._publish_landing_enable(True)
            self._start_aruco_brake_thread(reason)

            yaw_txt = "N/A" if self.landing_locked_yaw_deg is None else f"{self.landing_locked_yaw_deg:.1f}deg"
            text = (
                f"ARUCO LATCH -> ABORT WP -> HARD BRAKE {self.aruco_brake_sec:.2f}s -> LANDING GUIDANCE ON | "
                f"yaw_lock={yaw_txt} | {reason}"
            )
            self._mission_state("ARUCO_LANDING")
            self._web_warn(text)
            self._aruco_status(text)

    # ==================== ROS CALLBACKS ====================

    def _cb_waypoint(self, msg: NavSatFix):
        if self.mission_running or self.landing_guidance_active or self.landing:
            self.get_logger().warn("Ignore new waypoint: mission/landing is active")
            return

        new_lat = float(msg.latitude)
        new_lon = float(msg.longitude)
        if self.waypoints:
            last = self.waypoints[-1]
            d_last = haversine(last["lat"], last["lon"], new_lat, new_lon)
            if d_last < 0.8:
                self.get_logger().warn(f"Ignore duplicate waypoint: d_last={d_last:.2f}m")
                return

        self.waypoints.append({"lat": new_lat, "lon": new_lon})
        self.saved_waypoints = list(self.waypoints)

        wp_idx = len(self.waypoints)
        self._web_log(f"WP{wp_idx} added: lat={new_lat:.7f}, lon={new_lon:.7f}")
        self._mission_state(f"WP{wp_idx}_ADDED")

    def _cb_clear_waypoints(self, msg: Bool):
        if not msg.data:
            return
        if self.mission_running or self.landing_guidance_active or self.landing:
            self.get_logger().warn("Ignore clear waypoints: mission/landing is active")
            return
        self.waypoints.clear()
        self.saved_waypoints.clear()
        self._mission_state("WPS_CLEARED")
        self._web_warn("Waypoints CLEARED")

    def _cb_takeoff(self, msg: Bool):
        if msg.data:
            threading.Thread(target=self._takeoff, daemon=True).start()

    def _cb_start(self, msg: Bool):
        if msg.data:
            self._web_warn("START received. Mission can also start automatically when GUIDED+ARMED.")
            self._start_fly_mission_thread("start_trajectory")

    def _cb_land(self, msg: Bool):
        if msg.data:
            threading.Thread(target=self._land, daemon=True).start()

    def _cb_landing_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
            if not isinstance(cmd, dict):
                return
            self.latest_landing_cmd = cmd
            self.last_landing_cmd_t = time.time()
            self._relay_landing_cmd_to_aruco_status(cmd)
        except Exception as e:
            self._web_warn(f"landing_cmd JSON invalid: {e}")

    def _relay_landing_cmd_to_aruco_status(self, cmd: dict):
        now = time.time()
        if now - self.last_aruco_status_pub_t < self.aruco_status_period_s:
            return
        self.last_aruco_status_pub_t = now

        marker_id = cmd.get("marker_id", self.last_aruco_id)
        try:
            marker_id = int(marker_id) if marker_id is not None else -1
        except Exception:
            marker_id = -1

        marker_name = "BIG" if marker_id == 150 else "SMALL" if marker_id == 40 else "ARUCO"
        mode = str(cmd.get("mode", ""))
        phase = str(cmd.get("phase", ""))
        reason = str(cmd.get("reason", ""))

        err_x = cmd.get("err_x", cmd.get("filtered_err_x", None))
        err_y = cmd.get("err_y", cmd.get("filtered_err_y", None))
        alt = cmd.get("alt_m", None)
        vx = cmd.get("vx", cmd.get("vx_body", cmd.get("vx_local", None)))
        vy = cmd.get("vy", cmd.get("vy_body", cmd.get("vy_local", None)))
        vz = cmd.get("vz_down", None)

        parts = [marker_name, f"mode={mode}"]
        if phase:
            parts.append(f"phase={phase}")
        if err_x is not None and err_y is not None:
            try:
                parts.append(f"err_x={float(err_x):.1f}")
                parts.append(f"err_y={float(err_y):.1f}")
            except Exception:
                pass
        if alt is not None:
            try:
                parts.append(f"alt={float(alt):.2f}m")
            except Exception:
                pass
        if vx is not None and vy is not None and vz is not None:
            try:
                parts.append(f"vx={float(vx):.2f}")
                parts.append(f"vy={float(vy):.2f}")
                parts.append(f"vz={float(vz):.2f}")
            except Exception:
                pass
        if reason:
            parts.append(f"reason={reason}")

        self._aruco_status(" ".join(parts))

    def _cb_person_event(self, msg: String):
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
        except Exception as e:
            self.get_logger().warn(f"person_event JSON invalid: {e}")
            return

        event = str(payload.get("event", "")).lower()
        score = payload.get("score", None)
        self.person_last_event_t = time.time()

        if event == "candidate":
            self._start_person_verify_hold(f"score={score}")
            return

        if event == "confirmed":
            self.person_confirmed_seen = True
            self._mission_state("PERSON_CONFIRMED")
            self._web_warn(
                f"PERSON CONFIRMED -> AL_node should publish /detected_object_gps | "
                f"lat={payload.get('drone_lat')} lon={payload.get('drone_lon')} score={score}"
            )
            # Giữ thêm rất ngắn để AL_node kịp publish GPS/snapshot rồi tiếp tục WP.
            if self.mission_stage == "WP":
                self.person_hold_active = True
                self.person_hold_until_t = max(self.person_hold_until_t, time.time() + 0.6)
                self._publish_detection_hold_active(True)

    def _cb_aruco_detections_json(self, msg: String):
        try:
            payload = json.loads(msg.data)
            detections = payload.get("detections", []) if isinstance(payload, dict) else payload
            if not isinstance(detections, list):
                return

            best_id = -1
            best_score = None
            best_ex = 0.0
            best_ey = 0.0
            for det in detections:
                if not isinstance(det, dict):
                    continue
                raw_id = det.get("marker_id", det.get("id", det.get("aruco_id")))
                if raw_id is None:
                    continue
                try:
                    mid = int(raw_id)
                except Exception:
                    continue
                if not self._is_valid_aruco_id(mid):
                    continue

                try:
                    ex = float(det.get("filtered_err_x", det.get("err_x", det.get("raw_err_x", 0.0))) or 0.0)
                    ey = float(det.get("filtered_err_y", det.get("err_y", det.get("raw_err_y", 0.0))) or 0.0)
                    score = abs(ex) + abs(ey)
                except Exception:
                    ex = 0.0
                    ey = 0.0
                    score = 0.0

                if best_score is None or score < best_score:
                    best_score = score
                    best_id = mid
                    best_ex = ex
                    best_ey = ey

            if best_id in VALID_ARUCO_IDS:
                self.aruco_seen = True
                self.last_aruco_seen_t = time.time()
                self.last_aruco_id = best_id

                marker_name = "BIG" if best_id == 150 else "SMALL"
                now = time.time()
                if now - self.last_aruco_status_pub_t >= self.aruco_status_period_s:
                    self.last_aruco_status_pub_t = now
                    self._aruco_status(
                        f"{marker_name} id={best_id} err_x={best_ex:.1f} err_y={best_ey:.1f} stage={self.mission_stage}"
                    )

                self._latch_aruco_and_start_landing("/vision/aruco_detections_json", best_id)

        except Exception as e:
            self._web_warn(f"aruco_detections_json parse error: {e}")

    def _cb_aruco_found(self, msg: Bool):
        if msg.data:
            # Topic Bool không có ID. Nếu last_aruco_id hợp lệ thì latch.
            # Nếu chưa có ID, vẫn cho latch vì nhiều aruco_node chỉ publish /aruco/found.
            self._latch_aruco_and_start_landing("found_bool", self.last_aruco_id)
        else:
            if not self.aruco_latched and time.time() - self.last_aruco_seen_t > self.aruco_seen_timeout:
                self.aruco_seen = False

    def _cb_aruco_selected_id(self, msg: Int32):
        marker_id = int(msg.data)
        self.last_aruco_id = marker_id
        if self._is_valid_aruco_id(marker_id):
            self._latch_aruco_and_start_landing("selected_id", marker_id)

    def _cb_aruco_ids_text(self, msg: String):
        data = msg.data or ""
        marker_id = -1
        if "150" in data:
            marker_id = 150
        elif "40" in data:
            marker_id = 40
        if marker_id in VALID_ARUCO_IDS:
            self.last_aruco_id = marker_id
            self._latch_aruco_and_start_landing("ids_text", marker_id)

    def _cb_aruco_status_text(self, msg: String):
        data = (msg.data or "").upper()
        marker_id = -1
        if "ID=150" in data or "150" in data:
            marker_id = 150
        elif "ID=40" in data or "40" in data:
            marker_id = 40
        if marker_id in VALID_ARUCO_IDS:
            self.last_aruco_id = marker_id
            self._latch_aruco_and_start_landing("status_text", marker_id)

    # ==================== MAVLINK COMMANDS ====================

    def _set_mode(self, mode_num: int) -> bool:
        if self.master is None:
            self.get_logger().warn("Set mode failed: MAVLink not connected")
            return False

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_num,
        )

        t_start = time.time()
        while time.time() - t_start < 3.0:
            if mode_num == MODE_GUIDED and self.current_mode == "GUIDED":
                return True
            if mode_num == MODE_LAND and self.current_mode == "LAND":
                return True
            time.sleep(0.1)
        return False

    def _arm(self) -> bool:
        if self.master is None:
            return False
        if not self.gps_ok:
            self.get_logger().warn("No GPS!")
            return False

        self.get_logger().info("Arming...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

        t_start = time.time()
        while time.time() - t_start < 5.0:
            if self.armed:
                self.get_logger().info("Armed!")
                return True
            time.sleep(0.1)
        self.get_logger().warn("Arm timeout!")
        return False

    def _send_position(self, lat: float, lon: float, alt: float):
        if self.master is None or lat is None or lon is None:
            return

        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7),
            int(lon * 1e7),
            float(alt),
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )

    def _ned_error_to_gps_target(self, lat: float, lon: float) -> tuple[float, float, float]:
        """
        Sai số từ vị trí hiện tại tới target GPS theo trục local N/E.
        Return: north_m, east_m, dist_m.
        """
        if self.current_lat is None or self.current_lon is None:
            return 0.0, 0.0, 0.0

        north_m = (float(lat) - float(self.current_lat)) * 111111.0
        east_m = (
            (float(lon) - float(self.current_lon))
            * 111111.0
            * max(0.2, cos(radians(float(self.current_lat))))
        )
        dist_m = sqrt(north_m * north_m + east_m * east_m)
        return north_m, east_m, dist_m

    def _send_velocity_to_gps_target(self, lat: float, lon: float, max_speed_mps: float) -> float:
        """
        Bay về target GPS bằng LOCAL_NED velocity chậm.
        Không đổi param ArduPilot, chỉ gửi vận tốc runtime.
        """
        north_m, east_m, dist_m = self._ned_error_to_gps_target(lat, lon)
        if dist_m < 0.05:
            self._send_local_velocity(0.0, 0.0, 0.0)
            return dist_m

        speed = max(0.05, float(max_speed_mps))
        speed = min(speed, dist_m)  # gần điểm thì tự giảm để khỏi overshoot
        vn = speed * north_m / dist_m
        ve = speed * east_m / dist_m
        self._send_local_velocity(vn, ve, 0.0)
        return dist_m

    def _hold_and_watch_aruco(self, hold_s: float, label: str) -> bool:
        """
        Đứng/chạy zero velocity tại điểm search và tiếp tục xem ArUco.
        Return True nếu đã chuyển sang landing.
        """
        t0 = time.time()
        while time.time() - t0 < hold_s:
            if not self.auto_mode or self.current_mode != "GUIDED" or not self.armed:
                return False
            if self.landing_guidance_active or self.landing:
                return True
            if self._aruco_recent():
                self._abort_wp_and_start_landing(f"ArUco recent during {label}_hold")
                return True
            self._send_local_velocity(0.0, 0.0, 0.0)
            self._send_body_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)
        return False

    def _velocity_type_mask(self) -> int:
        """
        SET_POSITION_TARGET_LOCAL_NED type_mask.

        Bản cũ dùng 0b0000111111000111:
        - dùng vx/vy/vz
        - nhưng IGNORE cả yaw và yaw_rate
        => yaw_rate=0 gửi xuống cuối message KHÔNG có tác dụng.

        Bản mới dùng 0b0000011111000111:
        - dùng vx/vy/vz
        - ignore yaw absolute
        - KHÔNG ignore yaw_rate
        => yaw_rate=0.0 được Pixhawk dùng để giữ không xoay yaw.
        """
        return 0b0000011111000111

    def _send_body_velocity(self, vx_body: float, vy_body: float, vz_down: float, yaw_rate_rad_s: float = 0.0):
        """
        Gửi BODY_NED velocity kèm yaw_rate=0 để khóa không cho yaw quay.

        Lưu ý:
        - yaw_rate trong MAVLink là rad/s.
        - 0.0 rad/s = không xoay yaw.
        - Không dùng CONDITION_YAW ở đây để tránh drone quay vòng.
        """
        if self.master is None:
            return
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            self._velocity_type_mask(),
            0, 0, 0,
            float(vx_body),
            float(vy_body),
            float(vz_down),
            0, 0, 0,
            0.0,
            float(yaw_rate_rad_s),
        )

    def _send_local_velocity(self, vx_north: float, vy_east: float, vz_down: float, yaw_rate_rad_s: float = 0.0):
        """
        Gửi LOCAL_NED velocity kèm yaw_rate=0 để khóa không cho yaw quay.
        """
        if self.master is None:
            return
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            self._velocity_type_mask(),
            0, 0, 0,
            float(vx_north),
            float(vy_east),
            float(vz_down),
            0, 0, 0,
            0.0,
            float(yaw_rate_rad_s),
        )

    def _set_yaw(self, yaw_deg: float):
        if self.master is None or self.current_heading is None:
            return

        diff = self._angle_diff_deg(yaw_deg, self.current_heading)
        if abs(diff) <= self.yaw_deadband_deg:
            return

        direction = 1 if diff > 0 else -1
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            float(yaw_deg),
            float(self.yaw_speed_deg_s),
            float(direction),
            0,
            0, 0, 0,
        )
        self.get_logger().info(
            f"YAW set: target={yaw_deg:.1f} current={self.current_heading:.1f} diff={diff:.1f}"
        )

    def _hold_landing_yaw(self):
        """
        TEST MODE: tắt hoàn toàn MAV_CMD_CONDITION_YAW.

        Mục đích:
        - Kiểm tra xem drone quay vòng có phải do CONDITION_YAW hay không.
        - Landing vẫn kéo tâm bằng vx/vy từ /landing_cmd.
        - Không gửi bất kỳ yaw command nào xuống Pixhawk trong lúc ArUco landing.
        """
        return

    def _hard_brake_now(self):
        # Gửi nhiều gói zero velocity ngay để xóa quán tính lệnh waypoint cũ.
        for _ in range(6):
            self._send_local_velocity(0.0, 0.0, 0.0)
            self._send_body_velocity(0.0, 0.0, 0.0)
            self._hold_landing_yaw()
            time.sleep(0.025)
        self.last_landing_yaw_hold_t = time.time()

    def _start_aruco_brake_thread(self, reason: str):
        if self.brake_thread_active:
            return

        def worker():
            self.brake_thread_active = True
            try:
                dt = 1.0 / max(self.aruco_brake_rate_hz, 1.0)
                while rclpy.ok() and self.landing_guidance_active and time.time() < self.aruco_brake_until_t:
                    self._send_local_velocity(0.0, 0.0, 0.0)
                    self._send_body_velocity(0.0, 0.0, 0.0)
                    self._hold_landing_yaw()
                    time.sleep(dt)
            finally:
                self.brake_thread_active = False
                self.get_logger().info(f"Hard-brake finished -> landing_node may pull XY now | {reason}")

        threading.Thread(target=worker, daemon=True).start()

    # ==================== LANDING GUIDANCE ====================

    def _log_landing_send(self, frame: str, vx: float, vy: float, vz: float, reason: str):
        now = time.time()
        if now - self.last_landing_send_log_t < self.landing_send_log_period:
            return
        self.last_landing_send_log_t = now
        text = f"LANDING SEND {frame.upper()} vx={vx:.3f} vy={vy:.3f} vz_down={vz:.3f} reason={reason}"
        self.get_logger().info(text)
        self._web_progress(text, self.landing_send_log_period)

    def _landing_guidance_tick(self):
        if not self.landing_guidance_active or self.landing:
            return

        now = time.time()

        # Trong đoạn phanh: bỏ qua track cmd, chỉ zero velocity + giữ yaw.
        if now < self.aruco_brake_until_t:
            self._send_local_velocity(0.0, 0.0, 0.0)
            self._send_body_velocity(0.0, 0.0, 0.0)
            self._hold_landing_yaw()
            return

        if now - self.last_landing_yaw_hold_t >= self.landing_yaw_hold_period:
            self._hold_landing_yaw()
            self.last_landing_yaw_hold_t = now

        # Nếu chưa có lệnh landing_node hoặc lệnh cũ: hover.
        if self.latest_landing_cmd is None or now - self.last_landing_cmd_t > self.landing_cmd_timeout_s:
            self._send_local_velocity(0.0, 0.0, 0.0)
            self._send_body_velocity(0.0, 0.0, 0.0)
            self._log_landing_send("hover", 0.0, 0.0, 0.0, "no_or_stale_landing_cmd")
            return

        mode = str(self.latest_landing_cmd.get("mode", "")).lower()
        reason = str(self.latest_landing_cmd.get("reason", ""))

        if mode == "hold":
            self._send_local_velocity(0.0, 0.0, 0.0)
            self._send_body_velocity(0.0, 0.0, 0.0)
            self._log_landing_send("hold", 0.0, 0.0, 0.0, reason)
            return

        if mode == "track":
            frame = str(
                self.latest_landing_cmd.get(
                    "frame",
                    self.latest_landing_cmd.get("command_frame", "body_ned"),
                )
            ).lower()

            try:
                vz = float(self.latest_landing_cmd.get("vz_down", 0.0) or 0.0)
            except Exception:
                vz = 0.0

            # landing_node hiện gửi yaw_rate=0.0.
            # Với type_mask mới, giá trị này KHÔNG bị ignore nữa.
            try:
                yaw_rate = float(self.latest_landing_cmd.get("yaw_rate", 0.0) or 0.0)
            except Exception:
                yaw_rate = 0.0

            if frame.startswith("local"):
                try:
                    vx = float(self.latest_landing_cmd.get("vx_local", self.latest_landing_cmd.get("vx", 0.0)) or 0.0)
                    vy = float(self.latest_landing_cmd.get("vy_local", self.latest_landing_cmd.get("vy", 0.0)) or 0.0)
                except Exception:
                    vx, vy = 0.0, 0.0
                self._send_local_velocity(vx, vy, vz, yaw_rate)
                self._log_landing_send("local", vx, vy, vz, reason)
            else:
                try:
                    vx = float(self.latest_landing_cmd.get("vx_body", self.latest_landing_cmd.get("vx", 0.0)) or 0.0)
                    vy = float(self.latest_landing_cmd.get("vy_body", self.latest_landing_cmd.get("vy", 0.0)) or 0.0)
                except Exception:
                    vx, vy = 0.0, 0.0
                self._send_body_velocity(vx, vy, vz, yaw_rate)
                self._log_landing_send("body", vx, vy, vz, reason)
            return

        if mode == "land":
            self._web_log("landing_cmd -> LAND")
            threading.Thread(target=self._land, daemon=True).start()
            return

        # Mode lạ thì hover.
        self._send_local_velocity(0.0, 0.0, 0.0)
        self._send_body_velocity(0.0, 0.0, 0.0)
        self._log_landing_send("unknown", 0.0, 0.0, 0.0, f"unknown_mode={mode}")

    # ==================== TAKEOFF / MISSION / LAND ====================

    def _auto_start(self):
        if not self.waypoints and self.saved_waypoints:
            self.waypoints = list(self.saved_waypoints)
            self._web_log(f"Reusing {len(self.waypoints)} saved WPs")

        self._mark_airborne_if_needed("auto_start")

        # Sửa: nếu đã cất cánh tay, không gọi takeoff lại.
        if not self._airborne_ok():
            self._takeoff()
        else:
            self.airborne = True
            if not self.home_valid:
                self._save_current_position_as_home("manual takeoff then GUIDED")
            self._web_warn("Already airborne by lidar/manual flight -> skip code takeoff")

        if not self.mission_running and not self.landing and not self.landing_guidance_active:
            if self.waypoints:
                self._start_fly_mission_thread("auto_start")
            else:
                self._web_warn(
                    "No waypoints! Hovering; ArUco landing guidance will start when marker is detected."
                )

    def _start_fly_mission_thread(self, source: str):
        t_wait = time.time()
        while self.auto_mode and self.current_mode != "GUIDED" and time.time() - t_wait < 1.0:
            time.sleep(0.05)

        if not self.auto_mode or self.current_mode != "GUIDED":
            self.get_logger().warn(f"Ignore mission start from {source}: mode={self.current_mode} auto={self.auto_mode}")
            return

        with self.mission_thread_lock:
            if self.mission_thread_active or self.mission_running:
                self.get_logger().warn(f"Ignore mission start from {source}: mission already active")
                return
            self.mission_thread_active = True

        def runner():
            try:
                self._fly_mission()
            finally:
                with self.mission_thread_lock:
                    self.mission_thread_active = False

        self.mission_thread = threading.Thread(target=runner, daemon=True)
        self.mission_thread.start()

    def _takeoff(self):
        if not self.ready:
            self._web_warn("Not ready!")
            return
        if self.airborne or self._airborne_by_lidar():
            self.airborne = True
            self._web_warn(f"Already airborne by lidar={self._lidar_info()} -> skip takeoff")
            return

        if not self.gps_ok:
            self._web_warn("Waiting GPS...")
            t_start = time.time()
            while not self.gps_ok and time.time() - t_start < 30.0:
                time.sleep(0.5)
            if not self.gps_ok:
                self._web_warn("GPS timeout!")
                return

        if not self._set_mode(MODE_GUIDED):
            self._web_warn("Failed GUIDED!")
            return

        time.sleep(0.5)

        if not self.armed:
            if not self._arm():
                self._web_warn("Arm failed!")
                return
            time.sleep(1.0)

        self._mission_state("TAKEOFF")
        self._web_log(f"Taking off to {self.takeoff_alt}m...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0,
            float("nan"),
            self.current_lat if self.current_lat is not None else 0,
            self.current_lon if self.current_lon is not None else 0,
            self.takeoff_alt,
        )

        t_start = time.time()
        while time.time() - t_start < 20.0:
            if self.lidar_height is not None:
                self._web_progress(f"TAKEOFF: lidar={self.lidar_height:.2f}m / {self.takeoff_alt}m", 0.5)
                if self.lidar_height >= self.takeoff_alt - 0.3:
                    self._web_log(f"Takeoff complete! lidar={self.lidar_height:.2f}m")
                    break
            else:
                self._web_progress("TAKEOFF: no lidar, waiting...", 0.5)
            time.sleep(0.5)
        else:
            self._web_warn(f"Takeoff timeout! lidar={self._lidar_info()}")

        self._mission_state("HOVER_SAVE_HOME")
        self._web_log(f"Hovering {self.stabilize_time}s to stabilize and save hover-home...")
        time.sleep(self.stabilize_time)
        self._save_current_position_as_home("after takeoff hover")
        self.airborne = True

    def _fly_mission(self):
        if self.mission_running:
            return
        if not self.waypoints:
            self.get_logger().warn("No waypoints!")
            return
        if self.current_mode != "GUIDED" or not self.auto_mode:
            self.get_logger().warn(f"Mission rejected: mode={self.current_mode} auto={self.auto_mode}")
            return

        self._mark_airborne_if_needed("fly_mission")
        if not self._airborne_ok():
            self.get_logger().warn("Mission rejected: vehicle is not airborne by state/lidar")
            return

        self.mission_running = True
        self.mission_stage = "WP"
        self.aruco_interrupt_enabled = False
        self.landing_guidance_active = False
        self.aruco_seen = False
        self.aruco_latched = False
        self.aruco_abort_done = False
        self.landing_locked_yaw_deg = None
        self.person_hold_active = False
        self._publish_landing_enable(False)
        self._publish_detection_hold_active(False)

        total = len(self.waypoints)
        score = 0
        self.current_mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._mission_state("WP_MISSION")
        self._web_log(
            f"Mission: {total} WPs | alt={self.cruise_alt}m | tol={self.waypoint_tolerance}m | "
            f"mission_id={self.current_mission_id}"
        )

        try:
            for i, wp in enumerate(list(self.waypoints)):
                if not self.auto_mode or self.current_mode != "GUIDED":
                    self._web_warn("Mission aborted by mode change")
                    return
                if not self.armed:
                    self._web_warn("Disarmed! Abort mission")
                    return
                if self.landing_guidance_active or self.landing:
                    self._web_warn("Mission stopped by landing")
                    return

                target_lat = float(wp["lat"])
                target_lon = float(wp["lon"])
                self._mission_state(f"GOTO_WP{i + 1}")
                self._web_log(f"Đang bay tới WP{i + 1}/{total}: lat={target_lat:.7f}, lon={target_lon:.7f}")

                # Yaw theo WP sẽ được cập nhật bên trong vòng lặp WP.
                # Khi thoát WP sang RETURN_HOME/SEARCH_ARUCO/LANDING thì không còn gọi yaw-to-WP nữa.
                reached = False
                self.last_wp_yaw_t = 0.0
                t_start = time.time()

                while time.time() - t_start < self.waypoint_timeout:
                    if not self.auto_mode or self.current_mode != "GUIDED":
                        self.get_logger().warn("Aborted by mode change!")
                        return
                    if not self.armed:
                        self.get_logger().warn("Disarmed! Abort!")
                        return
                    if self.landing_guidance_active or self.landing:
                        self.get_logger().warn("Aborted by landing guidance!")
                        return

                    # Giai đoạn WP: KHÔNG cho ArUco abort mission nữa.
                    # ArUco chỉ được dùng sau khi bay hết WP và đang quay về/search home.

                    # Nếu AL_node báo candidate, tạm phanh để AL_node verify nghiêm ngặt.
                    if self.person_hold_active:
                        self._send_local_velocity(0.0, 0.0, 0.0)
                        self._send_body_velocity(0.0, 0.0, 0.0)
                        time.sleep(0.1)
                        continue

                    # Chỉ tính yaw trong vòng lặp bay WP, không tính yaw ở return/search/landing.
                    now_yaw = time.time()
                    if (
                        self.enable_yaw_to_wp
                        and now_yaw - self.last_wp_yaw_t >= self.wp_yaw_update_period_s
                        and self.current_lat is not None
                        and self.current_lon is not None
                    ):
                        yaw = bearing(self.current_lat, self.current_lon, target_lat, target_lon)
                        self._set_yaw(yaw)
                        self.last_wp_yaw_t = now_yaw

                    self._send_position(target_lat, target_lon, self.cruise_alt)

                    if self.current_lat is not None and self.current_lon is not None:
                        dist = haversine(target_lat, target_lon, self.current_lat, self.current_lon)
                        self._web_progress(
                            f"Đang bay tới WP{i + 1}: dist={dist:.1f}m | "
                            f"current lat={self.current_lat:.7f}, lon={self.current_lon:.7f} | "
                            f"lidar={self._lidar_info()}"
                        )
                        if dist <= self.waypoint_tolerance:
                            reached_lat = float(self.current_lat)
                            reached_lon = float(self.current_lon)
                            reached_alt = float(self.current_alt) if self.current_alt is not None else 0.0

                            wp_error_m = self._log_wp_accuracy(
                                wp_index=i + 1,
                                target_lat=target_lat,
                                target_lon=target_lon,
                                target_alt=self.cruise_alt,
                                reached_lat=reached_lat,
                                reached_lon=reached_lon,
                                reached_alt=reached_alt,
                            )

                            self._mission_state(f"REACHED_WP{i + 1}")
                            self._web_log(
                                f"Đã tới WP{i + 1}: dist={dist:.1f}m | "
                                f"target=({target_lat:.7f},{target_lon:.7f}) | "
                                f"reached=({reached_lat:.7f},{reached_lon:.7f}) | "
                                f"saved_error={wp_error_m:.2f}m"
                            )
                            score += 1
                            reached = True
                            break

                    time.sleep(0.2)

                if not reached:
                    self._mission_state(f"WP{i + 1}_TIMEOUT")
                    self._web_warn(f"WP{i + 1} timeout!")

                if reached and self.hover_time > 0:
                    self._web_log(f"Hover at WP{i + 1}: {self.hover_time}s")
                    self._send_local_velocity(0.0, 0.0, 0.0)
                    self._send_body_velocity(0.0, 0.0, 0.0)
                    time.sleep(self.hover_time)

            self._mission_state("MISSION_DONE")
            self._web_log(f"Mission done: {score}/{total}")
            self.mission_running = False

            # Bay hết WP -> quay về hover-home. Trên đường về mới cho ArUco cắt ngang.
            self._return_home_then_search_aruco()

        finally:
            self.mission_running = False
            if not self.landing_guidance_active and not self.landing:
                self.aruco_interrupt_enabled = False
                if self.mission_stage not in {"SEARCH_ARUCO"}:
                    self.mission_stage = "IDLE"

    def _fly_to_gps_with_aruco_allowed(
        self,
        lat: float,
        lon: float,
        alt: float,
        timeout_s: float,
        label: str,
        speed_mps: Optional[float] = None,
        reach_tolerance_m: Optional[float] = None,
    ) -> bool:
        """
        Bay tới GPS target nhưng cho ArUco cắt ngang.

        RETURN_HOME:
        - Xa home: gửi position như cũ.
        - Gần home <= return_slow_enter_radius_m: dùng velocity chậm.
        - Rất gần home <= search_aruco_enter_radius_m: dừng return để vào SEARCH_ARUCO.

        SEARCH_ARUCO:
        - Luôn dùng velocity chậm tới từng điểm search, tránh lướt qua marker.
        """
        t_start = time.time()
        reach_tol = self.return_home_tolerance_m if reach_tolerance_m is None else float(reach_tolerance_m)

        while time.time() - t_start < timeout_s:
            if not self.auto_mode or self.current_mode != "GUIDED":
                self.get_logger().warn(f"{label} aborted by mode change")
                return False
            if not self.armed:
                self.get_logger().warn(f"{label} aborted: disarmed")
                return False
            if self.landing_guidance_active or self.landing:
                self.get_logger().warn(f"{label} stopped by ArUco landing")
                return True

            # Trong RETURN_HOME/SEARCH_ARUCO, ArUco được phép latch.
            if self._aruco_recent():
                self._abort_wp_and_start_landing(f"ArUco recent during {label}")
                return True

            dist = None
            if self.current_lat is not None and self.current_lon is not None:
                dist = haversine(lat, lon, self.current_lat, self.current_lon)

            # Về gần home thì không cố bay thẳng qua tâm nữa; chuyển qua search pattern.
            if label == "RETURN_HOME" and dist is not None and dist <= self.search_aruco_enter_radius_m:
                self._web_warn(
                    f"RETURN_HOME near home dist={dist:.1f}m <= {self.search_aruco_enter_radius_m:.1f}m "
                    "-> switch SEARCH_ARUCO to avoid overshoot"
                )
                self._send_local_velocity(0.0, 0.0, 0.0)
                self._send_body_velocity(0.0, 0.0, 0.0)
                return True

            if dist is not None and dist <= reach_tol:
                self._web_log(f"{label} reached dist={dist:.1f}m | lidar={self._lidar_info()}")
                self._send_local_velocity(0.0, 0.0, 0.0)
                self._send_body_velocity(0.0, 0.0, 0.0)
                return True

            # Điều khiển bay:
            if label == "RETURN_HOME":
                if dist is not None and dist <= self.return_slow_enter_radius_m:
                    d = self._send_velocity_to_gps_target(lat, lon, self.return_slow_speed_mps)
                    self._web_progress(
                        f"RETURN_HOME_SLOW dist={d:.1f}m speed={self.return_slow_speed_mps:.2f}m/s | "
                        f"lidar={self._lidar_info()}"
                    )
                else:
                    self._send_position(lat, lon, alt)
                    if dist is not None:
                        self._web_progress(f"RETURN_HOME_FAST dist={dist:.1f}m | lidar={self._lidar_info()}")
            else:
                use_speed = self.search_aruco_speed_mps if speed_mps is None else float(speed_mps)
                d = self._send_velocity_to_gps_target(lat, lon, use_speed)
                self._web_progress(
                    f"{label} velocity_to_point dist={d:.1f}m speed={use_speed:.2f}m/s | "
                    f"lidar={self._lidar_info()}"
                )

            time.sleep(0.2)

        self._web_warn(f"{label} timeout")
        return False

    def _return_home_then_search_aruco(self):
        if not self.home_valid or self.home_lat is None or self.home_lon is None:
            self._web_warn("Mission done but home invalid -> hover, cannot return/search ArUco")
            self._send_local_velocity(0.0, 0.0, 0.0)
            self._send_body_velocity(0.0, 0.0, 0.0)
            return

        # Xóa ArUco cũ đã thấy trong lúc bay WP để không latch nhầm stale marker.
        self.aruco_seen = False
        self.aruco_latched = False
        self.aruco_abort_done = False
        self.last_aruco_seen_t = 0.0
        self.last_aruco_id = -1

        self.aruco_interrupt_enabled = True
        self.mission_stage = "RETURN_HOME"
        self._publish_landing_enable(False)

        self._mission_state("RETURN_HOME")
        self._web_log(
            f"Mission done -> RETURN HOME lat={self.home_lat:.7f}, lon={self.home_lon:.7f}, "
            "ArUco interrupt ENABLED only now"
        )

        self._fly_to_gps_with_aruco_allowed(
            self.home_lat,
            self.home_lon,
            self.cruise_alt,
            self.return_home_timeout,
            "RETURN_HOME",
        )

        if self.landing_guidance_active or self.landing:
            return

        # Về home mà chưa thấy ArUco -> search quanh home bán kính 2m.
        self.mission_stage = "SEARCH_ARUCO"
        self._mission_state("SEARCH_ARUCO")
        self._web_warn(f"RETURN_HOME done/no ArUco -> SEARCH_ARUCO radius={self.search_radius_m:.1f}m")

        r = self.search_radius_m
        # Search pattern giống ý tưởng code mẫu: quét quanh home trong phạm vi 2m.
        # Không chỉ đứng ở tâm, vì nếu đã lướt qua ArUco thì marker có thể nằm lệch khỏi camera.
        pattern = [
            (0.0, 0.0),
            (r, 0.0),
            (r, r),
            (0.0, r),
            (-r, r),
            (-r, 0.0),
            (-r, -r),
            (0.0, -r),
            (r, -r),
            (r, 0.0),
            (0.0, 0.0),
        ]

        t_search = time.time()
        loop_idx = 0
        while time.time() - t_search < self.search_total_timeout:
            loop_idx += 1
            self._web_warn(f"SEARCH_ARUCO loop={loop_idx} radius={r:.1f}m speed={self.search_aruco_speed_mps:.2f}m/s")
            for north, east in pattern:
                if self.landing_guidance_active or self.landing:
                    return
                if not self.auto_mode or self.current_mode != "GUIDED" or not self.armed:
                    return

                lat, lon = offset_gps(self.home_lat, self.home_lon, north, east)
                self._web_log(f"SEARCH_ARUCO point N={north:.1f} E={east:.1f}")

                self._fly_to_gps_with_aruco_allowed(
                    lat,
                    lon,
                    self.cruise_alt,
                    self.search_waypoint_timeout_s,
                    "SEARCH_ARUCO",
                    speed_mps=self.search_aruco_speed_mps,
                    reach_tolerance_m=self.search_point_tolerance_m,
                )

                self._send_local_velocity(0.0, 0.0, 0.0)
                self._send_body_velocity(0.0, 0.0, 0.0)
                if self.landing_guidance_active or self.landing:
                    return

                if self._hold_and_watch_aruco(self.search_point_hold_s, "SEARCH_ARUCO"):
                    return

        self._web_warn("SEARCH_ARUCO timeout -> hover at home, landing guidance not enabled")
        self.aruco_interrupt_enabled = False
        self.mission_stage = "IDLE"
        self._publish_landing_enable(False)
        self._send_local_velocity(0.0, 0.0, 0.0)
        self._send_body_velocity(0.0, 0.0, 0.0)

    def _land(self):
        if not self.airborne and not self._airborne_by_lidar():
            self._web_warn("Already on ground!")
            return
        if self.landing:
            self._web_warn("Already landing!")
            return

        self.landing = True
        self.mission_running = False
        self.mission_stage = "LAND"
        self.aruco_interrupt_enabled = False
        self.person_hold_active = False
        self.landing_guidance_active = False
        self._publish_landing_enable(False)
        self._publish_detection_hold_active(False)
        self._mission_state("LAND")
        self._web_warn("Landing...")
        self._set_mode(MODE_LAND)

        t_start = time.time()
        while time.time() - t_start < 30.0:
            if self.lidar_height is not None:
                self.get_logger().info(f"  lidar={self.lidar_height:.2f}m | ground={self.lidar_ground_level:.2f}m")
                if self._is_landed():
                    self._web_log(f"Ground detected! {self.lidar_height:.2f}m")
                    time.sleep(1.0)
                    break
            if not self.armed:
                self._web_log("Disarmed -> Landed!")
                break
            time.sleep(0.3)
        else:
            self._web_warn("Land timeout!")

        self._on_landed()


def main(args=None):
    rclpy.init(args=args)
    node = DroneControl()
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
