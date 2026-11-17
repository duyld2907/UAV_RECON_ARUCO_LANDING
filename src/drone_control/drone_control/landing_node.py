#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
landing_node.py

Landing node PID velocity for ArUco precision landing.
Bản này thêm image_err_x_sign / image_err_y_sign để đảo chiều kéo khi camera/gimbal lắp ngược.

Quy ước body_ned:
  +vx = tiến về trước drone
  +vy = sang phải drone
  +vz_down = đi xuống

Mapping mặc định sau khi đã áp dụng sign:
  rel_fwd_m   = -Z * err_y / fy  -> vx_body
  rel_right_m =  Z * err_x / fx  -> vy_body

Nếu marker lệch phải nhưng drone kéo trái:
  image_err_x_sign = -1.0
Nếu tiến/lùi cũng ngược:
  image_err_y_sign = -1.0
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


# ---------------- helpers ----------------
def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def blend(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return (1.0 - t) * float(a) + t * float(b)


def norm2(x: float, y: float) -> float:
    return math.hypot(float(x), float(y))


def vector_limit(x: float, y: float, max_norm: float) -> Tuple[float, float]:
    max_norm = max(0.0, float(max_norm))
    n = norm2(x, y)
    if n > max_norm and n > 1e-9:
        s = max_norm / n
        return float(x) * s, float(y) * s
    return float(x), float(y)


def mean(vals) -> float:
    vals = list(vals)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def stddev(vals) -> float:
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((float(v) - m) ** 2 for v in vals) / len(vals))


def sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@dataclass
class Detection:
    marker_id: int
    err_x: float
    err_y: float
    center_x: float
    center_y: float
    size_px: float
    z_m: float
    stamp: float
    err_x_rate: float = 0.0
    err_y_rate: float = 0.0
    detected: bool = True
    predict_only: bool = False
    raw_err_x: float = 0.0
    raw_err_y: float = 0.0
    filtered_err_x: float = 0.0
    filtered_err_y: float = 0.0
    kf_age_s: float = 0.0
    kf_pos_std_px: float = 999.0
    kf_nis: float = 0.0
    kf_samples: int = 0
    meas_noise_px: float = 0.0
    stable_samples: int = 1
    stable_std_px: float = 0.0


@dataclass
class AxisPIDState:
    integ: float = 0.0
    last_t: Optional[float] = None

    def reset(self) -> None:
        self.integ = 0.0
        self.last_t = None


class AxisPID:
    def __init__(self) -> None:
        self.state: Dict[int, AxisPIDState] = {}

    def reset(self, key: Optional[int] = None) -> None:
        if key is None:
            self.state.clear()
        else:
            self.state.setdefault(key, AxisPIDState()).reset()

    def compute(
        self,
        key: int,
        err_m: float,
        err_rate_mps: float,
        now: float,
        kp: float,
        ki: float,
        kd: float,
        i_limit: float,
        i_enable: bool,
        out_limit: float,
    ) -> Tuple[float, Dict[str, float]]:
        st = self.state.setdefault(key, AxisPIDState())
        if st.last_t is None:
            dt = 0.0
        else:
            dt = clamp(float(now) - float(st.last_t), 0.001, 0.12)
        st.last_t = float(now)

        p_term = float(kp) * float(err_m)
        d_term = float(kd) * float(err_rate_mps)

        if i_enable and dt > 0.0 and abs(float(ki)) > 1e-9:
            st.integ += float(err_m) * dt
            st.integ = clamp(st.integ, -abs(i_limit), abs(i_limit))
        elif not i_enable:
            st.integ *= 0.92
            if abs(st.integ) < 1e-4:
                st.integ = 0.0

        i_term = float(ki) * st.integ
        raw = p_term + i_term + d_term
        out = clamp(raw, -abs(out_limit), abs(out_limit))

        if i_enable and abs(raw) > abs(out_limit) and abs(ki) > 1e-9:
            if raw * st.integ > 0.0:
                st.integ *= 0.80
                i_term = float(ki) * st.integ
                raw = p_term + i_term + d_term
                out = clamp(raw, -abs(out_limit), abs(out_limit))

        return float(out), {
            "pid_dt": float(dt),
            "pid_err_m": float(err_m),
            "pid_rate_mps": float(err_rate_mps),
            "pid_p": float(p_term),
            "pid_i": float(i_term),
            "pid_d": float(d_term),
            "pid_integral": float(st.integ),
            "pid_raw": float(raw),
            "pid_out": float(out),
            "pid_i_enable": 1.0 if i_enable else 0.0,
        }


class LandingNodePID(Node):
    def __init__(self):
        super().__init__("landing_node")
        self.declare_params()
        self.load_params()

        self.enabled = False
        self.phase = "IDLE"
        self.seq = 0

        self.latest: Dict[int, Detection] = {}
        self.buffers: Dict[int, Deque[Detection]] = {
            self.large_id: deque(),
            self.small_id: deque(),
        }
        self.last_seen_t: Dict[int, Optional[float]] = {self.large_id: None, self.small_id: None}
        self.last_visible_err_x: Dict[int, Optional[float]] = {self.large_id: None, self.small_id: None}
        self.last_visible_err_y: Dict[int, Optional[float]] = {self.large_id: None, self.small_id: None}
        self.last_real_det_t: Dict[int, Optional[float]] = {self.large_id: None, self.small_id: None}

        self.pid_fwd = AxisPID()
        self.pid_right = AxisPID()

        self.lidar_m: Optional[float] = None
        self.lidar_t: Optional[float] = None
        self.baro_m: Optional[float] = None
        self.baro_t: Optional[float] = None

        self.active_mid: Optional[int] = None
        self.last_marker_id: Optional[int] = None
        self.small_locked = False
        self.small_lock_t: Optional[float] = None
        self.final_since: Optional[float] = None

        self.last_cmd_vx = 0.0
        self.last_cmd_vy = 0.0
        self.last_cmd_t: Optional[float] = None

        self.cmd_pub = self.create_publisher(String, self.cmd_topic, 10)
        self.create_subscription(Bool, self.enable_topic, self.cb_enable, 10)
        self.create_subscription(String, self.det_topic, self.cb_detections, 10)
        self.create_subscription(Float32, self.lidar_topic, self.cb_lidar, 10)
        self.create_subscription(Float32, self.baro_topic, self.cb_baro, 10)
        self.create_timer(1.0 / max(5.0, self.control_hz), self.control_loop)

        self.get_logger().info(
            "PID landing node ready | filtered ArUco error -> PID velocity | "
            f"large_box=±({self.large_box_x:.0f},{self.large_box_y:.0f})px "
            f"small_box=±({self.small_box_x:.0f},{self.small_box_y:.0f})px "
            f"precision=±({self.small_precision_box_x:.0f},{self.small_precision_box_y:.0f})px | "
            f"image_err_sign=({self.image_err_x_sign:+.0f},{self.image_err_y_sign:+.0f})"
        )

    # ---------------- params ----------------
    def declare_params(self):
        p = self.declare_parameter

        # Topics / IDs / camera
        p("detections_topic", "/vision/aruco_detections_json")
        p("landing_enable_topic", "/landing_enable")
        p("lidar_topic", "/lidar_height_m")
        p("baro_topic", "/relative_alt_m")
        p("cmd_topic", "/landing_cmd")
        p("command_frame", "body_ned")
        p("large_id", 150)
        p("small_id", 40)
        p("camera_fx_px", 585.2398581)
        p("camera_fy_px", 586.40005266)
        p("camera_cx_px", 331.58390834)
        p("camera_cy_px", 242.95158027)

        # Dấu quy đổi sai số ảnh -> chiều body drone.
        p("image_err_x_sign", -1.0)
        p("image_err_y_sign", -1.0)

        p("z_min_m", 0.15)
        p("z_max_m", 8.0)
        p("alt_timeout_s", 1.0)
        p("detection_timeout_s", 0.38)
        p("predict_only_timeout_s", 0.16)
        p("gate_history_s", 0.18)
        p("control_loop_hz", 30.0)

        # Marker selection
        p("small_takeover_alt_m", 1.70)
        p("small_takeover_min_size_px", 22.0)
        p("small_takeover_gate_x_px", 170.0)
        p("small_takeover_gate_y_px", 190.0)
        p("small_takeover_min_kf_samples", 2)
        p("small_handoff_enable", True)
        p("small_handoff_alt_m", 2.35)
        p("small_handoff_min_size_px", 8.0)
        p("small_handoff_gate_x_px", 260.0)
        p("small_handoff_gate_y_px", 260.0)
        p("small_handoff_max_rate_px_s", 850.0)
        p("small_handoff_descend_vz", 0.035)
        p("small_lock_min_s", 1.0)
        p("small_return_to_large_enable", False)
        p("small_return_to_large_lost_s", 1.20)

        # Boxes / precision
        p("large_box_half_x_px", 220.0)
        p("large_box_half_y_px", 165.0)
        p("small_box_half_x_px", 140.0)
        p("small_box_half_y_px", 110.0)
        p("small_precision_enable", True)
        p("small_precision_start_z_m", 1.25)
        p("small_precision_full_z_m", 0.82)
        p("small_precision_box_half_x_px", 86.0)
        p("small_precision_box_half_y_px", 78.0)
        p("track_keep_margin_x_px", 45.0)
        p("track_keep_margin_y_px", 55.0)

        # PID gains
        p("large_pid_kp_fwd", 0.82)
        p("large_pid_kd_fwd", 0.20)
        p("large_pid_ki_fwd", 0.003)
        p("large_pid_kp_right", 0.62)
        p("large_pid_kd_right", 0.16)
        p("large_pid_ki_right", 0.002)
        p("small_pid_kp_fwd", 0.98)
        p("small_pid_kd_fwd", 0.18)
        p("small_pid_ki_fwd", 0.002)
        p("small_pid_kp_right", 0.72)
        p("small_pid_kd_right", 0.16)
        p("small_pid_ki_right", 0.001)
        p("pid_integral_limit_m_s", 0.18)
        p("pid_i_enable", True)
        p("pid_i_disable_err_px", 160.0)
        p("pid_i_disable_near_ground_m", 0.65)
        p("pid_i_reset_on_marker_switch", True)

        # Speed / acceleration shaping
        p("large_speed_max_mps", 0.52)
        p("small_speed_max_mps", 0.44)
        p("large_lateral_max_mps", 0.30)
        p("small_lateral_max_mps", 0.26)
        p("large_accel_max_mps2", 0.42)
        p("small_accel_max_mps2", 0.50)
        p("brake_accel_max_mps2", 0.42)
        p("near_center_speed_scale", 0.34)
        p("near_center_accel_scale", 0.42)
        p("near_center_x_px", 55.0)
        p("near_center_y_px", 55.0)
        p("cmd_alpha_far", 0.46)
        p("cmd_alpha_near", 0.24)
        p("min_cmd_deadband_mps", 0.004)

        # BIG/ID150 riêng
        p("large_control_deadband_x_px", 24.0)
        p("large_control_deadband_y_px", 20.0)
        p("large_rate_deadband_px_s", 18.0)
        p("large_rate_clip_px_s", 260.0)
        p("large_near_center_x_px", 95.0)
        p("large_near_center_y_px", 85.0)
        p("large_near_center_speed_scale", 0.24)
        p("large_near_center_accel_scale", 0.32)
        p("large_cmd_alpha_far", 0.34)
        p("large_cmd_alpha_near", 0.15)

        # Reverse / overshoot guard
        p("forward_reverse_guard_enable", True)
        p("reverse_brake_y_px", 115.0)
        p("reverse_brake_max_mps", 0.018)
        p("reverse_allow_y_px_large", 185.0)
        p("reverse_allow_y_px_small", 190.0)
        p("reverse_allow_max_mps_large", 0.13)
        p("reverse_allow_max_mps_small", 0.045)
        p("forward_sustain_when_blocked_mps", 0.010)

        # Small marker Y priority
        p("small_y_priority_enable", True)
        p("small_y_priority_y_mid_px", 60.0)
        p("small_y_priority_y_high_px", 90.0)
        p("small_y_priority_x_scale_mid_y", 0.34)
        p("small_y_priority_x_scale_high_y", 0.22)
        p("small_y_priority_x_rescue_px", 125.0)
        p("small_y_priority_x_rescue_scale", 0.72)
        p("small_y_priority_x_rescue_max_mps", 0.12)
        p("small_y_priority_y_boost_px", 120.0)
        p("small_y_priority_y_boost_scale", 1.05)
        p("small_y_priority_y_boost_max_mps", 0.36)

        # Descent / land gates
        p("large_descend_vz", 0.115)
        p("large_outside_descend_vz", 0.050)
        p("small_descend_vz", 0.085)
        p("small_mid_descend_vz", 0.060)
        p("small_precision_descend_vz", 0.040)
        p("small_outside_descend_vz", 0.0)
        p("small_approach_descend_enable", True)
        p("small_approach_descend_min_z_m", 0.62)
        p("small_approach_descend_vz", 0.045)
        p("small_approach_descend_gate_x_px", 150.0)
        p("small_approach_descend_gate_y_px", 150.0)
        p("small_approach_descend_max_rate_px_s", 680.0)
        p("small_approach_descend_min_samples", 2)
        p("small_descend_gate_x_px", 115.0)
        p("small_descend_gate_y_px", 115.0)
        p("small_precision_descend_gate_x_px", 78.0)
        p("small_precision_descend_gate_y_px", 88.0)
        p("small_descend_min_kf_samples", 2)
        p("small_precision_descend_min_kf_samples", 2)
        p("allow_large_descend_below_small_alt", True)
        p("final_land_alt_m", 0.56)
        p("final_hold_s", 0.08)
        p("land_only_on_small", True)
        p("final_land_gate_x_px", 96.0)
        p("final_land_gate_y_px", 126.0)
        p("final_land_max_rate_px_s", 820.0)
        p("final_land_min_samples", 1)
        p("final_land_allow_approach_gate", True)

        # Blind / lost marker
        p("blind_forward_on_lost_enable", True)
        p("blind_forward_speed_mps", 0.30)
        p("blind_forward_large_speed_mps", 0.30)
        p("blind_forward_small_speed_mps", 0.25)
        p("blind_forward_large_max_s", 2.60)
        p("blind_forward_small_max_s", 2.00)
        p("blind_forward_min_alt_m", 0.35)
        p("blind_forward_delay_s", 0.00)
        p("blind_forward_accel_mps2", 0.45)
        p("blind_forward_alpha", 0.48)
        p("blind_forward_require_last_y_not_behind", False)
        p("blind_forward_last_y_max_px", 9999.0)
        p("lost_hold_last_cmd_enable", True)
        p("lost_hold_last_cmd_s", 0.36)
        p("lost_hold_min_cmd_mps", 0.025)
        p("large_lost_hold_max_mps", 0.30)
        p("small_lost_hold_max_mps", 0.095)
        p("small_lost_hold_y_large_px", 120.0)
        p("small_lost_hold_y_large_s", 0.25)
        p("small_lost_hold_y_large_scale", 0.30)
        p("lost_brake_s", 0.30)

    def load_params(self):
        g = lambda name: self.get_parameter(name).value

        self.det_topic = str(g("detections_topic"))
        self.enable_topic = str(g("landing_enable_topic"))
        self.lidar_topic = str(g("lidar_topic"))
        self.baro_topic = str(g("baro_topic"))
        self.cmd_topic = str(g("cmd_topic"))
        self.command_frame = str(g("command_frame"))
        self.large_id = int(g("large_id"))
        self.small_id = int(g("small_id"))
        self.fx = float(g("camera_fx_px"))
        self.fy = float(g("camera_fy_px"))
        self.cx = float(g("camera_cx_px"))
        self.cy = float(g("camera_cy_px"))
        self.image_err_x_sign = 1.0 if float(g("image_err_x_sign")) >= 0.0 else -1.0
        self.image_err_y_sign = 1.0 if float(g("image_err_y_sign")) >= 0.0 else -1.0
        self.z_min = float(g("z_min_m"))
        self.z_max = float(g("z_max_m"))
        self.alt_timeout_s = float(g("alt_timeout_s"))
        self.det_timeout_s = float(g("detection_timeout_s"))
        self.predict_timeout_s = float(g("predict_only_timeout_s"))
        self.gate_history_s = float(g("gate_history_s"))
        self.control_hz = float(g("control_loop_hz"))

        self.small_takeover_alt = float(g("small_takeover_alt_m"))
        self.small_takeover_min_size = float(g("small_takeover_min_size_px"))
        self.small_takeover_gate_x = float(g("small_takeover_gate_x_px"))
        self.small_takeover_gate_y = float(g("small_takeover_gate_y_px"))
        self.small_takeover_min_kf_samples = int(g("small_takeover_min_kf_samples"))
        self.small_handoff_enable = bool(g("small_handoff_enable"))
        self.small_handoff_alt = float(g("small_handoff_alt_m"))
        self.small_handoff_min_size = float(g("small_handoff_min_size_px"))
        self.small_handoff_gate_x = float(g("small_handoff_gate_x_px"))
        self.small_handoff_gate_y = float(g("small_handoff_gate_y_px"))
        self.small_handoff_max_rate = float(g("small_handoff_max_rate_px_s"))
        self.small_handoff_descend_vz = float(g("small_handoff_descend_vz"))
        self.small_lock_min_s = float(g("small_lock_min_s"))
        self.small_return_large_enable = bool(g("small_return_to_large_enable"))
        self.small_return_large_lost_s = float(g("small_return_to_large_lost_s"))

        self.large_box_x = float(g("large_box_half_x_px"))
        self.large_box_y = float(g("large_box_half_y_px"))
        self.small_box_x = float(g("small_box_half_x_px"))
        self.small_box_y = float(g("small_box_half_y_px"))
        self.small_precision_enable = bool(g("small_precision_enable"))
        self.small_precision_start_z = float(g("small_precision_start_z_m"))
        self.small_precision_full_z = float(g("small_precision_full_z_m"))
        self.small_precision_box_x = float(g("small_precision_box_half_x_px"))
        self.small_precision_box_y = float(g("small_precision_box_half_y_px"))
        self.track_keep_margin_x = float(g("track_keep_margin_x_px"))
        self.track_keep_margin_y = float(g("track_keep_margin_y_px"))

        self.large_pid_kp_fwd = float(g("large_pid_kp_fwd"))
        self.large_pid_kd_fwd = float(g("large_pid_kd_fwd"))
        self.large_pid_ki_fwd = float(g("large_pid_ki_fwd"))
        self.large_pid_kp_right = float(g("large_pid_kp_right"))
        self.large_pid_kd_right = float(g("large_pid_kd_right"))
        self.large_pid_ki_right = float(g("large_pid_ki_right"))
        self.small_pid_kp_fwd = float(g("small_pid_kp_fwd"))
        self.small_pid_kd_fwd = float(g("small_pid_kd_fwd"))
        self.small_pid_ki_fwd = float(g("small_pid_ki_fwd"))
        self.small_pid_kp_right = float(g("small_pid_kp_right"))
        self.small_pid_kd_right = float(g("small_pid_kd_right"))
        self.small_pid_ki_right = float(g("small_pid_ki_right"))
        self.pid_integral_limit = float(g("pid_integral_limit_m_s"))
        self.pid_i_enable_param = bool(g("pid_i_enable"))
        self.pid_i_disable_err_px = float(g("pid_i_disable_err_px"))
        self.pid_i_disable_near_ground = float(g("pid_i_disable_near_ground_m"))
        self.pid_i_reset_on_marker_switch = bool(g("pid_i_reset_on_marker_switch"))

        self.large_speed_max = float(g("large_speed_max_mps"))
        self.small_speed_max = float(g("small_speed_max_mps"))
        self.large_lateral_max = float(g("large_lateral_max_mps"))
        self.small_lateral_max = float(g("small_lateral_max_mps"))
        self.large_accel_max = float(g("large_accel_max_mps2"))
        self.small_accel_max = float(g("small_accel_max_mps2"))
        self.brake_accel_max = float(g("brake_accel_max_mps2"))
        self.near_center_speed_scale = float(g("near_center_speed_scale"))
        self.near_center_accel_scale = float(g("near_center_accel_scale"))
        self.near_center_x = float(g("near_center_x_px"))
        self.near_center_y = float(g("near_center_y_px"))
        self.cmd_alpha_far = float(g("cmd_alpha_far"))
        self.cmd_alpha_near = float(g("cmd_alpha_near"))
        self.min_cmd_deadband = float(g("min_cmd_deadband_mps"))
        self.large_control_deadband_x = float(g("large_control_deadband_x_px"))
        self.large_control_deadband_y = float(g("large_control_deadband_y_px"))
        self.large_rate_deadband = float(g("large_rate_deadband_px_s"))
        self.large_rate_clip = float(g("large_rate_clip_px_s"))
        self.large_near_center_x = float(g("large_near_center_x_px"))
        self.large_near_center_y = float(g("large_near_center_y_px"))
        self.large_near_center_speed_scale = float(g("large_near_center_speed_scale"))
        self.large_near_center_accel_scale = float(g("large_near_center_accel_scale"))
        self.large_cmd_alpha_far = float(g("large_cmd_alpha_far"))
        self.large_cmd_alpha_near = float(g("large_cmd_alpha_near"))

        self.forward_reverse_guard_enable = bool(g("forward_reverse_guard_enable"))
        self.reverse_brake_y = float(g("reverse_brake_y_px"))
        self.reverse_brake_max = float(g("reverse_brake_max_mps"))
        self.reverse_allow_y_large = float(g("reverse_allow_y_px_large"))
        self.reverse_allow_y_small = float(g("reverse_allow_y_px_small"))
        self.reverse_allow_max_large = float(g("reverse_allow_max_mps_large"))
        self.reverse_allow_max_small = float(g("reverse_allow_max_mps_small"))
        self.forward_sustain_when_blocked = float(g("forward_sustain_when_blocked_mps"))

        self.small_y_priority_enable = bool(g("small_y_priority_enable"))
        self.small_y_priority_y_mid = float(g("small_y_priority_y_mid_px"))
        self.small_y_priority_y_high = float(g("small_y_priority_y_high_px"))
        self.small_y_priority_x_scale_mid = float(g("small_y_priority_x_scale_mid_y"))
        self.small_y_priority_x_scale_high = float(g("small_y_priority_x_scale_high_y"))
        self.small_y_priority_x_rescue = float(g("small_y_priority_x_rescue_px"))
        self.small_y_priority_x_rescue_scale = float(g("small_y_priority_x_rescue_scale"))
        self.small_y_priority_x_rescue_max = float(g("small_y_priority_x_rescue_max_mps"))
        self.small_y_priority_y_boost_px = float(g("small_y_priority_y_boost_px"))
        self.small_y_priority_y_boost_scale = float(g("small_y_priority_y_boost_scale"))
        self.small_y_priority_y_boost_max = float(g("small_y_priority_y_boost_max_mps"))

        self.large_vz = float(g("large_descend_vz"))
        self.large_out_vz = float(g("large_outside_descend_vz"))
        self.small_vz = float(g("small_descend_vz"))
        self.small_mid_vz = float(g("small_mid_descend_vz"))
        self.small_precision_vz = float(g("small_precision_descend_vz"))
        self.small_out_vz = float(g("small_outside_descend_vz"))
        self.small_approach_descend_enable = bool(g("small_approach_descend_enable"))
        self.small_approach_descend_min_z = float(g("small_approach_descend_min_z_m"))
        self.small_approach_descend_vz = float(g("small_approach_descend_vz"))
        self.small_approach_descend_gate_x = float(g("small_approach_descend_gate_x_px"))
        self.small_approach_descend_gate_y = float(g("small_approach_descend_gate_y_px"))
        self.small_approach_descend_max_rate = float(g("small_approach_descend_max_rate_px_s"))
        self.small_approach_descend_min_samples = int(g("small_approach_descend_min_samples"))
        self.small_descend_gate_x = float(g("small_descend_gate_x_px"))
        self.small_descend_gate_y = float(g("small_descend_gate_y_px"))
        self.small_precision_descend_gate_x = float(g("small_precision_descend_gate_x_px"))
        self.small_precision_descend_gate_y = float(g("small_precision_descend_gate_y_px"))
        self.small_descend_min_kf_samples = int(g("small_descend_min_kf_samples"))
        self.small_precision_descend_min_kf_samples = int(g("small_precision_descend_min_kf_samples"))
        self.allow_large_below_small = bool(g("allow_large_descend_below_small_alt"))
        self.final_land_alt = float(g("final_land_alt_m"))
        self.final_hold_s = float(g("final_hold_s"))
        self.land_only_small = bool(g("land_only_on_small"))
        self.final_land_gate_x = float(g("final_land_gate_x_px"))
        self.final_land_gate_y = float(g("final_land_gate_y_px"))
        self.final_land_max_rate = float(g("final_land_max_rate_px_s"))
        self.final_land_min_samples = int(g("final_land_min_samples"))
        self.final_land_allow_approach_gate = bool(g("final_land_allow_approach_gate"))

        self.blind_forward_on_lost_enable = bool(g("blind_forward_on_lost_enable"))
        self.blind_forward_speed = float(g("blind_forward_speed_mps"))
        self.blind_forward_large_speed = float(g("blind_forward_large_speed_mps"))
        self.blind_forward_small_speed = float(g("blind_forward_small_speed_mps"))
        self.blind_forward_large_max_s = float(g("blind_forward_large_max_s"))
        self.blind_forward_small_max_s = float(g("blind_forward_small_max_s"))
        self.blind_forward_min_alt = float(g("blind_forward_min_alt_m"))
        self.blind_forward_delay_s = float(g("blind_forward_delay_s"))
        self.blind_forward_accel = float(g("blind_forward_accel_mps2"))
        self.blind_forward_alpha = float(g("blind_forward_alpha"))
        self.blind_forward_require_last_y_not_behind = bool(g("blind_forward_require_last_y_not_behind"))
        self.blind_forward_last_y_max = float(g("blind_forward_last_y_max_px"))
        self.lost_hold_last_cmd_enable = bool(g("lost_hold_last_cmd_enable"))
        self.lost_hold_last_cmd_s = float(g("lost_hold_last_cmd_s"))
        self.lost_hold_min_cmd = float(g("lost_hold_min_cmd_mps"))
        self.large_lost_hold_max = float(g("large_lost_hold_max_mps"))
        self.small_lost_hold_max = float(g("small_lost_hold_max_mps"))
        self.small_lost_hold_y_large_px = float(g("small_lost_hold_y_large_px"))
        self.small_lost_hold_y_large_s = float(g("small_lost_hold_y_large_s"))
        self.small_lost_hold_y_large_scale = float(g("small_lost_hold_y_large_scale"))
        self.lost_brake_s = float(g("lost_brake_s"))

    # ---------------- callbacks ----------------
    def cb_enable(self, msg: Bool):
        self.enabled = bool(msg.data)
        self.phase = "WAIT_MARKER" if self.enabled else "IDLE"
        self.reset_control()
        self.publish_cmd("hold", 0.0, 0.0, 0.0, "enable" if self.enabled else "disabled")

    def reset_control(self):
        self.latest.clear()
        for b in self.buffers.values():
            b.clear()
        self.pid_fwd.reset()
        self.pid_right.reset()
        self.last_seen_t = {self.large_id: None, self.small_id: None}
        self.last_real_det_t = {self.large_id: None, self.small_id: None}
        self.last_visible_err_x = {self.large_id: None, self.small_id: None}
        self.last_visible_err_y = {self.large_id: None, self.small_id: None}
        self.active_mid = None
        self.last_marker_id = None
        self.small_locked = False
        self.small_lock_t = None
        self.final_since = None
        self.last_cmd_t = None
        self.last_cmd_vx = 0.0
        self.last_cmd_vy = 0.0

    def cb_lidar(self, msg: Float32):
        v = float(msg.data)
        if finite(v):
            self.lidar_m, self.lidar_t = v, time.time()

    def cb_baro(self, msg: Float32):
        v = float(msg.data)
        if finite(v):
            self.baro_m, self.baro_t = v, time.time()

    def _float_field(self, raw: dict, keys, default=0.0) -> float:
        for k in keys:
            if k in raw:
                v = raw.get(k)
                if v is not None and finite(v):
                    return float(v)
        return float(default)

    def cb_detections(self, msg: String):
        now = time.time()
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        dets = data.get("detections", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        if not isinstance(dets, list):
            return

        best: Dict[int, Detection] = {}
        for raw in dets:
            if not isinstance(raw, dict):
                continue
            try:
                mid = int(raw.get("marker_id", raw.get("id", -1)))
            except Exception:
                continue
            if mid not in (self.large_id, self.small_id):
                continue

            # Ưu tiên filtered -> err -> raw từ ArUco Kalman.
            ex = self._float_field(raw, ["filtered_err_x", "err_x", "raw_err_x"], 0.0)
            ey = self._float_field(raw, ["filtered_err_y", "err_y", "raw_err_y"], 0.0)
            raw_ex = self._float_field(raw, ["raw_err_x", "err_x", "filtered_err_x"], ex)
            raw_ey = self._float_field(raw, ["raw_err_y", "err_y", "filtered_err_y"], ey)
            size = self._float_field(raw, ["size_px", "avg_side_px", "side_px"], 0.0)
            z = self._float_field(raw, ["z_m", "tvec_z", "distance_m"], 0.0)
            ex_rate = self._float_field(raw, ["err_x_rate_px_s", "err_x_rate", "raw_err_x_rate"], 0.0)
            ey_rate = self._float_field(raw, ["err_y_rate_px_s", "err_y_rate", "raw_err_y_rate"], 0.0)

            # ÁP DỤNG ĐẢO DẤU TẠI LANDING NODE.
            # Không sửa JSON gốc từ ArUco, chỉ đổi lỗi dùng cho điều khiển.
            ex *= self.image_err_x_sign
            raw_ex *= self.image_err_x_sign
            ex_rate *= self.image_err_x_sign
            ey *= self.image_err_y_sign
            raw_ey *= self.image_err_y_sign
            ey_rate *= self.image_err_y_sign

            cx = self.cx + ex
            cy = self.cy + ey
            detected = bool(raw.get("detected", True))
            predict_only = bool(raw.get("predict_only", False))
            kf_age = self._float_field(raw, ["kf_age_s"], 0.0)
            kf_std = self._float_field(raw, ["kf_pos_std_px"], 999.0)
            kf_nis = self._float_field(raw, ["kf_nis"], 0.0)
            try:
                kf_samples = int(raw.get("kf_samples", 0))
            except Exception:
                kf_samples = 0
            meas_noise = self._float_field(raw, ["meas_noise_px"], 0.0)

            if not (finite(ex) and finite(ey)):
                continue
            if predict_only and kf_age > self.predict_timeout_s:
                continue

            d = Detection(
                marker_id=mid,
                err_x=ex,
                err_y=ey,
                center_x=cx,
                center_y=cy,
                size_px=size,
                z_m=z,
                stamp=now,
                err_x_rate=ex_rate,
                err_y_rate=ey_rate,
                detected=detected,
                predict_only=predict_only,
                raw_err_x=raw_ex,
                raw_err_y=raw_ey,
                filtered_err_x=ex,
                filtered_err_y=ey,
                kf_age_s=kf_age,
                kf_pos_std_px=kf_std,
                kf_nis=kf_nis,
                kf_samples=kf_samples,
                meas_noise_px=meas_noise,
            )

            penalty = 2000.0 if predict_only or not detected else 0.0
            score = penalty + abs(ex) + abs(ey) - 0.22 * size
            old = best.get(mid)
            if old is None:
                best[mid] = d
            else:
                old_penalty = 2000.0 if old.predict_only or not old.detected else 0.0
                old_score = old_penalty + abs(old.err_x) + abs(old.err_y) - 0.22 * old.size_px
                if score < old_score:
                    best[mid] = d

        for mid, d in best.items():
            b = self.buffers[mid]
            b.append(d)
            while b and now - b[0].stamp > self.gate_history_s:
                b.popleft()
            self._update_gate_stats(d, b)
            self.latest[mid] = d
            self.last_seen_t[mid] = now
            if d.detected and not d.predict_only:
                self.last_real_det_t[mid] = now

    def _update_gate_stats(self, d: Detection, b: Deque[Detection]) -> None:
        real = [x for x in b if x.detected and not x.predict_only]
        use = real if real else list(b)
        d.stable_samples = len(use)
        d.stable_std_px = max(stddev([x.err_x for x in use]), stddev([x.err_y for x in use])) if use else 999.0

    # ---------------- sensing / selection ----------------
    def fresh_det(self, mid: int, now: float) -> Optional[Detection]:
        d = self.latest.get(mid)
        if d is None:
            return None
        age = now - d.stamp
        if d.predict_only or not d.detected:
            if age > self.predict_timeout_s or d.kf_age_s > self.predict_timeout_s:
                return None
        elif age > self.det_timeout_s:
            return None
        return d

    def altitude(self, now: float) -> Tuple[Optional[float], str]:
        if self.lidar_m is not None and self.lidar_t is not None and now - self.lidar_t <= self.alt_timeout_s:
            if self.z_min <= self.lidar_m <= self.z_max:
                return float(self.lidar_m), "lidar"
        if self.baro_m is not None and self.baro_t is not None and now - self.baro_t <= self.alt_timeout_s:
            if 0.02 <= self.baro_m <= 30.0:
                return float(self.baro_m), "baro"
        return None, "none"

    def metric_state(self, d: Detection, now: float) -> Tuple[float, float, float, float, float, str]:
        alt, src = self.altitude(now)
        if alt is None:
            alt = d.z_m if finite(d.z_m) and d.z_m > 0.02 else 1.0
            src = "aruco_z" if d.z_m > 0.02 else "fallback_1m"
        z = clamp(alt, self.z_min, self.z_max)
        rel_f = -z * d.err_y / max(1.0, self.fy)
        rel_r = z * d.err_x / max(1.0, self.fx)
        rel_f_rate = -z * d.err_y_rate / max(1.0, self.fy)
        rel_r_rate = z * d.err_x_rate / max(1.0, self.fx)
        return rel_f, rel_r, rel_f_rate, rel_r_rate, z, src

    def _soft_deadband_px(self, value: float, deadband: float) -> float:
        av = abs(float(value))
        deadband = abs(float(deadband))
        if av <= deadband:
            return 0.0
        return sign(value) * (av - deadband)

    def _rate_condition_px_s(self, value: float, deadband: float, clip_px_s: float) -> float:
        value = float(value)
        deadband = abs(float(deadband))
        clip_px_s = max(deadband, abs(float(clip_px_s)))
        if abs(value) <= deadband:
            return 0.0
        return clamp(value, -clip_px_s, clip_px_s)

    def control_metric_state(self, mid: int, d: Detection, z: float):
        ex_ctrl = float(d.err_x)
        ey_ctrl = float(d.err_y)
        ex_rate_ctrl = float(d.err_x_rate)
        ey_rate_ctrl = float(d.err_y_rate)
        dbg = {
            "control_err_x_px": ex_ctrl,
            "control_err_y_px": ey_ctrl,
            "control_err_x_rate_px_s": ex_rate_ctrl,
            "control_err_y_rate_px_s": ey_rate_ctrl,
            "large_control_softened": 0.0,
        }

        if mid == self.large_id:
            ex_ctrl = self._soft_deadband_px(d.err_x, self.large_control_deadband_x)
            ey_ctrl = self._soft_deadband_px(d.err_y, self.large_control_deadband_y)
            ex_rate_ctrl = self._rate_condition_px_s(d.err_x_rate, self.large_rate_deadband, self.large_rate_clip)
            ey_rate_ctrl = self._rate_condition_px_s(d.err_y_rate, self.large_rate_deadband, self.large_rate_clip)
            dbg.update({
                "control_err_x_px": float(ex_ctrl),
                "control_err_y_px": float(ey_ctrl),
                "control_err_x_rate_px_s": float(ex_rate_ctrl),
                "control_err_y_rate_px_s": float(ey_rate_ctrl),
                "large_control_softened": 1.0,
            })

        rel_f = -float(z) * ey_ctrl / max(1.0, self.fy)
        rel_r = float(z) * ex_ctrl / max(1.0, self.fx)
        rel_f_rate = -float(z) * ey_rate_ctrl / max(1.0, self.fy)
        rel_r_rate = float(z) * ex_rate_ctrl / max(1.0, self.fx)
        dbg.update({
            "control_rel_fwd": float(rel_f),
            "control_rel_right": float(rel_r),
            "control_rel_fwd_rate": float(rel_f_rate),
            "control_rel_right_rate": float(rel_r_rate),
        })
        return rel_f, rel_r, rel_f_rate, rel_r_rate, dbg

    def small_takeover_ready(self, d: Detection, alt: Optional[float]) -> bool:
        if alt is None or alt > self.small_takeover_alt:
            return False
        if d.predict_only or not d.detected:
            return False
        if d.size_px < self.small_takeover_min_size:
            return False
        if d.kf_samples < self.small_takeover_min_kf_samples and d.stable_samples < self.small_takeover_min_kf_samples:
            return False
        if abs(d.err_x) > self.small_takeover_gate_x or abs(d.err_y) > self.small_takeover_gate_y:
            return False
        return True

    def small_handoff_ready(self, d: Detection, alt: Optional[float]) -> bool:
        if not self.small_handoff_enable:
            return False
        if alt is None or alt > self.small_handoff_alt:
            return False
        if d.predict_only or not d.detected:
            return False
        if d.size_px < self.small_handoff_min_size:
            return False
        if abs(d.err_x) > self.small_handoff_gate_x or abs(d.err_y) > self.small_handoff_gate_y:
            return False
        if abs(d.err_x_rate) > self.small_handoff_max_rate or abs(d.err_y_rate) > self.small_handoff_max_rate:
            return False
        return True

    def reset_pid_for_marker_switch(self):
        if self.pid_i_reset_on_marker_switch:
            self.pid_fwd.reset()
            self.pid_right.reset()

    def select_marker(self, now: float):
        large = self.fresh_det(self.large_id, now)
        small = self.fresh_det(self.small_id, now)
        alt, _ = self.altitude(now)
        dbg = {"small_locked": 1.0 if self.small_locked else 0.0, "small_takeover_ready": 0.0, "small_handoff_ready": 0.0}

        if small is not None:
            ready = self.small_takeover_ready(small, alt)
            handoff_ready = self.small_handoff_ready(small, alt)
            dbg["small_takeover_ready"] = 1.0 if ready else 0.0
            dbg["small_handoff_ready"] = 1.0 if handoff_ready else 0.0
            if ready and not self.small_locked:
                self.small_locked = True
                self.small_lock_t = now
                self.active_mid = self.small_id
                self.reset_pid_for_marker_switch()
                dbg["small_lock_new"] = 1.0

        if self.small_locked:
            if small is not None:
                return self.small_id, small, "small_locked", dbg
            if self.small_return_large_enable and large is not None:
                last_small = self.last_real_det_t.get(self.small_id)
                lost_s = 999.0 if last_small is None else now - last_small
                if lost_s > self.small_return_large_lost_s:
                    self.small_locked = False
                    self.active_mid = self.large_id
                    self.reset_pid_for_marker_switch()
                    return self.large_id, large, "small_lost_return_large", dbg
            return None, None, "small_locked_no_small", dbg

        if large is not None:
            self.active_mid = self.large_id
            return self.large_id, large, "large_primary", dbg
        if small is not None and self.small_takeover_ready(small, alt):
            self.small_locked = True
            self.small_lock_t = now
            self.active_mid = self.small_id
            self.reset_pid_for_marker_switch()
            return self.small_id, small, "small_fallback_ready", dbg
        if small is not None and self.small_handoff_ready(small, alt):
            self.active_mid = self.small_id
            return self.small_id, small, "small_handoff_no_big", dbg
        return None, None, "no_marker", dbg

    # ---------------- gates / shaping ----------------
    def small_precision_factor(self, z: Optional[float]) -> float:
        if not self.small_precision_enable or z is None or not finite(z):
            return 0.0
        start = max(0.05, self.small_precision_start_z)
        full = clamp(self.small_precision_full_z, 0.05, start - 0.02)
        if z >= start:
            return 0.0
        if z <= full:
            return 1.0
        return clamp((start - z) / max(0.02, start - full), 0.0, 1.0)

    def box_halves(self, mid: int, z: Optional[float]) -> Tuple[float, float]:
        if mid == self.small_id:
            pf = self.small_precision_factor(z)
            return blend(self.small_box_x, self.small_precision_box_x, pf), blend(self.small_box_y, self.small_precision_box_y, pf)
        return self.large_box_x, self.large_box_y

    def box_debug(self, mid: int, d: Detection, z: float) -> Dict[str, float]:
        hx, hy = self.box_halves(mid, z)
        pf = self.small_precision_factor(z) if mid == self.small_id else 0.0
        inside = abs(d.err_x) <= hx and abs(d.err_y) <= hy
        near = abs(d.err_x) <= hx + self.track_keep_margin_x and abs(d.err_y) <= hy + self.track_keep_margin_y
        center_score = max(abs(d.err_x) / max(1.0, hx), abs(d.err_y) / max(1.0, hy))
        return {
            "inside_box": 1.0 if inside else 0.0,
            "near_box": 1.0 if near else 0.0,
            "box_half_x_px": float(hx),
            "box_half_y_px": float(hy),
            "small_precision_factor": float(pf),
            "small_precision_active": 1.0 if pf > 0.05 else 0.0,
            "center_score": float(center_score),
        }

    def pid_gains(self, mid: int):
        if mid == self.small_id:
            return self.small_pid_kp_fwd, self.small_pid_ki_fwd, self.small_pid_kd_fwd, self.small_pid_kp_right, self.small_pid_ki_right, self.small_pid_kd_right
        return self.large_pid_kp_fwd, self.large_pid_ki_fwd, self.large_pid_kd_fwd, self.large_pid_kp_right, self.large_pid_ki_right, self.large_pid_kd_right

    def pid_integral_allowed(self, mid: int, d: Detection, alt: Optional[float], inside_or_near: bool) -> bool:
        if not self.pid_i_enable_param:
            return False
        if d.predict_only or not d.detected:
            return False
        if alt is not None and alt < self.pid_i_disable_near_ground:
            return False
        if max(abs(d.err_x), abs(d.err_y)) > self.pid_i_disable_err_px:
            return False
        if not inside_or_near:
            return False
        return True

    def adaptive_limits(self, mid: int, d: Detection, z: float):
        is_small = mid == self.small_id
        pf = self.small_precision_factor(z) if is_small else 0.0
        base_speed = self.small_speed_max if is_small else self.large_speed_max
        base_lat = self.small_lateral_max if is_small else self.large_lateral_max
        base_accel = self.small_accel_max if is_small else self.large_accel_max

        if is_small:
            near_x = blend(self.near_center_x, 42.0, pf)
            near_y = blend(self.near_center_y, 42.0, pf)
            near_speed_scale = self.near_center_speed_scale
            near_accel_scale = self.near_center_accel_scale
            alpha_near = self.cmd_alpha_near
            alpha_far = self.cmd_alpha_far
        else:
            near_x = self.large_near_center_x
            near_y = self.large_near_center_y
            near_speed_scale = self.large_near_center_speed_scale
            near_accel_scale = self.large_near_center_accel_scale
            alpha_near = self.large_cmd_alpha_near
            alpha_far = self.large_cmd_alpha_far

        nx = clamp(abs(d.err_x) / max(1.0, near_x), 0.0, 1.0)
        ny = clamp(abs(d.err_y) / max(1.0, near_y), 0.0, 1.0)
        far_factor = max(nx, ny)
        speed_scale = blend(near_speed_scale, 1.0, far_factor)
        accel_scale = blend(near_accel_scale, 1.0, far_factor)
        alpha = blend(alpha_near, alpha_far, far_factor)

        if is_small and pf > 0.0:
            base_speed = min(base_speed, blend(base_speed, min(base_speed, 0.34), pf))
            base_lat = min(base_lat, blend(base_lat, min(base_lat, 0.24), pf))

        speed_max = max(0.05, base_speed * speed_scale)
        lat_max = max(0.04, base_lat * speed_scale)
        accel_max = max(0.12, base_accel * accel_scale)
        return speed_max, lat_max, accel_max, alpha, {
            "far_factor": float(far_factor),
            "speed_scale": float(speed_scale),
            "accel_scale": float(accel_scale),
            "alpha_used": float(alpha),
            "speed_max_used_mps": float(speed_max),
            "lat_max_used_mps": float(lat_max),
            "accel_max_used_mps2": float(accel_max),
        }

    def apply_small_y_priority(self, mid: int, d: Detection, vx: float, vy: float):
        dbg = {
            "small_y_priority_used": 0.0,
            "small_y_priority_vx_before": float(vx),
            "small_y_priority_vy_before": float(vy),
            "small_y_priority_vx_after": float(vx),
            "small_y_priority_vy_after": float(vy),
        }
        if mid != self.small_id or not self.small_y_priority_enable:
            return float(vx), float(vy), dbg

        ax = abs(d.err_x)
        ay = abs(d.err_y)
        x_scale = 1.0
        if ay >= self.small_y_priority_y_high:
            x_scale = clamp(self.small_y_priority_x_scale_high, 0.0, 1.0)
        elif ay >= self.small_y_priority_y_mid:
            x_scale = clamp(self.small_y_priority_x_scale_mid, 0.0, 1.0)
        rescue = ax >= self.small_y_priority_x_rescue
        if rescue:
            x_scale = max(x_scale, clamp(self.small_y_priority_x_rescue_scale, 0.0, 1.0))
        vy2 = float(vy) * x_scale
        if rescue:
            vy2 = clamp(vy2, -abs(self.small_y_priority_x_rescue_max), abs(self.small_y_priority_x_rescue_max))

        vx2 = float(vx)
        if ay >= self.small_y_priority_y_boost_px and vx2 * (-d.err_y) > 0.0:
            vx2 = clamp(vx2 * max(1.0, self.small_y_priority_y_boost_scale), -abs(self.small_y_priority_y_boost_max), abs(self.small_y_priority_y_boost_max))

        dbg.update({
            "small_y_priority_used": 1.0,
            "small_y_priority_abs_x_px": ax,
            "small_y_priority_abs_y_px": ay,
            "small_y_priority_x_scale": float(x_scale),
            "small_y_priority_x_rescue": 1.0 if rescue else 0.0,
            "small_y_priority_vx_after": float(vx2),
            "small_y_priority_vy_after": float(vy2),
        })
        return float(vx2), float(vy2), dbg

    def apply_forward_reverse_guard(self, mid: int, d: Detection, vx: float):
        dbg = {"reverse_guard_used": 0.0, "reverse_guard_vx_before": float(vx), "reverse_guard_vx_after": float(vx)}
        if not self.forward_reverse_guard_enable:
            return float(vx), dbg
        dbg["reverse_guard_used"] = 1.0
        if vx >= 0.0:
            return float(vx), dbg

        is_small = mid == self.small_id
        allow_y = self.reverse_allow_y_small if is_small else self.reverse_allow_y_large
        allow_max = self.reverse_allow_max_small if is_small else self.reverse_allow_max_large
        if d.err_y >= allow_y:
            vx2 = max(float(vx), -abs(allow_max))
            dbg["reverse_guard_vx_after"] = float(vx2)
            return float(vx2), dbg
        if d.err_y >= self.reverse_brake_y:
            vx2 = max(float(vx), -abs(self.reverse_brake_max))
            dbg["reverse_guard_vx_after"] = float(vx2)
            return float(vx2), dbg
        sustain = self.forward_sustain_when_blocked if self.last_cmd_vx > self.forward_sustain_when_blocked else 0.0
        vx2 = max(0.0, sustain)
        dbg["reverse_guard_vx_after"] = float(vx2)
        return float(vx2), dbg

    def smooth_and_accel_limit(self, vx: float, vy: float, accel_max: float, alpha: float, now: float):
        if abs(vx) < self.min_cmd_deadband:
            vx = 0.0
        if abs(vy) < self.min_cmd_deadband:
            vy = 0.0
        if self.last_cmd_t is None or now - self.last_cmd_t > 0.45:
            self.last_cmd_t = now
            self.last_cmd_vx = float(vx)
            self.last_cmd_vy = float(vy)
            return float(vx), float(vy), {"slew_first": 1.0, "slew_alpha": float(alpha), "slew_accel_max": float(accel_max)}

        dt = clamp(now - self.last_cmd_t, 0.015, 0.12)
        target_vx = clamp(alpha, 0.0, 1.0) * float(vx) + (1.0 - clamp(alpha, 0.0, 1.0)) * self.last_cmd_vx
        target_vy = clamp(alpha, 0.0, 1.0) * float(vy) + (1.0 - clamp(alpha, 0.0, 1.0)) * self.last_cmd_vy
        accel_used = float(accel_max)
        if target_vx * self.last_cmd_vx < 0.0 or target_vy * self.last_cmd_vy < 0.0:
            accel_used = max(accel_used, self.brake_accel_max)
        dvx, dvy = vector_limit(target_vx - self.last_cmd_vx, target_vy - self.last_cmd_vy, max(0.01, accel_used) * dt)
        self.last_cmd_vx += dvx
        self.last_cmd_vy += dvy
        self.last_cmd_t = now
        return self.last_cmd_vx, self.last_cmd_vy, {"slew_first": 0.0, "slew_alpha": float(alpha), "slew_accel_max": float(accel_used), "slew_dt": float(dt)}

    # ---------------- descent / land ----------------
    def small_descend_gate_ok(self, d: Detection, z: float) -> bool:
        if d.predict_only or not d.detected:
            return False
        pf = self.small_precision_factor(z)
        gate_x = blend(self.small_descend_gate_x, self.small_precision_descend_gate_x, pf)
        gate_y = blend(self.small_descend_gate_y, self.small_precision_descend_gate_y, pf)
        min_samples = int(round(blend(self.small_descend_min_kf_samples, self.small_precision_descend_min_kf_samples, pf)))
        sample_ok = d.kf_samples >= min_samples or d.stable_samples >= min_samples
        return sample_ok and abs(d.err_x) <= gate_x and abs(d.err_y) <= gate_y

    def small_approach_descend_ok(self, d: Detection, z: float) -> bool:
        if not self.small_approach_descend_enable:
            return False
        if d.predict_only or not d.detected:
            return False
        if finite(z) and z <= max(0.05, self.final_land_alt - 0.03):
            return False
        sample_ok = d.kf_samples >= self.small_approach_descend_min_samples or d.stable_samples >= self.small_approach_descend_min_samples
        pos_ok = abs(d.err_x) <= self.small_approach_descend_gate_x and abs(d.err_y) <= self.small_approach_descend_gate_y
        rate_ok = abs(d.err_x_rate) <= self.small_approach_descend_max_rate and abs(d.err_y_rate) <= self.small_approach_descend_max_rate
        return bool(sample_ok and pos_ok and rate_ok)

    def descend_rate(self, mid: int, d: Detection, inside: bool, alt: Optional[float], z: float, selection: str = "") -> float:
        if d.predict_only or not d.detected:
            return 0.0
        if mid == self.small_id:
            tight_ok = inside and self.small_descend_gate_ok(d, z)
            approach_ok = self.small_approach_descend_ok(d, z)
            if selection.startswith("small_handoff") and self.small_handoff_ready(d, alt):
                if approach_ok or (abs(d.err_x) <= self.small_handoff_gate_x and abs(d.err_y) <= self.small_handoff_gate_y):
                    return max(self.small_handoff_descend_vz, self.small_approach_descend_vz * 0.75)
            if not tight_ok:
                return self.small_approach_descend_vz if approach_ok else self.small_out_vz
            pf = self.small_precision_factor(z)
            if pf <= 0.0:
                return self.small_vz
            return blend(self.small_mid_vz, self.small_precision_vz, pf)
        if alt is not None and alt <= self.small_takeover_alt and not self.allow_large_below_small:
            return 0.0
        return self.large_vz if inside else self.large_out_vz

    def final_land_gate_ok(self, mid: int, d: Detection, z: float) -> bool:
        if self.land_only_small and mid != self.small_id:
            return False
        if d.predict_only or not d.detected:
            return False
        sample_ok = d.kf_samples >= self.final_land_min_samples or d.stable_samples >= self.final_land_min_samples
        pos_ok = abs(d.err_x) <= self.final_land_gate_x and abs(d.err_y) <= self.final_land_gate_y
        rate_ok = abs(d.err_x_rate) <= self.final_land_max_rate and abs(d.err_y_rate) <= self.final_land_max_rate
        if sample_ok and pos_ok and rate_ok:
            return True
        if self.final_land_allow_approach_gate and finite(z) and z <= self.final_land_alt:
            return self.small_approach_descend_ok(d, z) and sample_ok and rate_ok
        return False

    def should_land(self, mid: int, d: Detection, inside: bool, alt: Optional[float], z: float, now: float) -> bool:
        if alt is None or alt > self.final_land_alt:
            self.final_since = None
            return False
        if not self.final_land_gate_ok(mid, d, z):
            self.final_since = None
            return False
        if self.final_since is None:
            self.final_since = now
        return now - self.final_since >= self.final_hold_s

    # ---------------- main loop ----------------
    def control_loop(self):
        now = time.time()
        if not self.enabled:
            return

        alt, alt_src = self.altitude(now)
        mid, det, sel, sel_dbg = self.select_marker(now)
        if det is None or mid is None:
            self.handle_lost(now, alt, alt_src, sel)
            return

        if self.last_marker_id != mid:
            self.reset_pid_for_marker_switch()
        self.last_marker_id = mid
        self.last_visible_err_x[mid] = det.err_x
        self.last_visible_err_y[mid] = det.err_y

        rel_f, rel_r, rel_f_rate, rel_r_rate, z, zsrc = self.metric_state(det, now)
        ctrl_rel_f, ctrl_rel_r, ctrl_rel_f_rate, ctrl_rel_r_rate, ctrl_dbg = self.control_metric_state(mid, det, z)
        bdbg = self.box_debug(mid, det, z)
        inside = bdbg["inside_box"] >= 0.5
        near = bdbg["near_box"] >= 0.5

        speed_max, lat_max, accel_max, alpha, limit_dbg = self.adaptive_limits(mid, det, z)
        kp_f, ki_f, kd_f, kp_r, ki_r, kd_r = self.pid_gains(mid)
        i_enable = self.pid_integral_allowed(mid, det, alt, inside or near)

        vx_pid, fdbg = self.pid_fwd.compute(mid, ctrl_rel_f, ctrl_rel_f_rate, now, kp_f, ki_f, kd_f, self.pid_integral_limit, i_enable, speed_max)
        vy_pid, rdbg = self.pid_right.compute(mid, ctrl_rel_r, ctrl_rel_r_rate, now, kp_r, ki_r, kd_r, self.pid_integral_limit, i_enable, lat_max)

        raw_vx = vx_pid
        raw_vy = clamp(vy_pid, -lat_max, lat_max)
        raw_vx, raw_vy, small_y_dbg = self.apply_small_y_priority(mid, det, raw_vx, raw_vy)
        raw_vx, reverse_dbg = self.apply_forward_reverse_guard(mid, det, raw_vx)
        raw_vx, raw_vy = vector_limit(raw_vx, raw_vy, speed_max)

        vx, vy, slew_dbg = self.smooth_and_accel_limit(raw_vx, raw_vy, accel_max, alpha, now)
        vx2, post_guard_dbg = self.apply_forward_reverse_guard(mid, det, vx)
        if abs(vx2 - vx) > 1e-6:
            vx = vx2
            self.last_cmd_vx = float(vx)

        vz = self.descend_rate(mid, det, inside, alt, z, sel)
        self.phase = "TRACK_SMALL" if mid == self.small_id else "TRACK_BIG"
        if det.predict_only or not det.detected:
            self.phase = "PREDICT_TRACK_SMALL" if mid == self.small_id else "PREDICT_TRACK_BIG"

        extra = {
            **sel_dbg,
            **bdbg,
            **ctrl_dbg,
            **limit_dbg,
            **{f"fwd_{k}": v for k, v in fdbg.items()},
            **{f"right_{k}": v for k, v in rdbg.items()},
            **small_y_dbg,
            **reverse_dbg,
            **{f"post_{k}": v for k, v in post_guard_dbg.items()},
            **slew_dbg,
            "rel_fwd": rel_f,
            "rel_right": rel_r,
            "rel_fwd_rate": rel_f_rate,
            "rel_right_rate": rel_r_rate,
            "raw_vx": raw_vx,
            "raw_vy": raw_vy,
            "cmdspd": norm2(vx, vy),
            "z_used": z,
            "small_descend_gate_ok": 1.0 if self.small_descend_gate_ok(det, z) else 0.0,
            "small_approach_descend_ok": 1.0 if (mid == self.small_id and self.small_approach_descend_ok(det, z)) else 0.0,
            "final_land_gate_ok": 1.0 if self.final_land_gate_ok(mid, det, z) else 0.0,
            "pid_controller_used": 1.0,
            "vision_predict_only": 1.0 if det.predict_only else 0.0,
            "vision_detected": 1.0 if det.detected else 0.0,
            "image_err_x_sign": self.image_err_x_sign,
            "image_err_y_sign": self.image_err_y_sign,
        }

        if self.should_land(mid, det, inside, alt, z, now):
            self.phase = "FINAL_LAND"
            self.publish_cmd("land", 0.0, 0.0, 0.0, "final_small_pid_gate_ok", mid, det, alt, alt_src, extra)
            return

        self.publish_cmd("track", vx, vy, vz, f"pid_velocity/{sel}/{zsrc}", mid, det, alt, alt_src, extra)

    # ---------------- lost marker ----------------
    def blind_forward_speed_for_marker(self, mid: int) -> float:
        if mid == self.small_id:
            v = self.blind_forward_small_speed
        elif mid == self.large_id:
            v = self.blind_forward_large_speed
        else:
            v = self.blind_forward_speed
        if not finite(v) or abs(v) < 1e-4:
            v = self.blind_forward_speed
        return abs(float(v))

    def lost_marker_context(self) -> Optional[int]:
        if self.last_marker_id in (self.large_id, self.small_id):
            return self.last_marker_id
        best_mid = None
        best_t = -1.0
        for mid in (self.large_id, self.small_id):
            t = self.last_real_det_t.get(mid)
            if t is not None and t > best_t:
                best_t = float(t)
                best_mid = mid
        return best_mid

    def blind_forward_reacquire_allowed(self, mid: int, lost_age: float, alt: Optional[float]):
        is_small = mid == self.small_id
        max_s = self.blind_forward_small_max_s if is_small else self.blind_forward_large_max_s
        last_ey = self.last_visible_err_y.get(mid)
        alt_ok = alt is None or alt >= self.blind_forward_min_alt
        y_ok = True
        if self.blind_forward_require_last_y_not_behind and last_ey is not None:
            y_ok = float(last_ey) <= self.blind_forward_last_y_max
        age_ok = lost_age >= self.blind_forward_delay_s and lost_age <= max_s
        allowed = self.blind_forward_on_lost_enable and age_ok and alt_ok and y_ok
        return bool(allowed), {
            "blind_forward_allowed": 1.0 if allowed else 0.0,
            "blind_forward_lost_age_s": float(lost_age),
            "blind_forward_max_s": float(max_s),
            "blind_forward_speed_mps": float(self.blind_forward_speed_for_marker(mid)),
        }

    def handle_lost(self, now: float, alt: Optional[float], alt_src: str, reason: str):
        mid = self.lost_marker_context()
        if mid in (self.large_id, self.small_id):
            last_real = self.last_real_det_t.get(mid)
            lost_age = 999.0 if last_real is None else now - last_real
            is_small = mid == self.small_id

            self.pid_fwd.state.setdefault(mid, AxisPIDState()).integ *= 0.85
            self.pid_right.state.setdefault(mid, AxisPIDState()).integ *= 0.85

            hold_s = self.lost_hold_last_cmd_s
            scale = 1.0
            last_ey = self.last_visible_err_y.get(mid)
            small_lost_y_large = False
            if is_small and last_ey is not None and abs(last_ey) >= self.small_lost_hold_y_large_px:
                hold_s = min(hold_s, self.small_lost_hold_y_large_s)
                scale = clamp(self.small_lost_hold_y_large_scale, 0.0, 1.0)
                small_lost_y_large = True

            blind_ok, blind_dbg = self.blind_forward_reacquire_allowed(mid, lost_age, alt)
            if blind_ok:
                vx_target = self.blind_forward_speed_for_marker(mid)
                vy_target = 0.0
                vx, vy, slew_dbg = self.smooth_and_accel_limit(vx_target, vy_target, max(0.05, self.blind_forward_accel), clamp(self.blind_forward_alpha, 0.05, 1.0), now)
                self.phase = "BLIND_FORWARD_REACQUIRE_SMALL" if is_small else "BLIND_FORWARD_REACQUIRE_BIG"
                self.publish_cmd("track", vx, vy, 0.0, "blind_forward_reacquire_no_marker_no_descend", mid, None, alt, alt_src, {**blind_dbg, **slew_dbg, "lost_age_s": lost_age})
                return

            if self.lost_hold_last_cmd_enable and lost_age <= hold_s:
                vx = self.last_cmd_vx * scale
                vy = self.last_cmd_vy * scale
                if norm2(vx, vy) >= self.lost_hold_min_cmd:
                    vmax = self.small_lost_hold_max if is_small else self.large_lost_hold_max
                    vx, vy = vector_limit(vx, vy, vmax)
                    self.phase = "LOST_HOLD_LAST_CMD"
                    self.publish_cmd("track", vx, vy, 0.0, "lost_marker_hold_last_cmd_no_descend", mid, None, alt, alt_src, {
                        "lost_age_s": lost_age,
                        "lost_hold_last_cmd_used": 1.0,
                        "lost_hold_scale_used": scale,
                        "small_lost_y_large": 1.0 if small_lost_y_large else 0.0,
                    })
                    return

            if lost_age <= hold_s + self.lost_brake_s:
                vx = self.last_cmd_vx * 0.35
                vy = self.last_cmd_vy * 0.35
                vmax = self.small_lost_hold_max if is_small else self.large_lost_hold_max
                vx, vy = vector_limit(vx, vy, vmax * 0.45)
                self.phase = "LOST_BRAKE"
                self.publish_cmd("track", vx, vy, 0.0, "lost_marker_brake_no_descend", mid, None, alt, alt_src, {"lost_age_s": lost_age})
                return

        self.pid_fwd.reset()
        self.pid_right.reset()
        self.phase = "WAIT_MARKER_SMALL" if self.small_locked else "WAIT_MARKER"
        self.last_cmd_vx = 0.0
        self.last_cmd_vy = 0.0
        self.publish_cmd("hold", 0.0, 0.0, 0.0, reason, None, None, alt, alt_src)

    # ---------------- publish ----------------
    def publish_cmd(self, mode: str, vx: float, vy: float, vz_down: float, reason: str,
                    marker_id: Optional[int] = None, det: Optional[Detection] = None,
                    alt: Optional[float] = None, alt_src: str = "none",
                    extra: Optional[Dict[str, float]] = None):
        self.seq += 1
        cmd = {
            "source": "landing_node_pid_velocity_axis_sign_2026_06_12",
            "seq": self.seq,
            "stamp": time.time(),
            "mode": mode,
            "phase": self.phase,
            "frame": self.command_frame,
            "vx": float(vx),
            "vy": float(vy),
            "vz_down": float(vz_down),
            "vx_body": float(vx),
            "vy_body": float(vy),
            "vx_local": float(vx),
            "vy_local": float(vy),
            "yaw_rate": 0.0,
            "use_yaw": False,
            "reason": reason,
            "marker_id": None if marker_id is None else int(marker_id),
            "alt_m": None if alt is None else float(alt),
            "alt_source": alt_src,
            "alt_band": "SMALL_ONLY" if marker_id == self.small_id else "BIG_ONLY" if marker_id == self.large_id else "NONE",
            "image_err_x_sign": float(self.image_err_x_sign),
            "image_err_y_sign": float(self.image_err_y_sign),
        }
        if det is not None:
            cmd.update({
                "err_x": float(det.err_x),
                "err_y": float(det.err_y),
                "raw_err_x": float(det.raw_err_x),
                "raw_err_y": float(det.raw_err_y),
                "filtered_err_x": float(det.filtered_err_x),
                "filtered_err_y": float(det.filtered_err_y),
                "err_x_rate_px_s": float(det.err_x_rate),
                "err_y_rate_px_s": float(det.err_y_rate),
                "center_x": float(det.center_x),
                "center_y": float(det.center_y),
                "size_px": float(det.size_px),
                "stable_samples": int(det.stable_samples),
                "stable_std_px": float(det.stable_std_px),
                "kf_age_s": float(det.kf_age_s),
                "kf_pos_std_px": float(det.kf_pos_std_px),
                "kf_nis": float(det.kf_nis),
                "kf_samples": int(det.kf_samples),
                "meas_noise_px": float(det.meas_noise_px),
                "predict_only": 1.0 if det.predict_only else 0.0,
                "detected": 1.0 if det.detected else 0.0,
                "marker_age_s": max(0.0, time.time() - det.stamp),
            })
        if extra:
            for k, v in extra.items():
                try:
                    cmd[k] = float(v)
                except Exception:
                    pass
        msg = String()
        msg.data = json.dumps(cmd, separators=(",", ":"), ensure_ascii=False)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LandingNodePID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
