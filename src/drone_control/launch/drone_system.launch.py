# Launch tuned: D435 RGB + debug rotation for camera mounted upside-down
#
# Nguyên tắc quan trọng:
#   - /camera/bottom/image_raw giữ RAW, không xoay, để ArUco + landing dùng đúng calib/PID/Kalman.
#   - AL_node tự xoay ảnh 180° trước khi YOLO/RKNN detect để AI nhìn thẳng.
#   - aruco_node chỉ xoay debug video/debug image, không xoay dữ liệu điều khiển.
#   - web_map_node tự xoay 180° khi phải lấy raw /camera/bottom/image_raw để web live nhìn thẳng.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_ngrok = LaunchConfiguration("use_ngrok")
    enable_ai = LaunchConfiguration("enable_ai")
    enable_raw_preview = LaunchConfiguration("enable_raw_preview")

    # Debug overlay/reference boxes. ArUco chỉ vẽ/debug, landing_node mới là nơi quyết định gate thật.
    large_box_x = 220.0
    large_box_y = 165.0
    small_box_x = 140.0
    small_box_y = 110.0
    small_precision_box_x = 86.0
    small_precision_box_y = 78.0

    # Calib mới của Intel RealSense D435 color stream 640x480
    d435_fx = 585.2398581
    d435_fy = 586.40005266
    d435_cx = 331.58390834
    d435_cy = 242.95158027

    return LaunchDescription([
        DeclareLaunchArgument("use_ngrok", default_value="false"),
        DeclareLaunchArgument("enable_ai", default_value="true"),
        DeclareLaunchArgument("enable_raw_preview", default_value="true"),

        Node(
            package="drone_control",
            executable="camera_node",
            name="camera_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                # D435 RGB/color node của bạn là /dev/video4.
                "camera_index": 4,
                "width": 640,
                "height": 480,
                "fps": 20.0,
                "topic_name": "/camera/bottom/image_raw",
                "frame_id": "bottom_camera",

                # Giữ ảnh raw không xoay cho ArUco/landing. Nếu camera_node của bạn quy ước -2 = no flip
                # thì giữ nguyên. Không dùng flip/rotate tại camera_node để tránh lệch hệ tọa độ calib.
                "flip_code": -2,

                # D435 color ở /dev/video4 báo format YUYV 640x480.
                "fourcc": "YUYV",

                "enable_preview": enable_raw_preview,
                "preview_topic_name": "/camera/bottom/raw_preview/compressed",
                "preview_fps": 4.0,
                "jpeg_quality": 50,
            }],
        ),

        TimerAction(period=1.0, actions=[
            Node(
                package="drone_control",
                executable="aruco_node",
                name="aruco_node",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    "image_topic": "/camera/bottom/image_raw",
                    "detections_json_topic": "/vision/aruco_detections_json",
                    "landing_cmd_topic": "/landing_cmd",
                    "sync_desired_mode_from_landing_cmd": True,

                    "large_id": 150,
                    "small_id": 40,
                    "marker_size_large_m": 0.70,
                    "marker_size_small_m": 0.08,

                    "use_calib": True,
                    "calib_path": "/ws/src/drone_control/config/camera_calib.npz",

                    # Chỉ xoay debug video/debug image để quan sát; không ảnh hưởng detect/JSON/PID/Kalman.
                    "debug_rotate_180": True,

                    # Quan trọng: landing_node mới ưu tiên filtered_err_x/y và err_rate từ ArUco Kalman.
                    "use_kalman_err": True,
                    "json_err_source": "filtered",
                    "publish_predict_only_json": False,
                    "kf_predict_timeout_s": 0.16,
                    "kf_accel_noise_px_s2": 350.0,
                    "kf_meas_noise_px_large": 7.0,
                    "kf_meas_noise_px_small": 11.0,
                    "kf_reset_timeout_s": 0.80,
                    "kf_gate_nis": 25.0,

                    "marker_state_timeout_s": 0.45,
                    "use_clahe": True,
                    "use_small_blur": True,
                    "detect_on_raw_gray": True,
                    "detect_on_processed": True,

                    # Không nhận ID40 quá bé/nhiễu để landing lock sai.
                    "min_avg_side_px_large": 5.5,
                    "min_avg_side_px_small": 8.0,

                    # JSON vẫn publish tất cả ID hợp lệ; primary legacy không được tự nhảy SMALL trong LARGE mode.
                    "small_priority_in_large": False,
                    "allow_small_override_in_large_mode": False,
                    "small_override_min_side_px": 24.0,

                    # Giữ False để nhẹ web live. Debug video vẫn được ghi thẳng nhờ debug_rotate_180.
                    "publish_debug_image": False,
                    "draw_landing_boxes": False,
                    "large_box_half_x_px": large_box_x,
                    "large_box_half_y_px": large_box_y,
                    "small_box_half_x_px": small_box_x,
                    "small_box_half_y_px": small_box_y,

                    "record_debug_video": True,
                    "record_raw_video": False,
                    "video_fps": 5.0,
                    "record_video_stride": 3,
                    "write_latest_debug_jpg": False,
                    "latest_debug_jpg_period_s": 1.0,
                    "csv_flush_every_n": 10,
                    "log_root_dir": "/ws/src/drone_control/aruco_logs",
                }],
            )
        ]),

        TimerAction(period=2.0, actions=[
            Node(
                condition=IfCondition(enable_ai),
                package="drone_control",
                executable="AL_node",
                name="AL_node",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    "model_path": "/ws/models/best(1).rknn",
                    "image_topic": "/camera/bottom/image_raw",
                    "debug_topic": "/camera/image_debug",
                    "detection_topic": "/person_detection",
                    "event_topic": "/person_event",
                    "gps_in_topic": "/drone_gps",
                    "gps_out_topic": "/detected_object_gps",
                    "hold_topic": "/detection_hold_active",
                    "record_dir": "/ws/detect_logs",

                    # Xoay riêng cho AI để model và debug AI nhìn thẳng. Không ảnh hưởng landing.
                    "rotate_input_180": True,

                    "cruise_image_size": 640,
                    "hold_image_size": 640,
                    "candidate_min_hits": 1,
                    "confirmed_min_hits": 3,
                    "hold_verify_timeout_sec": 1.25,

                    # Giữ record của AI vì cần xem lại sau chuyến bay.
                    "save_video": True,
                    "save_csv": True,
                    "save_snapshot_on_confirm": True,
                }],
            )
        ]),

        TimerAction(period=3.0, actions=[
            Node(
                package="drone_control",
                executable="landing_node",
                name="landing_node",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    # Topics / IDs / camera
                    "detections_topic": "/vision/aruco_detections_json",
                    "landing_enable_topic": "/landing_enable",
                    "lidar_topic": "/lidar_height_m",
                    "baro_topic": "/relative_alt_m",
                    "cmd_topic": "/landing_cmd",
                    "command_frame": "body_ned",
                    "large_id": 150,
                    "small_id": 40,

                    # D435 color 640x480 calib mới
                    "camera_fx_px": d435_fx,
                    "camera_fy_px": d435_fy,
                    "camera_cx_px": d435_cx,
                    "camera_cy_px": d435_cy,

                    # Đảo dấu trục ảnh cho camera/gimbal đang lắp ngược.
                    # Nếu marker lệch phải mà drone kéo trái: X = -1.0.
                    # Nếu tiến/lùi cũng ngược: đổi Y từ 1.0 thành -1.0.
                    "image_err_x_sign": -1.0,
                    "image_err_y_sign": -1.0,

                    "z_min_m": 0.15,
                    "z_max_m": 8.0,
                    "alt_timeout_s": 1.0,
                    "detection_timeout_s": 0.38,
                    "predict_only_timeout_s": 0.16,
                    "gate_history_s": 0.14,
                    "control_loop_hz": 30.0,

                    # Marker selection: không lock ID40 quá sớm, nhưng nếu BIG mất và SMALL đã thấy
                    # thì dùng SMALL handoff để tránh đoạn đi thẳng/no-descend.
                    "small_takeover_alt_m": 1.70,
                    "small_takeover_min_size_px": 22.0,
                    "small_takeover_gate_x_px": 170.0,
                    "small_takeover_gate_y_px": 190.0,
                    "small_takeover_min_kf_samples": 2,
                    "small_handoff_enable": True,
                    "small_handoff_alt_m": 2.35,
                    "small_handoff_min_size_px": 8.0,
                    "small_handoff_gate_x_px": 260.0,
                    "small_handoff_gate_y_px": 260.0,
                    "small_handoff_max_rate_px_s": 850.0,
                    "small_handoff_descend_vz": 0.040,
                    "small_lock_min_s": 1.0,
                    "small_return_to_large_enable": False,
                    "small_return_to_large_lost_s": 1.20,

                    # Boxes / precision
                    "large_box_half_x_px": large_box_x,
                    "large_box_half_y_px": large_box_y,
                    "small_box_half_x_px": small_box_x,
                    "small_box_half_y_px": small_box_y,
                    "small_precision_enable": True,
                    "small_precision_start_z_m": 1.25,
                    "small_precision_full_z_m": 0.82,
                    "small_precision_box_half_x_px": small_precision_box_x,
                    "small_precision_box_half_y_px": small_precision_box_y,
                    "track_keep_margin_x_px": 45.0,
                    "track_keep_margin_y_px": 55.0,

                    # PID velocity controller. Mapping trong code:
                    #   err_y -> rel_fwd  -> vx
                    #   err_x -> rel_right -> vy
                    "large_pid_kp_fwd": 0.82,
                    "large_pid_kd_fwd": 0.20,
                    "large_pid_ki_fwd": 0.003,
                    "large_pid_kp_right": 0.62,
                    "large_pid_kd_right": 0.16,
                    "large_pid_ki_right": 0.002,
                    "small_pid_kp_fwd": 1.05,
                    "small_pid_kd_fwd": 0.18,
                    "small_pid_ki_fwd": 0.002,
                    "small_pid_kp_right": 0.72,
                    "small_pid_kd_right": 0.16,
                    "small_pid_ki_right": 0.001,
                    "pid_integral_limit_m_s": 0.18,
                    "pid_i_enable": True,
                    "pid_i_disable_err_px": 160.0,
                    "pid_i_disable_near_ground_m": 0.65,
                    "pid_i_reset_on_marker_switch": True,

                    # Speed / acceleration shaping.
                    "large_speed_max_mps": 0.62,
                    "small_speed_max_mps": 0.52,
                    "large_lateral_max_mps": 0.34,
                    "small_lateral_max_mps": 0.26,
                    "large_accel_max_mps2": 0.52,
                    "small_accel_max_mps2": 0.62,
                    "brake_accel_max_mps2": 0.42,
                    "near_center_speed_scale": 0.34,
                    "near_center_accel_scale": 0.42,
                    "near_center_x_px": 55.0,
                    "near_center_y_px": 55.0,
                    "cmd_alpha_far": 0.46,
                    "cmd_alpha_near": 0.24,
                    "min_cmd_deadband_mps": 0.004,

                    # BIG/ID150 riêng: chống dội do marker lớn nhiễu tâm.
                    "large_control_deadband_x_px": 24.0,
                    "large_control_deadband_y_px": 20.0,
                    "large_rate_deadband_px_s": 18.0,
                    "large_rate_clip_px_s": 260.0,
                    "large_near_center_x_px": 95.0,
                    "large_near_center_y_px": 85.0,
                    "large_near_center_speed_scale": 0.30,
                    "large_near_center_accel_scale": 0.40,
                    "large_cmd_alpha_far": 0.42,
                    "large_cmd_alpha_near": 0.20,

                    # Reverse / overshoot guard.
                    "forward_reverse_guard_enable": True,
                    "reverse_brake_y_px": 115.0,
                    "reverse_brake_max_mps": 0.018,
                    "reverse_allow_y_px_large": 185.0,
                    "reverse_allow_y_px_small": 190.0,
                    "reverse_allow_max_mps_large": 0.13,
                    "reverse_allow_max_mps_small": 0.045,
                    "forward_sustain_when_blocked_mps": 0.010,

                    # ID40 Y priority.
                    "small_y_priority_enable": True,
                    "small_y_priority_y_mid_px": 60.0,
                    "small_y_priority_y_high_px": 90.0,
                    "small_y_priority_x_scale_mid_y": 0.34,
                    "small_y_priority_x_scale_high_y": 0.22,
                    "small_y_priority_x_rescue_px": 125.0,
                    "small_y_priority_x_rescue_scale": 0.72,
                    "small_y_priority_x_rescue_max_mps": 0.12,
                    "small_y_priority_y_boost_px": 120.0,
                    "small_y_priority_y_boost_scale": 1.14,
                    "small_y_priority_y_boost_max_mps": 0.46,

                    # Descent / land gates.
                    "large_descend_vz": 0.135,
                    "large_outside_descend_vz": 0.060,
                    "small_descend_vz": 0.095,
                    "small_mid_descend_vz": 0.065,
                    "small_precision_descend_vz": 0.040,
                    "small_outside_descend_vz": 0.0,

                    # SMALL approach descend.
                    "small_approach_descend_enable": True,
                    "small_approach_descend_min_z_m": 0.62,
                    "small_approach_descend_vz": 0.050,
                    "small_approach_descend_gate_x_px": 150.0,
                    "small_approach_descend_gate_y_px": 150.0,
                    "small_approach_descend_max_rate_px_s": 680.0,
                    "small_approach_descend_min_samples": 2,

                    "small_descend_gate_x_px": 115.0,
                    "small_descend_gate_y_px": 115.0,
                    "small_precision_descend_gate_x_px": 78.0,
                    "small_precision_descend_gate_y_px": 88.0,
                    "small_descend_min_kf_samples": 2,
                    "small_precision_descend_min_kf_samples": 2,
                    "allow_large_descend_below_small_alt": True,
                    "final_land_alt_m": 0.56,
                    "final_hold_s": 0.08,
                    "land_only_on_small": True,
                    "final_land_gate_x_px": 96.0,
                    "final_land_gate_y_px": 126.0,
                    "final_land_max_rate_px_s": 820.0,
                    "final_land_min_samples": 1,
                    "final_land_allow_approach_gate": True,

                    # Blind forward reacquire.
                    "blind_forward_on_lost_enable": True,
                    "blind_forward_speed_mps": 0.30,
                    "blind_forward_large_speed_mps": 0.45,
                    "blind_forward_small_speed_mps": 0.16,
                    "blind_forward_large_max_s": 6.00,
                    "blind_forward_small_max_s": 3.00,
                    "blind_forward_min_alt_m": 0.35,
                    "blind_forward_delay_s": 0.45,
                    "blind_forward_accel_mps2": 0.60,
                    "blind_forward_alpha": 0.62,
                    "blind_forward_require_last_y_not_behind": False,
                    "blind_forward_last_y_max_px": 9999.0,

                    # Lost marker fallback sau khi hết blind-forward window.
                    "lost_hold_last_cmd_enable": True,
                    "lost_hold_last_cmd_s": 0.50,
                    "lost_hold_min_cmd_mps": 0.006,
                    "large_lost_hold_max_mps": 0.55,
                    "small_lost_hold_max_mps": 0.18,
                    "small_lost_hold_y_large_px": 999.0,
                    "small_lost_hold_y_large_s": 0.50,
                    "small_lost_hold_y_large_scale": 1.00,
                    "lost_brake_s": 0.10,
                }],
            )
        ]),

        TimerAction(period=4.0, actions=[
            Node(
                package="drone_control",
                executable="drone_control",
                name="drone_control",
                output="screen",
                emulate_tty=True,
            )
        ]),

        TimerAction(period=5.0, actions=[
            Node(
                package="drone_control",
                executable="web_map_node",
                name="web_map_node",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    
                    "cam_height": 240,
                    "jpeg_quality": 50,
                }],
            )
        ]),

        TimerAction(period=7.0, actions=[
            ExecuteProcess(
                condition=IfCondition(use_ngrok),
                cmd=["bash", "-lc", "if command -v ngrok >/dev/null 2>&1; then ngrok http 5000 --log=stdout; else echo '[WARN] ngrok not installed or not in PATH'; fi"],
                output="screen",
            )
        ]),
    ])
