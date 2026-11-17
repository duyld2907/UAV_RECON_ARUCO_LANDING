#!/usr/bin/env python3
"""
AL Node - RKNN + ROS2
- 2 mode: cruise / hold
- Cruise: nh? h?n, ch? t m candidate
- Hold: n?ng h?n, x c nh?n confirmed
- Ghi video/log ?? xem l?i sau chuy?n bay
"""

import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String, Bool
from rknnlite.api import RKNNLite


def imgmsg_to_cv(msg: Image):
    if msg.encoding not in ('bgr8', 'rgb8'):
        raise ValueError(f'Unsupported encoding: {msg.encoding}')
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == 'rgb8':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame.copy()


def cv_to_imgmsg(frame, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = frame.shape[1] * 3
    msg.data = frame.tobytes()
    return msg


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    rw, rh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw = (new_w - rw) // 2
    dh = (new_h - rh) // 2
    canvas[dh:dh + rh, dw:dw + rw] = resized
    return canvas, r, dw, dh


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def nms_xyxy(dets, iou_thr=0.45):
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d['score'], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou_xyxy(best['bbox'], d['bbox']) < iou_thr]
    return keep


class ALNode(Node):
    def __init__(self):
        super().__init__('AL_node')

        # Core paths/topics
        self.declare_parameter('model_path', '/ws/models/best(1).rknn')
        self.declare_parameter('image_topic', '/camera/bottom/image_raw')
        self.declare_parameter('debug_topic', '/camera/image_debug')
        self.declare_parameter('detection_topic', '/person_detection')
        self.declare_parameter('event_topic', '/person_event')
        self.declare_parameter('gps_in_topic', '/drone_gps')
        self.declare_parameter('gps_out_topic', '/detected_object_gps')
        self.declare_parameter('hold_topic', '/detection_hold_active')
        self.declare_parameter('record_dir', '/ws/detect_logs')
        # Chỉ xoay ảnh bên trong AI node để model/debug nhìn đúng chiều.
        # Không làm thay đổi /camera/bottom/image_raw nên không ảnh hưởng ArUco landing.
        self.declare_parameter('rotate_input_180', False)

        # Decode/inference
        self.declare_parameter('single_class', True)
        self.declare_parameter('box_format', 'xyxy')
        self.declare_parameter('nms_threshold', 0.45)

        # Cruise mode (light)
        self.declare_parameter('cruise_image_size', 512)
        self.declare_parameter('cruise_every_n', 2)
        self.declare_parameter('candidate_score_threshold', 0.65)
        self.declare_parameter('candidate_min_hits', 2)

        # Hold mode (stronger verification)
        self.declare_parameter('hold_image_size', 640)
        self.declare_parameter('hold_every_n', 1)
        self.declare_parameter('confirmed_score_threshold', 0.50)
        self.declare_parameter('confirmed_min_hits', 3)
        self.declare_parameter('hold_verify_timeout_sec', 2.5)

        # Geometric filters
        self.declare_parameter('min_box_area_ratio', 0.003)
        self.declare_parameter('max_box_area_ratio', 0.35)
        self.declare_parameter('min_aspect_ratio', 0.18)
        self.declare_parameter('max_aspect_ratio', 1.8)
        self.declare_parameter('edge_margin_ratio', 0.03)
        self.declare_parameter('center_gate_ratio', 0.20)
        self.declare_parameter('max_center_jump_px', 120.0)

        # Logging
        self.declare_parameter('save_video', True)
        self.declare_parameter('save_csv', True)
        self.declare_parameter('save_snapshot_on_confirm', True)
        self.declare_parameter('snapshot_cooldown_sec', 3.0)
        self.declare_parameter('candidate_cooldown_sec', 3.0)
        self.declare_parameter('confirmed_cooldown_sec', 3.0)

        self.model_path = str(self.get_parameter('model_path').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.detection_topic = str(self.get_parameter('detection_topic').value)
        self.event_topic = str(self.get_parameter('event_topic').value)
        self.gps_in_topic = str(self.get_parameter('gps_in_topic').value)
        self.gps_out_topic = str(self.get_parameter('gps_out_topic').value)
        self.hold_topic = str(self.get_parameter('hold_topic').value)
        self.record_dir = str(self.get_parameter('record_dir').value)
        self.rotate_input_180 = bool(self.get_parameter('rotate_input_180').value)

        self.single_class = bool(self.get_parameter('single_class').value)
        self.box_format = str(self.get_parameter('box_format').value)
        self.nms_threshold = float(self.get_parameter('nms_threshold').value)

        self.cruise_image_size = int(self.get_parameter('cruise_image_size').value)
        self.cruise_every_n = max(1, int(self.get_parameter('cruise_every_n').value))
        self.candidate_score_threshold = float(self.get_parameter('candidate_score_threshold').value)
        self.candidate_min_hits = int(self.get_parameter('candidate_min_hits').value)

        self.hold_image_size = int(self.get_parameter('hold_image_size').value)
        self.hold_every_n = max(1, int(self.get_parameter('hold_every_n').value))
        self.confirmed_score_threshold = float(self.get_parameter('confirmed_score_threshold').value)
        self.confirmed_min_hits = int(self.get_parameter('confirmed_min_hits').value)
        self.hold_verify_timeout_sec = float(self.get_parameter('hold_verify_timeout_sec').value)

        self.min_box_area_ratio = float(self.get_parameter('min_box_area_ratio').value)
        self.max_box_area_ratio = float(self.get_parameter('max_box_area_ratio').value)
        self.min_aspect_ratio = float(self.get_parameter('min_aspect_ratio').value)
        self.max_aspect_ratio = float(self.get_parameter('max_aspect_ratio').value)
        self.edge_margin_ratio = float(self.get_parameter('edge_margin_ratio').value)
        self.center_gate_ratio = float(self.get_parameter('center_gate_ratio').value)
        self.max_center_jump_px = float(self.get_parameter('max_center_jump_px').value)

        self.save_video = bool(self.get_parameter('save_video').value)
        self.save_csv = bool(self.get_parameter('save_csv').value)
        self.save_snapshot_on_confirm = bool(self.get_parameter('save_snapshot_on_confirm').value)
        self.snapshot_cooldown_sec = float(self.get_parameter('snapshot_cooldown_sec').value)
        self.candidate_cooldown_sec = float(self.get_parameter('candidate_cooldown_sec').value)
        self.confirmed_cooldown_sec = float(self.get_parameter('confirmed_cooldown_sec').value)

        os.makedirs(self.record_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.video_path = os.path.join(self.record_dir, f'debug_{stamp}.avi')
        self.csv_path = os.path.join(self.record_dir, f'detect_{stamp}.csv')
        self.meta_path = os.path.join(self.record_dir, f'output_meta_{stamp}.txt')

        # ROS I/O
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 1)
        self.det_pub = self.create_publisher(String, self.detection_topic, 10)
        self.event_pub = self.create_publisher(String, self.event_topic, 10)
        self.gps_pub = self.create_publisher(NavSatFix, self.gps_out_topic, 10)

        self.create_subscription(Image, self.image_topic, self._cb_image, 1)
        self.create_subscription(NavSatFix, self.gps_in_topic, self._cb_gps, 10)
        self.create_subscription(Bool, self.hold_topic, self._cb_hold_active, 10)

        # State
        self.latest_frame = None
        self.latest_stamp = None
        self.latest_id = 0
        self.frame_lock = threading.Lock()
        self.stop_flag = False

        self.drone_lat = None
        self.drone_lon = None
        self.drone_alt = None
        self.hold_active = False
        self.hold_started_t = None

        self.last_center = None
        self.candidate_hits = 0
        self.confirmed_hits = 0
        self.last_candidate_t = 0.0
        self.last_confirmed_t = 0.0
        self.last_snapshot_t = 0.0

        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.logged_output_meta = False

        self.mode_name = 'CRUISE'
        self.last_det_state = 'NONE'

        self.rknn = RKNNLite()
        self._init_model()
        self._init_csv()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info('AL_node started')
        self.get_logger().info(f'rotate_input_180={self.rotate_input_180}')
        self.get_logger().info('candidate = nghi ngo -> xin drone dung 2.2s')
        self.get_logger().info('confirmed = sau khi dung lai, van thay on dinh -> dang tin hon')

    # ---------- init ----------
    def _init_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model not found: {self.model_path}')
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f'load_rknn failed: {ret}')
        try:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        except Exception:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f'init_runtime failed: {ret}')

    def _init_csv(self):
        if not self.save_csv:
            return
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'unix_time', 'frame_id', 'mode', 'state', 'score', 'confirmed',
            'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
            'drone_lat', 'drone_lon', 'drone_alt'
        ])
        self.csv_file.flush()

    # ---------- callbacks ----------
    def _cb_gps(self, msg: NavSatFix):
        self.drone_lat = msg.latitude
        self.drone_lon = msg.longitude
        self.drone_alt = msg.altitude

    def _cb_hold_active(self, msg: Bool):
        prev = self.hold_active
        self.hold_active = bool(msg.data)
        if self.hold_active and not prev:
            self.hold_started_t = time.time()
            self.confirmed_hits = 0
        elif not self.hold_active and prev:
            self.hold_started_t = None
            self.confirmed_hits = 0

    def _cb_image(self, msg: Image):
        try:
            frame = imgmsg_to_cv(msg)
        except Exception as e:
            self.get_logger().error(f'Image convert error: {e}')
            return

        # D435 đang lắp ngược 180°. Xoay riêng cho AI để YOLO và debug nhìn đúng chiều.
        # Đây chỉ là bản copy trong AL_node, không ảnh hưởng ảnh raw dùng cho ArUco/PID/Kalman.
        if self.rotate_input_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        with self.frame_lock:
            self.latest_frame = frame
            self.latest_stamp = msg.header.stamp
            self.latest_id += 1

    # ---------- output parsing ----------
    def _log_output_meta_once(self, outputs):
        if self.logged_output_meta:
            return
        try:
            with open(self.meta_path, 'w') as f:
                f.write('RKNN output metadata\n')
                for i, out in enumerate(outputs):
                    arr = np.array(out)
                    f.write(f'output[{i}] shape={arr.shape} dtype={arr.dtype}\n')
            self.logged_output_meta = True
            self.get_logger().info(f'Saved output metadata -> {self.meta_path}')
        except Exception as e:
            self.get_logger().warn(f'Write meta failed: {e}')

    def _parse_outputs(self, outputs, orig_w, orig_h, ratio, dw, dh, score_threshold):
        dets = []
        for out in outputs:
            arr = np.array(out)
            if arr.ndim == 0:
                continue
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim > 2:
                last = arr.shape[-1]
                arr = arr.reshape(-1, last)
            elif arr.ndim == 1:
                continue
            if arr.shape[-1] < 5:
                continue

            for row in arr:
                row = row.astype(np.float32)
                if self.single_class:
                    score = float(row[4])
                    if score < score_threshold:
                        continue
                    box4 = row[:4]
                else:
                    obj = float(row[4])
                    cls = row[5:]
                    if cls.size == 0:
                        continue
                    score = obj * float(np.max(cls))
                    if score < score_threshold:
                        continue
                    box4 = row[:4]

                vals = box4.copy()
                if np.max(np.abs(vals)) <= 2.0:
                    # normalized coordinates
                    vals[0] *= max(orig_w, orig_h)
                    vals[1] *= max(orig_w, orig_h)
                    vals[2] *= max(orig_w, orig_h)
                    vals[3] *= max(orig_w, orig_h)

                if self.box_format == 'xywh':
                    cx, cy, bw, bh = vals
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                else:
                    x1, y1, x2, y2 = vals

                x1 = (x1 - dw) / ratio
                y1 = (y1 - dh) / ratio
                x2 = (x2 - dw) / ratio
                y2 = (y2 - dh) / ratio

                box = clip_box([x1, y1, x2, y2], orig_w, orig_h)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                dets.append({'bbox': box, 'score': score, 'label': 'person'})

        return nms_xyxy(dets, self.nms_threshold)

    # ---------- filtering ----------
    def _passes_geometry(self, det, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det['bbox']
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / float(w * h)
        aspect = bw / float(bh)
        margin_x = self.edge_margin_ratio * w
        margin_y = self.edge_margin_ratio * h

        if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
            return False
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return False
        if x1 < margin_x or y1 < margin_y or x2 > (w - margin_x) or y2 > (h - margin_y):
            return False
        return True

    def _passes_center_gate(self, det, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det['bbox']
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        gate_x = self.center_gate_ratio * w
        gate_y = self.center_gate_ratio * h
        return abs(cx - w / 2.0) <= gate_x and abs(cy - h / 2.0) <= gate_y

    def _choose_best(self, dets, frame):
        valid = []
        for d in dets:
            if not self._passes_geometry(d, frame):
                continue
            valid.append(d)
        if not valid:
            return None
        return max(valid, key=lambda d: d['score'])

    def _update_track_hits(self, det, min_hits, mode):
        if det is None:
            if mode == 'CRUISE':
                self.candidate_hits = 0
            else:
                self.confirmed_hits = 0
            self.last_center = None
            return 0

        x1, y1, x2, y2 = det['bbox']
        cur = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
        good_jump = True
        if self.last_center is not None:
            jump = float(np.linalg.norm(cur - self.last_center))
            good_jump = jump <= self.max_center_jump_px
        self.last_center = cur

        if mode == 'CRUISE':
            self.candidate_hits = self.candidate_hits + 1 if good_jump else 1
            return self.candidate_hits
        self.confirmed_hits = self.confirmed_hits + 1 if good_jump else 1
        return self.confirmed_hits

    # ---------- events ----------
    def _publish_person_event(self, event_name, det, frame_id):
        payload = {
            'event': event_name,
            'frame_id': frame_id,
            'score': float(det['score']) if det is not None else None,
            'bbox': [int(v) for v in det['bbox']] if det is not None else None,
            'drone_lat': self.drone_lat,
            'drone_lon': self.drone_lon,
            'drone_alt': self.drone_alt,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.event_pub.publish(msg)

    def _publish_detection_json(self, det, frame_id, state):
        payload = {
            'frame_id': frame_id,
            'state': state,
            'score': float(det['score']) if det is not None else None,
            'bbox': [int(v) for v in det['bbox']] if det is not None else None,
            'drone_lat': self.drone_lat,
            'drone_lon': self.drone_lon,
            'drone_alt': self.drone_alt,
            'hold_active': self.hold_active,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.det_pub.publish(msg)

    def _publish_detected_gps(self, det, confirmed, frame):
        if not confirmed or det is None:
            return
        if self.drone_lat is None or self.drone_lon is None:
            return
        # x?p x? = GPS drone khi target n?m g?n gi?a ?nh
        if not self._passes_center_gate(det, frame):
            return
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.latitude = float(self.drone_lat)
        msg.longitude = float(self.drone_lon)
        msg.altitude = float(self.drone_alt if self.drone_alt is not None else 0.0)
        self.gps_pub.publish(msg)

    # ---------- logging / overlay ----------
    def _maybe_init_writer(self, frame):
        if not self.save_video or self.writer is not None:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (w, h))

    def _save_snapshot(self, frame, confirmed):
        if not self.save_snapshot_on_confirm or not confirmed:
            return
        now = time.time()
        if now - self.last_snapshot_t < self.snapshot_cooldown_sec:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.record_dir, f'snap_{ts}.jpg')
        cv2.imwrite(path, frame)
        self.last_snapshot_t = now

    def _write_csv(self, frame_id, det, state):
        if not self.csv_writer:
            return
        if det is None:
            self.csv_writer.writerow([
                time.time(), frame_id, self.mode_name, state, '', '', '', '', '', '', '', '',
                self.drone_lat, self.drone_lon, self.drone_alt,
            ])
        else:
            x1, y1, x2, y2 = det['bbox']
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            self.csv_writer.writerow([
                time.time(), frame_id, self.mode_name, state, det['score'], state == 'CONFIRMED',
                x1, y1, x2, y2, cx, cy,
                self.drone_lat, self.drone_lon, self.drone_alt,
            ])
        self.csv_file.flush()

    def _draw_overlay(self, frame, det, state):
        vis = frame.copy()
        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        cv2.line(vis, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

        mode_text = f'MODE: {self.mode_name}'
        hold_text = f'HOLD: {self.hold_active}'
        gps_text = f'GPS~ {self.drone_lat}, {self.drone_lon}'
        cv2.putText(vis, mode_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, hold_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, gps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if det is not None:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if state == 'CONFIRMED' else (0, 165, 255) if state == 'CANDIDATE' else (180, 180, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            lines = [
                f'person {det["score"]:.2f} {state}',
                f'GPS~ {self.drone_lat if self.drone_lat is not None else "?"}, {self.drone_lon if self.drone_lon is not None else "?"}',
            ]
            ty = max(18, y1 - 8)
            for i, txt in enumerate(lines):
                cv2.putText(vis, txt, (x1, ty - i * -18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(vis, 'NO PERSON', (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return vis

    # ---------- worker ----------
    def _get_active_mode_params(self):
        if self.hold_active:
            self.mode_name = 'HOLD'
            return self.hold_image_size, self.hold_every_n, self.confirmed_score_threshold
        self.mode_name = 'CRUISE'
        return self.cruise_image_size, self.cruise_every_n, self.candidate_score_threshold

    def _worker_loop(self):
        last_seen_id = -1
        while not self.stop_flag:
            with self.frame_lock:
                if self.latest_frame is None or self.latest_id == last_seen_id:
                    frame = None
                else:
                    frame = self.latest_frame.copy()
                    stamp = self.latest_stamp
                    frame_id = self.latest_id
                    last_seen_id = frame_id

            if frame is None:
                time.sleep(0.01)
                continue

            image_size, every_n, score_thr = self._get_active_mode_params()
            if frame_id % every_n != 0:
                continue

            orig_h, orig_w = frame.shape[:2]
            net_in, ratio, dw, dh = letterbox(frame, image_size)
            net_in = cv2.cvtColor(net_in, cv2.COLOR_BGR2RGB)
            net_in = np.expand_dims(net_in, axis=0)

            try:
                outputs = self.rknn.inference(inputs=[net_in])
            except Exception as e:
                self.get_logger().error(f'RKNN inference failed: {e}')
                time.sleep(0.05)
                continue

            self._log_output_meta_once(outputs)
            dets = self._parse_outputs(outputs, orig_w, orig_h, ratio, dw, dh, score_thr)
            best = self._choose_best(dets, frame)

            state = 'NONE'
            confirmed = False
            if self.hold_active:
                # during hold: verify stronger
                if self.hold_started_t is not None and (time.time() - self.hold_started_t) > self.hold_verify_timeout_sec:
                    self.confirmed_hits = 0
                hits = self._update_track_hits(best, self.confirmed_min_hits, 'HOLD')
                if best is not None and best['score'] >= self.confirmed_score_threshold and self._passes_center_gate(best, frame):
                    if hits >= self.confirmed_min_hits:
                        state = 'CONFIRMED'
                        confirmed = True
                        now = time.time()
                        if now - self.last_confirmed_t > self.confirmed_cooldown_sec:
                            self.last_confirmed_t = now
                            self.get_logger().info(
                                f'CONFIRMED at frame {frame_id} score={best["score"]:.2f}'
                            )
                            self._publish_person_event('confirmed', best, frame_id)
                elif best is not None:
                    state = 'VERIFYING'
            else:
                hits = self._update_track_hits(best, self.candidate_min_hits, 'CRUISE')
                if best is not None and best['score'] >= self.candidate_score_threshold:
                    state = 'TRACK'
                    if hits >= self.candidate_min_hits:
                        state = 'CANDIDATE'
                        now = time.time()
                        if now - self.last_candidate_t > self.candidate_cooldown_sec:
                            self.last_candidate_t = now
                            self.get_logger().info(
                                f'Candidate at frame {frame_id} score={best["score"]:.2f} -> xin drone dung lai'
                            )
                            self._publish_person_event('candidate', best, frame_id)
                else:
                    self.candidate_hits = 0

            vis = self._draw_overlay(frame, best, state)
            self._maybe_init_writer(vis)
            if self.writer is not None:
                self.writer.write(vis)
            self._save_snapshot(vis, confirmed)
            self._publish_detection_json(best, frame_id, state)
            self._publish_detected_gps(best, confirmed, frame)
            self._write_csv(frame_id, best, state)

            if not self.stop_flag and rclpy.ok():
                try:
                    self.debug_pub.publish(cv_to_imgmsg(vis, stamp))
                except Exception:
                    pass

    # ---------- shutdown ----------
    def destroy_node(self):
        self.stop_flag = True
        try:
            self.worker.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.writer is not None:
                self.writer.release()
        except Exception:
            pass
        try:
            if self.csv_file is not None:
                self.csv_file.close()
        except Exception:
            pass
        try:
            self.rknn.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ALNode()
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


if __name__ == '__main__':
    main()