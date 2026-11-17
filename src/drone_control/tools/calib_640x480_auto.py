#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calib camera 640x480 cho bàn cờ 9 ô ngang x 7 ô dọc, cạnh ô 3 cm.
OpenCV dùng số GÓC TRONG: 8 x 6.
File lưu đè đúng chỗ code ArUco đang đọc:
    /ws/src/drone_control/config/camera_calib.npz
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np


def open_camera(index: int, width: int, height: int, fps: float):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(20):
        cap.read()

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"Camera opened: {real_w}x{real_h}@{real_fps:.1f}, FOURCC={fourcc_text}")
    return cap


def make_object_points(cols: int, rows: int, square_size_m: float):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def enough_motion(prev_corners, corners, min_motion_px: float) -> bool:
    if prev_corners is None:
        return True
    diff = np.linalg.norm(corners.reshape(-1, 2) - prev_corners.reshape(-1, 2), axis=1)
    return float(np.mean(diff)) >= min_motion_px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)

    # Bàn cờ của bạn: 9 ô ngang x 7 ô dọc, cạnh 3 cm.
    # Số góc trong = 8 x 6.
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square", type=float, default=0.03)

    parser.add_argument("--samples", type=int, default=35)
    parser.add_argument("--min-motion-px", type=float, default=18.0)
    parser.add_argument("--save-path", type=str, default="/ws/src/drone_control/config/camera_calib.npz")
    parser.add_argument("--debug-dir", type=str, default="/ws/src/drone_control/calib_640_debug")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    debug_dir = Path(args.debug_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CALIB CAMERA 640x480 - BOARD 9x7 SQUARE 3CM")
    print(f"Image size    : {args.width}x{args.height}")
    print(f"Board squares : 9 ngang x 7 doc")
    print(f"Inner corners : {args.cols} x {args.rows}")
    print(f"Square size   : {args.square} m")
    print(f"Save path     : {save_path}")
    print("=" * 72)

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    pattern_size = (args.cols, args.rows)
    objp = make_object_points(args.cols, args.rows, args.square)

    objpoints = []
    imgpoints = []
    prev_corners = None
    last_capture_t = 0.0
    frame_count = 0
    found_count = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("Dua ban co vao camera: giua, 4 goc, nghieng nhe, xa/gan.")
    print("Script tu chup mau. Ctrl+C neu muon dung som.")

    try:
        while len(objpoints) < args.samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARN: cannot read frame")
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            now = time.time()
            if found:
                found_count += 1
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if (now - last_capture_t) >= 0.45 and enough_motion(prev_corners, corners_refined, args.min_motion_px):
                    objpoints.append(objp.copy())
                    imgpoints.append(corners_refined.copy())
                    prev_corners = corners_refined.copy()
                    last_capture_t = now

                    idx = len(objpoints)
                    vis = frame.copy()
                    cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
                    cv2.imwrite(str(debug_dir / f"calib_{idx:03d}.jpg"), vis)
                    print(f"Captured {idx:02d}/{args.samples}")

            if frame_count % 60 == 0:
                print(f"frames={frame_count}, found_frames={found_count}, captured={len(objpoints)}/{args.samples}")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()

    if len(objpoints) < 12:
        raise RuntimeError(f"Not enough samples: {len(objpoints)}. Can toi thieu 12, nen 30-45.")

    print("Calibrating...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (args.width, args.height), None, None
    )

    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total_error += err * err
        total_points += len(projected)
    mean_error = float(np.sqrt(total_error / max(total_points, 1)))

    np.savez(
        save_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=np.array(args.width, dtype=np.int32),
        image_height=np.array(args.height, dtype=np.int32),
        cols=np.array(args.cols, dtype=np.int32),
        rows=np.array(args.rows, dtype=np.int32),
        square_size=np.array(args.square, dtype=np.float32),
        rms=np.array(rms, dtype=np.float64),
        mean_reprojection_error_px=np.array(mean_error, dtype=np.float64),
    )

    print("=" * 72)
    print("DONE")
    print(f"saved: {save_path}")
    print(f"RMS: {rms:.4f}")
    print(f"mean reprojection error: {mean_error:.4f} px")
    print("camera_matrix=")
    print(camera_matrix)
    print("dist_coeffs=")
    print(dist_coeffs)
    print(f"image_width={args.width}")
    print(f"image_height={args.height}")
    print(f"cols={args.cols}, rows={args.rows}, square_size={args.square}")
    print("=" * 72)


if __name__ == "__main__":
    main()
