import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="calib_images_d435_640")
    parser.add_argument("--max-images", type=int, default=35)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    debug_dir = Path(args.output_dir + "_debug")
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    pattern_size = (args.cols, args.rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    captured = 0
    frame_count = 0
    last_save = 0.0

    print("=== HEADLESS CHESSBOARD CAPTURE ===")
    print(f"Camera index: {args.camera_index}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Pattern inner corners: cols={args.cols}, rows={args.rows}")
    print(f"Output: {save_dir}")
    print("Di chuyển bàn cờ chậm qua nhiều góc. Ctrl+C để dừng.")
    print("Khi detect được chessboard, ảnh sẽ tự lưu.")

    try:
        while captured < args.max_images:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Cannot read frame")
                time.sleep(0.2)
                continue

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

            now = time.time()

            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if now - last_save >= args.interval:
                    img_path = save_dir / f"img_{captured:03d}.jpg"
                    dbg_path = debug_dir / f"img_{captured:03d}_debug.jpg"

                    cv2.imwrite(str(img_path), frame)

                    dbg = frame.copy()
                    cv2.drawChessboardCorners(dbg, pattern_size, corners2, found)
                    cv2.putText(dbg, f"FOUND saved {captured}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imwrite(str(dbg_path), dbg)

                    captured += 1
                    last_save = now
                    print(f"[OK] Saved {captured}/{args.max_images}: {img_path}")
            else:
                if frame_count % 30 == 0:
                    print(f"[NOT FOUND] saved={captured}/{args.max_images}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    cap.release()
    print(f"Done. Saved {captured} images.")


if __name__ == "__main__":
    main()
