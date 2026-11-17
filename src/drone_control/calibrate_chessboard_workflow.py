import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def make_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def capture_images(args):
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    pattern_size = (args.cols, args.rows)
    captured = 0
    frame_id = 0

    print("=== CAPTURE MODE ===")
    print("q: thoat")
    print("s: luu frame hien tai neu detect duoc chessboard")
    print("Nen chup 15-25 anh o nhieu goc khac nhau.")

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame tu camera.")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, corners2, found)
            cv2.putText(display, "FOUND - press s to save", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(display, "NOT FOUND", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(display, f"saved: {captured}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("capture_chessboard", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if found:
                out_path = save_dir / f"img_{frame_id:03d}.jpg"
                cv2.imwrite(str(out_path), frame)
                captured += 1
                frame_id += 1
                print(f"Saved: {out_path}")
            else:
                print("Chessboard chua detect duoc, khong luu.")

    cap.release()
    cv2.destroyAllWindows()


def calibrate_from_images(args):
    image_dir = Path(args.image_dir)
    image_paths = sorted(glob.glob(str(image_dir / "*.jpg"))) + sorted(glob.glob(str(image_dir / "*.png")))
    if not image_paths:
        raise RuntimeError(f"Khong tim thay anh trong: {image_dir}")

    pattern_size = (args.cols, args.rows)
    square_size = args.square_size
    objp = make_object_points(pattern_size, square_size)

    objpoints = []
    imgpoints = []
    image_size = None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    good = 0
    bad = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Skip unreadable image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            bad += 1
            print(f"Not found: {path}")
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        good += 1
        print(f"OK: {path}")

    if good < 10:
        raise RuntimeError(f"Chi co {good} anh tot. Nen co it nhat 10 anh tot de calib.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    out = {
        "pattern_inner_corners": {
            "cols": args.cols,
            "rows": args.rows
        },
        "square_size": square_size,
        "square_unit": args.square_unit,
        "num_input_images": len(image_paths),
        "num_good_images": good,
        "num_bad_images": bad,
        "rms": float(ret),
        "mean_reprojection_error": float(mean_error),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }

    out_json = Path(args.output_json)
    out_npz = Path(args.output_npz)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    np.savez(
        out_npz,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        cols=args.cols,
        rows=args.rows,
        square_size=square_size,
    )

    print("\n=== CALIBRATION DONE ===")
    print(f"Good images: {good}/{len(image_paths)}")
    print(f"RMS: {ret:.6f}")
    print(f"Mean reprojection error: {mean_error:.6f}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved NPZ : {out_npz}")
    print("Camera matrix:\n", camera_matrix)
    print("Dist coeffs:\n", dist_coeffs)


def main():
    parser = argparse.ArgumentParser(description="Chessboard camera calibration workflow")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_cap = sub.add_parser("capture", help="Capture chessboard images from webcam")
    p_cap.add_argument("--camera-index", type=int, default=0)
    p_cap.add_argument("--width", type=int, default=1280)
    p_cap.add_argument("--height", type=int, default=720)
    p_cap.add_argument("--cols", type=int, default=7, help="Number of INNER corners along width")
    p_cap.add_argument("--rows", type=int, default=7, help="Number of INNER corners along height")
    p_cap.add_argument("--output-dir", type=str, default="calib_images")

    p_cal = sub.add_parser("calibrate", help="Calibrate from saved images")
    p_cal.add_argument("--image-dir", type=str, default="calib_images")
    p_cal.add_argument("--cols", type=int, default=7, help="Number of INNER corners along width")
    p_cal.add_argument("--rows", type=int, default=7, help="Number of INNER corners along height")
    p_cal.add_argument("--square-size", type=float, required=True,
                       help="Real square size, e.g. 0.017 for 17 mm if using meters")
    p_cal.add_argument("--square-unit", type=str, default="meter")
    p_cal.add_argument("--output-json", type=str, default="calib_output/camera_calib.json")
    p_cal.add_argument("--output-npz", type=str, default="calib_output/camera_calib.npz")

    args = parser.parse_args()

    if args.mode == "capture":
        capture_images(args)
    elif args.mode == "calibrate":
        calibrate_from_images(args)


if __name__ == "__main__":
    main()
