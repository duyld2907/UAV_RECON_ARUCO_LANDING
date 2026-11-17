import cv2
import os
import argparse


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def make_panel(img, title, panel_w=360, panel_h=300):
    img = to_bgr(img)

    h, w = img.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = 255 * (cv2.UMat(panel_h + 45, panel_w, cv2.CV_8UC3).get())

    x = (panel_w - new_w) // 2
    y = 45 + (panel_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    cv2.putText(
        canvas,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Extract ArUco image processing pipeline from video log.")
    parser.add_argument("--video", required=True, help="Path to debug video .avi")
    parser.add_argument("--time", type=float, default=80.0, help="Time in seconds to extract frame")
    parser.add_argument("--out", default="/home/orangepi/UAV_RECON_ARUCO_LANDING/aruco_pipeline_output", help="Output folder")

    parser.add_argument("--crop_left", type=float, default=0.40)
    parser.add_argument("--crop_right", type=float, default=1.00)
    parser.add_argument("--crop_top", type=float, default=0.00)
    parser.add_argument("--crop_bottom", type=float, default=1.00)

    args = parser.parse_args()
    ensure_dir(args.out)

    if not os.path.exists(args.video):
        raise RuntimeError(f"Không tìm thấy video: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {args.video}")
    print(f"FPS: {fps:.2f}")
    print(f"Thời lượng xấp xỉ: {duration:.2f} giây")
    print(f"Lấy frame tại: {args.time:.2f} giây")

    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Không đọc được frame tại thời điểm đã chọn. Hãy chỉnh lại --time.")

    h, w = frame.shape[:2]

    x1 = int(w * args.crop_left)
    x2 = int(w * args.crop_right)
    y1 = int(h * args.crop_top)
    y2 = int(h * args.crop_bottom)

    frame_crop = frame[y1:y2, x1:x2]

    if frame_crop.size == 0:
        raise RuntimeError("Vùng crop bị rỗng. Hãy chỉnh lại crop_left/right/top/bottom.")

    img_original = frame_crop.copy()

    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        23,
        7
    )

    binary = adaptive.copy()

    cv2.imwrite(os.path.join(args.out, "01_anh_camera_goc.png"), img_original)
    cv2.imwrite(os.path.join(args.out, "02_anh_xam.png"), gray)
    cv2.imwrite(os.path.join(args.out, "03_CLAHE_Gaussian.png"), blur)
    cv2.imwrite(os.path.join(args.out, "04_adaptive_threshold.png"), adaptive)
    cv2.imwrite(os.path.join(args.out, "05_anh_nhi_phan.png"), binary)

    panels = [
        make_panel(img_original, "(a) Anh camera"),
        make_panel(gray, "(b) Anh xam"),
        make_panel(blur, "(c) CLAHE/Gaussian"),
        make_panel(adaptive, "(d) Adaptive threshold"),
        make_panel(binary, "(e) Anh nhi phan"),
    ]

    montage = cv2.hconcat(panels)
    cv2.imwrite(os.path.join(args.out, "chuoi_xu_ly_aruco.png"), montage)

    print("Da xuat xong anh vao thu muc:")
    print(args.out)
    print("File chinh de chen Word:")
    print(os.path.join(args.out, "chuoi_xu_ly_aruco.png"))


if __name__ == "__main__":
    main()
