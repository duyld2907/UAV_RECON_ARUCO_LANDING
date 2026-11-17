#!/usr/bin/env python3

import json
import math
import time
import unittest


LARGE_ID = 150
SMALL_ID = 40
LARGE_MIN_SIZE_PX = 70.0
LARGE_MAX_ERR_X_PX = 200.0
LARGE_MAX_ERR_Y_PX = 160.0
SMALL_MIN_SIZE_PX = 22.0
SMALL_MAX_ERR_X_PX = 180.0
SMALL_MAX_ERR_Y_PX = 160.0
MAX_AGE_S = 0.50
MIXED_ALT_M = 3.0
SMALL_ONLY_ALT_M = 1.8


def finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def best_by_id(payload, now):
    data = json.loads(payload)
    detections = data.get("detections", [])
    out = {}
    for raw in detections:
        marker_id = int(raw.get("marker_id", raw.get("id", -1)))
        if marker_id not in (LARGE_ID, SMALL_ID):
            continue
        err_x = float(raw.get("err_x", 0.0))
        err_y = float(raw.get("err_y", 0.0))
        size_px = float(raw.get("size_px", raw.get("avg_side_px", 0.0)))
        stamp = float(raw.get("stamp", data.get("stamp", now)))
        if not (finite(err_x) and finite(err_y) and finite(size_px) and finite(stamp)):
            continue
        age = max(0.0, now - stamp)
        score = abs(err_x) + abs(err_y)
        if marker_id not in out or score < out[marker_id]["score"]:
            out[marker_id] = {
                "marker_id": marker_id,
                "err_x": err_x,
                "err_y": err_y,
                "size_px": size_px,
                "age": age,
                "score": score,
            }
    return out


def alt_band(alt_m):
    if alt_m is None or alt_m > MIXED_ALT_M:
        return "BIG_ONLY"
    if alt_m > SMALL_ONLY_ALT_M:
        return "MIXED"
    return "SMALL_ONLY"


def usable_large(det):
    return (
        det is not None
        and det["age"] <= MAX_AGE_S
        and det["size_px"] >= LARGE_MIN_SIZE_PX
        and abs(det["err_x"]) <= LARGE_MAX_ERR_X_PX
        and abs(det["err_y"]) <= LARGE_MAX_ERR_Y_PX
    )


def usable_small(det):
    return (
        det is not None
        and det["age"] <= MAX_AGE_S
        and det["size_px"] >= SMALL_MIN_SIZE_PX
        and abs(det["err_x"]) <= SMALL_MAX_ERR_X_PX
        and abs(det["err_y"]) <= SMALL_MAX_ERR_Y_PX
    )


def decide(payload, alt_m, now):
    detections = best_by_id(payload, now)
    large = detections.get(LARGE_ID)
    small = detections.get(SMALL_ID)
    band = alt_band(alt_m)

    if any(det["age"] > MAX_AGE_S for det in detections.values()):
        return {"mode": "hold", "reason": "stale_marker", "alt_band": band}

    if band == "BIG_ONLY":
        if usable_large(large):
            return {"mode": "track", "marker_id": LARGE_ID, "alt_band": band}
        if large is not None and large["size_px"] < LARGE_MIN_SIZE_PX:
            return {"mode": "hold", "reason": "large_too_small", "alt_band": band}
        return {"mode": "hold", "reason": "large_too_far", "alt_band": band}

    if band == "MIXED":
        if usable_large(large):
            return {"mode": "track", "marker_id": LARGE_ID, "alt_band": band}
        if usable_small(small):
            return {"mode": "track", "marker_id": SMALL_ID, "alt_band": band}
        if small is not None and small["size_px"] < SMALL_MIN_SIZE_PX:
            return {"mode": "hold", "reason": "small_too_small", "alt_band": band}
        return {"mode": "hold", "reason": "no_marker", "alt_band": band}

    if usable_small(small):
        return {"mode": "track", "marker_id": SMALL_ID, "alt_band": band}
    if small is not None and small["size_px"] < SMALL_MIN_SIZE_PX:
        return {"mode": "hold", "reason": "small_too_small", "alt_band": band}
    return {"mode": "hold", "reason": "no_marker", "alt_band": band}


def clamp_landing_cmd(cmd):
    out = {}
    for key in ("vx", "vy", "vz", "yaw_rate"):
        v = cmd.get(key, 0.0)
        if not finite(v):
            raise ValueError(f"invalid {key}")
        out[key] = float(v)
    out["vx"] = max(-0.45, min(0.45, out["vx"]))
    out["vy"] = max(-0.45, min(0.45, out["vy"]))
    out["vz"] = max(0.0, min(0.15, out["vz"]))
    out["yaw_rate"] = 0.0
    return out


def payload(marker_id, size_px, err_x=0.0, err_y=0.0, stamp=None):
    now = time.time() if stamp is None else stamp
    return json.dumps({
        "stamp": now,
        "detections": [{
            "marker_id": marker_id,
            "size_px": size_px,
            "err_x": err_x,
            "err_y": err_y,
            "stamp": stamp if stamp is not None else now,
        }],
    })


class LandingPipelineOfflinePolicyTest(unittest.TestCase):
    def test_id150_good_tracks(self):
        now = time.time()
        decision = decide(payload(LARGE_ID, 90.0, 40.0, -25.0, now), 3.5, now)
        self.assertEqual(decision["mode"], "track")
        self.assertEqual(decision["marker_id"], LARGE_ID)

    def test_id150_too_far_or_small_holds(self):
        now = time.time()
        self.assertEqual(decide(payload(LARGE_ID, 40.0, 0.0, 0.0, now), 3.5, now)["mode"], "hold")
        self.assertEqual(decide(payload(LARGE_ID, 90.0, 260.0, 0.0, now), 3.5, now)["mode"], "hold")

    def test_id40_12px_not_used(self):
        now = time.time()
        decision = decide(payload(SMALL_ID, 12.0, 0.0, 0.0, now), 1.6, now)
        self.assertEqual(decision["mode"], "hold")
        self.assertIn("small_too_small", decision["reason"])

    def test_id40_24px_near_2m_tracks_in_mixed(self):
        now = time.time()
        decision = decide(payload(SMALL_ID, 24.0, 20.0, 10.0, now), 2.0, now)
        self.assertEqual(decision["mode"], "track")
        self.assertEqual(decision["marker_id"], SMALL_ID)
        self.assertEqual(decision["alt_band"], "MIXED")

    def test_stale_detection_holds(self):
        now = time.time()
        decision = decide(payload(LARGE_ID, 90.0, 0.0, 0.0, now - 1.0), 3.5, now)
        self.assertEqual(decision["mode"], "hold")
        self.assertEqual(decision["reason"], "stale_marker")

    def test_nan_inf_command_rejected_or_clamped(self):
        with self.assertRaises(ValueError):
            clamp_landing_cmd({"vx": math.nan, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0})
        with self.assertRaises(ValueError):
            clamp_landing_cmd({"vx": 0.0, "vy": math.inf, "vz": 0.0, "yaw_rate": 0.0})
        cmd = clamp_landing_cmd({"vx": 1.0, "vy": -1.0, "vz": 1.0, "yaw_rate": 1.0})
        self.assertEqual(cmd, {"vx": 0.45, "vy": -0.45, "vz": 0.15, "yaw_rate": 0.0})


if __name__ == "__main__":
    unittest.main()
