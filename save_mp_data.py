# save_mp_data.py
# ===============
# Fast engine body capture: video -> body NPZ using Google MediaPipe
# (Pose Landmarker + Hand Landmarker). CPU friendly, ~60 MB of models,
# runs on Windows, Linux and macOS.
#
# Command line and output contract match save_mhr_data.py (the Pro / SAM 3D
# Body engine) so the add-on can run either one: same flags, same
# "[PROGRESS n] msg" / "[WARNING] ..." lines, and an NPZ that npz_to_bvh.py
# consumes through the same cleanup pipeline. See pipeline/mp_skeleton.py
# for how MediaPipe landmarks are mapped onto that pipeline's skeleton.
#
# Part of Open Mocap, GPL-3.0-or-later.

import argparse
import os as _os
_os.environ.setdefault("GLOG_minloglevel", "2")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
CKPT_DIR = SCRIPT_DIR / "checkpoints"
MEDIAPIPE_DIR = CKPT_DIR / "mediapipe"

from pipeline.body_preview import emit_progress, open_video, run_preview  # noqa: E402
from pipeline import mp_skeleton  # noqa: E402

MODEL_URLS = {
    "pose_landmarker_heavy.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
}

# MediaPipe pose indices used for hand crops
L_ELB, R_ELB, L_WRI, R_WRI = 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20


def _model_path(name):
    """Local model path; downloads once if missing (the installer normally
    fetches these so capture itself runs offline)."""
    p = MEDIAPIPE_DIR / name
    if p.is_file():
        return str(p)
    MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
    url = MODEL_URLS[name]
    print(f"Downloading {name} ...")
    import urllib.request
    tmp = str(p) + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, p)
    return str(p)


def _mp():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        return mp, vision, BaseOptions
    except Exception as e:
        print(f"ERROR: mediapipe is not installed ({e}). Run the installer.")
        sys.exit(1)


def _delegate(BaseOptions):
    # MediaPipe's Python GPU delegate only exists on Linux/macOS; CPU is the
    # reliable default everywhere and is plenty fast for these models.
    want_gpu = os.environ.get("OPENMOCAP_MP_GPU", "0") == "1"
    return BaseOptions.Delegate.GPU if want_gpu else BaseOptions.Delegate.CPU


def make_pose(video_mode=True, num_poses=2, model="heavy"):
    mp, vision, BaseOptions = _mp()
    name = "pose_landmarker_%s.task" % model
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model_path(name), delegate=_delegate(BaseOptions)),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_poses=num_poses, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
    return vision.PoseLandmarker.create_from_options(opts)


def make_hand():
    mp, vision, BaseOptions = _mp()
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model_path("hand_landmarker.task"),
                                 delegate=_delegate(BaseOptions)),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35)
    return vision.HandLandmarker.create_from_options(opts)


def mp_image(rgb):
    mp, _, _ = _mp()
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def _pose_box(lms, w, h):
    xs = np.array([l.x for l in lms]) * w
    ys = np.array([l.y for l in lms]) * h
    pad_x = 0.08 * (xs.max() - xs.min())
    pad_y = 0.08 * (ys.max() - ys.min())
    return [float(xs.min() - pad_x), float(ys.min() - pad_y),
            float(xs.max() + pad_x), float(ys.max() + pad_y)]


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PoseDetector:
    """Duck-types the YOLO detector interface run_preview() expects."""

    def __init__(self):
        self.pose = make_pose(video_mode=False, num_poses=3, model="full")

    def run_human_detection(self, frame_bgr, det_cat_id=0, bbox_thr=0.5,
                            default_to_full_image=False):
        h, w = frame_bgr.shape[:2]
        res = self.pose.detect(mp_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        boxes = [_pose_box(l, w, h) for l in (res.pose_landmarks or [])]
        if not boxes and default_to_full_image:
            boxes = [[0, 0, w, h]]
        return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _square_crop(rgb, cx, cy, size, out_size):
    h, w = rgb.shape[:2]
    size = max(16.0, float(size))
    x0, y0 = cx - size / 2.0, cy - size / 2.0
    scale = out_size / size
    m = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]], dtype=np.float32)
    crop = cv2.warpAffine(rgb, m, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return crop, (x0, y0, scale)


def _gate_detections(pvis, pvalid, fps, step=1, min_torso_vis=0.6, min_run_s=0.5,
                     pimg=None, frame_h=0, min_height_frac=0.15):
    """Reject frames that are not a real, visible person.

    MediaPipe's tracker sometimes latches onto furniture or keeps a stale
    pose after the subject leaves. A frame stays valid only when the torso
    (shoulders + hips) is confidently visible, and it belongs to a run of
    valid frames at least `min_run_s` long, so one-off flickers on an empty
    frame are dropped and get interpolated / held like any other miss.
    """
    ok = pvalid & (pvis[:, [11, 12, 23, 24]].mean(axis=1) >= min_torso_vis)
    if pimg is not None and frame_h > 0:
        # Tiny "people" are almost always furniture / texture false hits.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            h = np.nanmax(pimg[:, :, 1], axis=1) - np.nanmin(pimg[:, :, 1], axis=1)
        ok &= np.nan_to_num(h) >= min_height_frac * frame_h
    min_run = max(2, int(math.ceil(min_run_s * fps / max(1, step))))
    sampled = np.zeros(len(ok), bool)
    sampled[::max(1, step)] = True
    sampled[-1] = True
    idx = np.flatnonzero(sampled)
    seq = ok[idx]
    out = seq.copy()
    i = 0
    while i < len(seq):
        if seq[i]:
            j = i
            while j < len(seq) and seq[j]:
                j += 1
            if j - i < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    res = np.zeros(len(ok), bool)
    res[idx] = out
    return res


def focal_px_for(width, height, focal_mm):
    """Pixel focal length from a 35mm-equivalent focal length (long edge = 36mm)."""
    return float(focal_mm) * max(width, height) / 36.0


def main():
    parser = argparse.ArgumentParser(description="MediaPipe body capture -> NPZ (Open Mocap Fast engine)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="mp_frames.npz")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-step", type=int, default=10)
    parser.add_argument("--boxes-out", default="")
    parser.add_argument("--no-fov-cache", action="store_true",
                        help="Accepted for CLI compatibility (MediaPipe has no FOV model).")
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--focal-mm", type=float, default=0.0,
                        help="35mm-equivalent focal length; 0 = assume a typical "
                             "phone / webcam lens (OPENMOCAP_DEFAULT_FOCAL_MM, 28mm)")
    parser.add_argument("--capture-skip", type=int, default=0)
    parser.add_argument("--pose-model", default="heavy", choices=["heavy", "full"])
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    if args.preview:
        output = Path(args.output) if args.output else Path("preview.mp4")
        emit_progress(2, "Loading person detector (MediaPipe)")
        detector = PoseDetector()
        emit_progress(30, "Starting body preview")
        run_preview(detector, video, output, step=max(1, args.preview_step),
                    start_frame=args.start_frame, max_frames=args.max_frames,
                    boxes_out=Path(args.boxes_out) if args.boxes_out else None)
        detector.pose.close()
        emit_progress(100, "Body preview complete")
        return

    emit_progress(2, "Loading MediaPipe models")
    pose = make_pose(video_mode=True, num_poses=2, model=args.pose_model)
    hand = None if args.no_hands else make_hand()

    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (0 < fps < 1000):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames > 0:
        total = min(args.start_frame + args.max_frames, total)
    start = args.start_frame
    n = max(0, total - start)
    if n == 0:
        print("ERROR: no frames in the requested range")
        sys.exit(1)
    focal_mm = args.focal_mm if args.focal_mm > 0 else float(
        os.environ.get("OPENMOCAP_DEFAULT_FOCAL_MM", "28"))
    focal_px = focal_px_for(W, H, focal_mm)
    print(f"  {n} frames at {fps:.1f} FPS, {W}x{H}, focal {focal_mm:.1f}mm (35mm eq) = {focal_px:.0f}px")

    pw = np.full((n, 33, 3), np.nan)
    pimg = np.full((n, 33, 2), np.nan)
    pvis = np.zeros((n, 33))
    pvalid = np.zeros(n, bool)
    hw = np.full((n, 2, 21, 3), np.nan)
    himg = np.full((n, 2, 21, 2), np.nan)
    hvalid = np.zeros((n, 2), bool)

    step = max(1, args.capture_skip + 1)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    emit_progress(5, "Starting body tracking")
    multi = 0
    prev_box = None
    t0 = time.time()
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            n = i
            break
        if step > 1 and i % step != 0 and i != n - 1:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ts = int(round((start + i) * 1000.0 / fps))
        res = pose.detect_for_video(mp_image(rgb), ts)
        if res.pose_landmarks:
            if len(res.pose_landmarks) > 1:
                multi += 1
                boxes = [_pose_box(l, W, H) for l in res.pose_landmarks]
                if prev_box is not None:
                    k = int(np.argmax([_iou(b, prev_box) for b in boxes]))
                else:
                    k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            else:
                k = 0
            lm = res.pose_landmarks[k]
            prev_box = _pose_box(lm, W, H)
            pvalid[i] = True
            pimg[i] = [(l.x * W, l.y * H) for l in lm]
            pvis[i] = [min(l.visibility if l.visibility is not None else 1.0,
                           l.presence if l.presence is not None else 1.0) for l in lm]
            pw[i] = [(l.x, l.y, l.z) for l in res.pose_world_landmarks[k]]

            if hand is not None:
                for side, (wri, elb, idx_f, pinky) in enumerate(((L_WRI, L_ELB, L_INDEX, L_PINKY),
                                                                 (R_WRI, R_ELB, R_INDEX, R_PINKY))):
                    if pvis[i, wri] < 0.2:
                        continue
                    px = pimg[i]
                    fore = float(np.linalg.norm(px[wri] - px[elb]))
                    tip = (px[idx_f] + px[pinky]) / 2.0
                    hdir = tip - px[wri]
                    c = px[wri] + hdir * 0.9
                    size = max(fore * 1.6, float(np.linalg.norm(hdir)) * 5.0, 0.06 * max(W, H))
                    crop, (x0, y0, sc) = _square_crop(rgb, c[0], c[1], size, 256)
                    hres = hand.detect(mp_image(crop))
                    if not hres.hand_landmarks:
                        continue
                    hl = hres.hand_landmarks[0]
                    pts = np.array([(l.x * 256 / sc + x0, l.y * 256 / sc + y0) for l in hl])
                    if np.linalg.norm(pts[0] - px[wri]) > max(fore, 1.0) * 0.9:
                        continue  # matched the other hand / a false positive
                    himg[i, side] = pts
                    hw[i, side] = [(l.x, l.y, l.z) for l in hres.hand_world_landmarks[0]]
                    hvalid[i, side] = True

        if i % 5 == 0 or i == n - 1:
            emit_progress(5 + (i + 1) / n * 85, f"Body frame {i + 1}/{n}")
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  Frame {start + i}/{start + n}  {'OK' if pvalid[i] else 'no person'}  "
                  f"({el / (i + 1):.3f}s/frame)")
    cap.release()
    pose.close()
    if hand is not None:
        hand.close()

    pw, pimg, pvis, pvalid = pw[:n], pimg[:n], pvis[:n], pvalid[:n]
    hw, himg, hvalid = hw[:n], himg[:n], hvalid[:n]
    pvalid = _gate_detections(pvis, pvalid, fps, step, pimg=pimg, frame_h=H)
    hvalid &= pvalid[:, None]
    # Like the Pro engine, frames before the subject appears and after they
    # leave are not part of the take. Interior gaps stay (interpolated).
    good = np.flatnonzero(pvalid)
    if len(good):
        a, b = int(good[0]), int(good[-1]) + 1
        if a > 0 or b < n:
            print(f"  Trimmed to frames {start + a}..{start + b - 1} (subject visible)")
        pw, pimg, pvis, pvalid = pw[a:b], pimg[a:b], pvis[a:b], pvalid[a:b]
        hw, himg, hvalid = hw[a:b], himg[a:b], hvalid[a:b]
        start, n = start + a, b - a
    n_good = int(pvalid.sum())
    if n_good == 0:
        print("\nERROR: No frames with a detected person!")
        sys.exit(1)
    if step == 1 and n_good < n * 0.7:
        print(f"[WARNING] {100 - n_good * 100 // n}% of frames had no person detected")
        print("  The subject may be partly out of frame or too small.")
    if multi > n_good * 0.05:
        print("[WARNING] Multiple people in frame; tracked the most prominent")

    emit_progress(92, "Solving skeleton")
    arrays = mp_skeleton.build_body_npz_arrays(pw, pvis, pvalid, pimg, hw, himg, hvalid,
                                               W, H, focal_px)
    arrays["focal_length"] = np.full(n, focal_px, dtype=np.float32)
    arrays["frame_indices"] = np.arange(start, start + n)
    arrays["fps"] = np.array(fps)
    arrays["engine"] = np.array("mediapipe")
    arrays["pose_valid"] = pvalid
    arrays["hand_valid"] = hvalid
    arrays["openmocap_npz_version"] = np.array(1)
    arrays.pop("cam_t_valid", None)
    # Fields the SAM engine writes that this engine has no equivalent for.
    arrays["body_pose_params"] = np.zeros((n, 133), np.float32)
    arrays["global_rot_euler"] = np.zeros((n, 3), np.float32)
    arrays["hand_pose_params"] = np.zeros((n, 108), np.float32)

    emit_progress(98, "Saving results")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **arrays)
    el = time.time() - t0
    print(f"\nProcessed {n} frames ({n_good} with a person, hands L/R "
          f"{int(hvalid[:, 0].sum())}/{int(hvalid[:, 1].sum())}) in {el:.1f}s")
    print(f"Saved to {out}")
    emit_progress(100, "Body tracking complete")


if __name__ == "__main__":
    main()
