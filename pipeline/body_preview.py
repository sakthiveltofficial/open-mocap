# body_preview.py
# ===============
# Torch-free body-capture helpers shared by both capture engines
# (save_mhr_data.py for Pro / SAM 3D Body and save_mp_data.py for
# Fast / MediaPipe): progress output, video opening, person-box
# selection and the stepped detection preview renderer.
#
# Moved verbatim out of save_mhr_data.py (BlendCap 1.0.5, Arcomade,
# GPL-3.0-or-later) so the MediaPipe engine can reuse them without
# importing torch.

import bisect
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

def emit_progress(pct, msg=""):
    print(f"[PROGRESS {int(pct)}] {msg}", flush=True)


def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap
    print(f"ERROR: Cannot open {video_path}")
    sys.exit(1)



def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    if a.size < 4 or b.size < 4:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_area(b):
    if b is None:
        return 0.0
    b = np.asarray(b, dtype=np.float64).flatten()
    if b.size < 4:
        return 0.0
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def _pick_person(result, prev_bbox):
    if len(result) == 1:
        return result[0]
    if prev_bbox is None:
        return max(result, key=lambda p: _bbox_area(p.get("bbox")))
    return max(result, key=lambda p: _bbox_iou(p.get("bbox"), prev_bbox))



def _pick_box(boxes, prev_bbox):
    """Pick one person box from a YOLO [N,4] result: prefer IoU continuity
    with the previous frame, else fall back to the largest-area box."""
    if boxes is None or len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]
    if prev_bbox is not None:
        best = max(boxes, key=lambda b: _bbox_iou(b, prev_bbox))
        if _bbox_iou(best, prev_bbox) > 0:
            return best
    return max(boxes, key=_bbox_area)


def _preview_payload_at(idx, sample_indices, payloads):
    """Overlay payload for frame `idx` from sparse per-sample detections.

    sample_indices: sorted video-frame indices that were actually checked;
    payloads[i] is the drawable result at sample_indices[i], or None on a
    detector miss. Between two DETECTED samples the payload interpolates;
    around a missed sample the nearest detected payload holds up to the
    midpoint of the gap, then frames read as a miss. Returns (a, b, t):
    lerp a->b by t when b is not None, use `a` as-is when b is None, and
    a miss when a is None.
    """
    n = len(sample_indices)
    if n == 0:
        return None, None, 0.0
    pos = bisect.bisect_right(sample_indices, idx)
    if pos == 0:
        return payloads[0], None, 0.0
    if pos >= n:
        return payloads[n - 1], None, 0.0
    i0, i1 = pos - 1, pos
    a, b = payloads[i0], payloads[i1]
    f0, f1 = sample_indices[i0], sample_indices[i1]
    if a is not None and b is not None:
        return a, b, (idx - f0) / float(f1 - f0)
    # One side missed: the nearer sample's state wins, held statically.
    if idx - f0 <= f1 - idx:
        return a, None, 0.0
    return b, None, 0.0


def run_preview(detector, video_path, output_path, step=10,
                start_frame=0, max_frames=0, boxes_out=None):
    """Stepped body bounding-box preview -> MP4, in two passes.

    With `boxes_out` set (the combined body+face preview), the render pass
    is skipped entirely: pass 1's sampled detections are written to a JSON
    sidecar instead, and save_face_data's preview renders the single
    combined video from it - one full-length encode instead of two.

    Pass 1 runs ONLY the YOLO person detector on every Nth frame — the
    expensive part, and its cost is unchanged. Pass 2 re-reads the video
    sequentially and writes EVERY frame at the source frame rate,
    interpolating the detection box between checked frames, so the
    preview plays at full speed with real motion instead of a held-frame
    slideshow. Around a missed sample the nearest good box holds to the
    midpoint of the gap, then frames read as "NO PERSON" (see
    _preview_payload_at). Green box + "BODY OK" = person detected, red
    "NO PERSON" = detector miss. Detector-only (no mesh inference), so
    it stays fast even on long clips. Same CLI/output contract the
    addon's preview path expects (proxy_path = preview.mp4).
    """
    print(f"\nReading video: {video_path}")
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end = min(start_frame + max_frames, vid_total) if max_frames > 0 else vid_total
    vid_frames = list(range(start_frame, end, max(1, step)))

    # Scale down for preview (cap long edge at 1920, otherwise native)
    max_edge = 1920
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        out_w = int(width * scale) // 2 * 2
        out_h = int(height * scale) // 2 * 2
    else:
        out_w, out_h = width, height

    print(f"  {len(vid_frames)} frames to check (every {step})")
    print(f"  Preview resolution: {out_w}x{out_h}")

    # ── Pass 1: detection on the sampled frames only ──────────────
    detected = missed = 0
    prev_bbox = None  # last good box, for IoU continuity across samples
    sample_boxes = []  # one entry per vid_frames item: [x1,y1,x2,y2] or None
    n_samples = len(vid_frames)
    # Pass 1 spans 30->75 when we also render (pass 2 takes 75->100), or
    # the whole 30->100 when the render is skipped (boxes_out mode).
    p1_span = 70.0 if boxes_out is not None else 45.0

    for count, vid_idx in enumerate(vid_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO expects BGR — the full pipeline converts RGB->BGR before
        # detection, and cv2 already hands us BGR, so pass frame as-is.
        # default_to_full_image=False so a true miss reads as "no person"
        # rather than a full-frame box.
        try:
            boxes = detector.run_human_detection(
                frame, det_cat_id=0, bbox_thr=0.5, default_to_full_image=False)
        except Exception as e:
            print(f"  Frame {vid_idx}: detection FAILED ({e})")
            boxes = None

        box = _pick_box(boxes, prev_bbox)
        if box is not None:
            sample_boxes.append([float(v) for v in box[:4]])
            prev_bbox = box
            detected += 1
        else:
            sample_boxes.append(None)
            missed += 1

        if n_samples > 0:
            emit_progress(30 + (count + 1) / n_samples * p1_span,
                          f"Body preview {count + 1}/{n_samples}")
        if count % 10 == 0:
            print(f"  Frame {vid_idx} - {'body OK' if box is not None else 'no person'}")

    sample_indices = vid_frames[:len(sample_boxes)]

    if boxes_out is not None:
        # Combined body+face preview: the face preview renders the ONE
        # combined video, so hand it the sampled detections and skip this
        # script's render pass entirely.
        cap.release()
        boxes_path = Path(boxes_out)
        boxes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(boxes_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "fps": fps,
                "start_frame": int(start_frame),
                "end_frame": int(end),
                "sample_indices": [int(i) for i in sample_indices],
                "boxes": sample_boxes,
            }, f)
        total_checked = detected + missed
        print(f"\nDetection samples saved: {boxes_path}")
        print(f"  {total_checked} frames checked: {detected} body OK, {missed} no person")
        if total_checked > 0 and missed > total_checked * 0.3:
            # The [WARNING] form is parsed by the addon and shown in a popup.
            print(f"[WARNING] {missed * 100 // total_checked}% of preview frames had no person")
            print("  The subject may be partly out of frame or too small.")
        return

    # ── Pass 2: render EVERY frame, box interpolated between samples ──
    # Cheap next to detection (decode + draw + encode only). Writing all
    # frames at the native fps keeps the preview's timing identical to
    # the source and 1:1 frame-aligned with it (the combined face preview
    # relies on that alignment when it draws over this file as its
    # background).
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video {output_path}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_render = max(1, end - start_frame)
    for idx in range(start_frame, end):
        ret, frame = cap.read()
        if not ret:
            break

        a, b, t = _preview_payload_at(idx, sample_indices, sample_boxes)
        box = [av + (bv - av) * t for av, bv in zip(a, b)] \
            if (a is not None and b is not None) else a
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "BODY OK", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO PERSON", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.putText(frame, f"Frame {idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if (out_w, out_h) != (width, height):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

        done = idx - start_frame + 1
        if done % 50 == 0 or done == n_render:
            emit_progress(75 + done / n_render * 25,
                          f"Rendering preview {done}/{n_render}")

    cap.release()
    writer.release()

    total_checked = detected + missed
    size_mb = os.path.getsize(str(output_path)) / (1024 * 1024)
    print(f"\nPreview saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  {total_checked} frames checked: {detected} body OK, {missed} no person")
    if total_checked > 0 and missed > total_checked * 0.3:
        # The [WARNING] form is parsed by the addon and shown in a popup.
        print(f"[WARNING] {missed * 100 // total_checked}% of preview frames had no person")
        print("  The subject may be partly out of frame or too small.")

