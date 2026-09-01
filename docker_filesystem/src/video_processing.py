"""
video_processing.py -- reusable video-detection logic (library, belongs in src/).

Holds process_video(): read a video frame by frame, run the detector, write an
annotated output video. Imported by the detection_video.py entry point (and by
any future async web wrapper). No argparse / no __main__ here -- this is library
code, not a runnable script.

Reuses the already-tested DetectionPredictor and draw_detections. The one thing
this bridge must get right: OpenCV reads/writes frames in BGR, but the model
(and draw_detections) work in RGB, so every frame is converted BGR->RGB before
inference and RGB->BGR before writing. Skip that and the model silently sees
color-swapped images and degrades.

Video I/O skeleton (VideoCapture/VideoWriter, fps/dim preservation, mp4v codec)
is adapted from SwishAI's processing loop.
"""

import time

import cv2
import numpy as np
from PIL import Image

# NB: when this lives in src/, the import becomes
#   from src.detection_serving import DetectionPredictor, draw_detections
# (or a relative `from .detection_serving import ...`). Adjust to match the
# package layout.
from src.detection_serving import DetectionPredictor, draw_detections


def process_video(input_path, output_path, predictor,
                  stride=1, max_seconds=None, progress_every=30):
    """Read input_path frame by frame, run detection, write annotated frames to
    output_path. Returns a small summary dict.

    stride: process every Nth frame (1 = every frame). Skipped frames are still
            written, annotated with the most recent detections, so the output
            stays smooth and full-length.
    max_seconds: cap processing to this many seconds of video (safety valve).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = total_frames
    if max_seconds is not None:
        max_frames = min(total_frames, int(fps * max_seconds))

    # writer matches the source fps + dimensions so the output plays at the
    # right speed and size. mp4v is a widely-compatible codec.
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing {input_path}")
    print(f"  {w}x{h} @ {fps:.1f}fps | {max_frames} frames "
          f"| stride {stride} | device {predictor.device}", flush=True)

    frame_idx = 0
    n_processed = 0
    last_dets = []          # reused for skipped frames so output stays annotated
    total_detections = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        success, frame_bgr = cap.read()
        if not success:
            break

        # only run the (expensive) detector every `stride` frames
        if frame_idx % stride == 0:
            # BGR (OpenCV) -> RGB (model/PIL). THE bridge -- skip it and the
            # model sees channel-swapped images.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            last_dets = predictor.predict(pil)
            total_detections += len(last_dets)
            n_processed += 1

        # draw current (or most-recent) detections on this frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = draw_detections(Image.fromarray(frame_rgb), last_dets)
        # RGB (PIL) -> BGR (OpenCV) for writing
        annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
        writer.write(annotated_bgr)

        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            elapsed = time.time() - t0
            rate = frame_idx / elapsed if elapsed else 0
            print(f"  frame {frame_idx}/{max_frames} "
                  f"| {rate:.1f} fps processed", flush=True)

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    if elapsed:
        print(f"Done: wrote {output_path} | {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx/elapsed:.1f} fps)", flush=True)
    else:
        print(f"Done: wrote {output_path}", flush=True)

    return {
        'frames_written': frame_idx,
        'frames_detected_on': n_processed,
        'total_detections': total_detections,
        'seconds': round(elapsed, 1),
        'output': str(output_path),
    }
