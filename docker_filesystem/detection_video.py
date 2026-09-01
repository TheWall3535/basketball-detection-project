"""
detection_video.py -- entry point to annotate a basketball video from the
command line (file in, annotated file out).

This is a runnable script, NOT a client: it processes a local file directly and
does not talk to a server. (Serving video over HTTP would need the async job
pattern -- upload -> background thread -> poll -> download -- which is plumbing
that doesn't improve the detector, so it's documented rather than built.)

The reusable processing logic lives in video_processing.process_video; this file
is just argument parsing, predictor construction, and the call.

Usage:
    python detection_video.py game.mp4 --checkpoint model.pt --output out.mp4
    python detection_video.py game.mp4 --checkpoint model.pt --device cuda
    python detection_video.py game.mp4 --checkpoint model.pt --stride 2 --max-seconds 20
"""

import os
import argparse

from src.detection_serving import DetectionPredictor
from src.video_processing import process_video



def main():
    parser = argparse.ArgumentParser(description="Annotate a basketball video")
    parser.add_argument('input', help="input video file path")
    parser.add_argument('--checkpoint', required=True, help="trained .pt state_dict")
    parser.add_argument('--output', default=None,
                        help="output video path (default: <input>_annotated.mp4)")
    parser.add_argument('--device', default='cpu', help="'cpu' or 'cuda'")
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--stride', type=int, default=1,
                        help="run detection every Nth frame (1 = every frame)")
    parser.add_argument('--max-seconds', type=float, default=None,
                        help="cap processing to this many seconds (safety valve)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"input video not found: {args.input}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    # graceful GPU fallback: asking for cuda without a GPU falls back to cpu
    # rather than erroring, so the same command works on any machine.
    device = args.device
    try:
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: --device cuda requested but no GPU; using cpu.")
            device = 'cpu'
    except Exception:
        pass

    output = args.output or (os.path.splitext(args.input)[0] + '_annotated.mp4')

    predictor = DetectionPredictor(
        args.checkpoint, device=device, score_threshold=args.score_threshold)

    summary = process_video(
        args.input, output, predictor,
        stride=args.stride, max_seconds=args.max_seconds)
    print(summary)


if __name__ == '__main__':
    main()
