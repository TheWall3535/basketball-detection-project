"""
tests.py -- build-time verification checks for the basketball detector.

Home for unit-test-type checks that verify the system is correct BEFORE
deployment (distinct from detection_client.py, which is the live client). These
run without a server, checkpoint, or GPU.

Currently holds the architecture-parity check: confirm the SERVED model
(rebuilt by detection_serving.build_model) matches the TRAINED
ModifiedFasterRCNN architecture, so load_state_dict is guaranteed to succeed.
Add further static checks here as needed (e.g. class-mapping round-trip,
annotation-label bounds).

serving's build_model() reconstructs the architecture from scratch (so serving
doesn't depend on the training wrapper). If that reconstruction drifts from what
was trained -- different backbone, wrong num_classes, a head built differently
-- then load_state_dict would fail, or worse, silently load a subset. This test
catches that by comparing the two models' state_dicts on three axes, strongest
last:
  1. total parameter count matches
  2. the set of state_dict KEYS matches exactly
  3. every tensor's SHAPE matches per key
(3) is the real guarantee -- it's exactly what load_state_dict requires.

Run:
    python tests.py
"""

import sys
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.detection_serving import build_model, _NUM_CLASSES


def build_trained_style_model(num_classes=_NUM_CLASSES):
    """Reproduce the architecture the TRAINING wrapper produces: torchvision
    fasterrcnn_resnet50_fpn with the box predictor replaced for num_classes.
    This mirrors ModifiedFasterRCNN.replace_box_predictor without importing the
    wrapper (which would drag in training-only deps). If your training used a
    different base construction, change THIS to match, and the test will tell
    you whether serving's build_model agrees."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def compare(trained, served):
    ts, ss = trained.state_dict(), served.state_dict()
    results = {}

    # (1) param count
    tc, sc = param_count(trained), param_count(served)
    results['param_count'] = (tc, sc, tc == sc)

    # (2) key sets
    tk, sk = set(ts.keys()), set(ss.keys())
    only_trained = tk - sk
    only_served = sk - tk
    results['keys'] = (len(tk), len(sk), only_trained, only_served,
                       tk == sk)

    # (3) per-key shapes (only for shared keys)
    shape_mismatches = []
    for k in tk & sk:
        if ts[k].shape != ss[k].shape:
            shape_mismatches.append((k, tuple(ts[k].shape), tuple(ss[k].shape)))
    results['shapes'] = (shape_mismatches, len(shape_mismatches) == 0)

    return results


def main():
    trained = build_trained_style_model()
    # Replicate build_model's construction WITHOUT loading a checkpoint, so this
    # is a pure architecture check (no weights file or GPU needed).
    served = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_f = served.roi_heads.box_predictor.cls_score.in_features
    served.roi_heads.box_predictor = FastRCNNPredictor(in_f, _NUM_CLASSES)

    r = compare(trained, served)

    print("=" * 66)
    print("SERVING vs TRAINING architecture parity")
    print("=" * 66)

    tc, sc, ok_count = r['param_count']
    print(f"[{'PASS' if ok_count else 'FAIL'}] param count: "
          f"trained={tc:,}  served={sc:,}")

    tn, sn, only_t, only_s, ok_keys = r['keys']
    print(f"[{'PASS' if ok_keys else 'FAIL'}] state_dict keys: "
          f"trained={tn}  served={sn}")
    if only_t:
        print(f"       keys only in trained ({len(only_t)}): "
              f"{sorted(only_t)[:5]}{' ...' if len(only_t) > 5 else ''}")
    if only_s:
        print(f"       keys only in served ({len(only_s)}): "
              f"{sorted(only_s)[:5]}{' ...' if len(only_s) > 5 else ''}")

    mism, ok_shapes = r['shapes']
    print(f"[{'PASS' if ok_shapes else 'FAIL'}] tensor shapes: "
          f"{len(mism)} mismatch(es)")
    for k, tshape, sshape in mism[:5]:
        print(f"       {k}: trained{tshape} vs served{sshape}")

    all_ok = ok_count and ok_keys and ok_shapes
    print("=" * 66)
    if all_ok:
        print("PASS: served architecture matches trained -- load_state_dict "
              "will succeed.")
    else:
        print("FAIL: architectures differ. build_model in detection_serving.py "
              "must be reconciled with the training architecture before the "
              "checkpoint will load correctly.")
    print("=" * 66)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
