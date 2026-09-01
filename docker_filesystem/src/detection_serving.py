"""
detection_serving.py -- serving core for the basketball detector.

Loads a trained Faster R-CNN checkpoint (state_dict, NOT a pickled wrapper) by
rebuilding the architecture and pouring the weights in -- the portable,
production-standard pattern. Deliberately does NOT depend on the training
wrapper (ModifiedFasterRCNN): serving needs only the bare torchvision model
plus trained weights, so the training code and its optimizer/early-stopper/
history are irrelevant here.

Provides:
  - build_model(checkpoint_path, num_classes)  -> ready eval-mode model
  - DetectionPredictor: preprocess -> infer -> (filtered) detections
  - draw_detections(image, dets)               -> annotated PIL image

Preprocessing MUST match what the model saw at eval time in training: the
deterministic tail (PIL -> float tensor in [0,1]); the model's internal
transform handles resize + normalize. No manual resize (see training notes).
"""

import io
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# --- class names: model index (1-5) -> human label. Index 0 is background.
# Fill from the Roboflow data.yaml `names:` list, IN ORDER. The dataset shipped
# YOLO-style 0-4; we shifted to 1-5 (0=background), so names[i] here is the
# label for model index i+1.
CLASS_NAMES = {
    1: "Ball",
    2: "Ball in Basket",
    3: "Player",
    4: "Basket",
    5: "Player Shooting",
}
# Order confirmed against the user's CLASS_NAMES_TO_INDEX (original YOLO 0-4
# shifted +1, background=0): ball=1, ball_in_basket=2, player=3, basket=4,
# player_shooting=5.

_NUM_CLASSES = 6   # 5 real + background


# deterministic eval transform: matches get_transforms(is_train=False) from
# training. PIL -> float tensor in [0,1]; model handles resize + normalize.
_eval_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


def build_model(checkpoint_path, num_classes=_NUM_CLASSES, device='cpu'):
    """Rebuild the Faster R-CNN architecture and load trained weights.

    Constructs the same architecture used in training (resnet50-fpn backbone,
    box predictor resized to num_classes), then loads the state_dict. No
    dependency on the training wrapper -- weights are just tensors."""
    # weights=None: we're loading OUR trained weights, not COCO pretrained
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


class DetectionPredictor:
    """Wraps a loaded model: image -> filtered detections. Built once, reused
    per request (the model/session analog from Assignment 9)."""

    def __init__(self, checkpoint_path, device='cpu', score_threshold=0.5,
                 class_names=None, num_classes=_NUM_CLASSES):
        self.device = device
        self.score_threshold = score_threshold
        self.class_names = class_names or CLASS_NAMES
        self.model = build_model(checkpoint_path, num_classes, device)

    def predict(self, image):
        """PIL image (or path) -> list of detection dicts:
        [{label, label_id, score, box:[x1,y1,x2,y2]}], filtered by threshold."""
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert('RGB')

        x = _eval_tf(image).to(self.device)
        with torch.no_grad():
            out = self.model([x])[0]   # single-image batch -> first result

        boxes = out['boxes'].detach().cpu()
        labels = out['labels'].detach().cpu()
        scores = out['scores'].detach().cpu()

        dets = []
        for box, lab, sc in zip(boxes, labels, scores):
            s = float(sc)
            if s < self.score_threshold:
                continue
            lid = int(lab)
            dets.append({
                'label': self.class_names.get(lid, str(lid)),
                'label_id': lid,
                'score': round(s, 4),
                'box': [round(float(v), 1) for v in box.tolist()],  # x1,y1,x2,y2
            })
        return dets


# --- a small palette so each class draws in a consistent color ---
_PALETTE = {
    1: (255, 87, 51),    # Ball            - orange-red
    2: (46, 204, 113),   # Ball in Basket  - green
    3: (155, 89, 182),   # Player          - purple
    4: (52, 152, 219),   # Basket          - blue
    5: (241, 196, 15),   # Player Shooting - yellow
}


def draw_detections(image, dets):
    """Return a copy of `image` (PIL, or path) with detection boxes + labels
    drawn on it. Colors are per-class for legibility."""
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in dets:
        x1, y1, x2, y2 = d['box']
        color = _PALETTE.get(d['label_id'], (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        caption = f"{d['label']} {d['score']:.2f}"
        # label background for readability
        tw = draw.textlength(caption, font=font) if font else 8 * len(caption)
        draw.rectangle([x1, y1 - 14, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - 13), caption, fill=(0, 0, 0), font=font)
    return image


def image_to_png_bytes(image):
    """PIL image -> PNG bytes, for returning an annotated image over HTTP."""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()
