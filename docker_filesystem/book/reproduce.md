# Reproducing the Pipeline

This page covers how to set up the environment and run each piece — training, image serving, and video processing.

## Dependencies

The project splits into a **serving** side (light) and a **build/training** side (heavy). Both are captured in `requirements.txt`:

```text
# --- serving (light) ---
flask
onnxruntime
onnx
numpy
Pillow
requests
opencv-python

# --- build / training (heavy) ---
torch
torchvision
torchmetrics
albumentations
```

Install into a fresh environment:

```bash
pip install -r requirements.txt
```

```{note}
Training requires a GPU to be practical. Inference (both image and video) runs on CPU, though a GPU makes video processing dramatically faster.
```

## Data

The project uses the [Roboflow basketball-detection dataset](https://universe.roboflow.com/) in **Pascal VOC** format — the same data SwishAI was trained on, which is what makes the benchmark comparison valid. Download it from Roboflow and note the path to its `train/`, `valid/`, and `test/` splits.

```{warning}
The dataset's labels are 0–4 (YOLO's convention). Faster R-CNN reserves class 0 for **background**, so the labels are shifted to 1–5 during data loading. Using the raw 0–4 labels would silently train one real class as background — a failure that produces no error, only degraded performance.
```

## Training a model

Training is driven from the notebook, which imports the modular code. In outline:

```python
from utils import get_transforms, make_detection_loader
from models_faster_rcnn import ModifiedFasterRCNN
# ... plus the dataset class and the torchvision Faster R-CNN constructor

# build the three splits, each with its own transforms
train_ds = BasketballDataset(train_dir, CLASS_MAPPING, transforms=get_transforms(is_train=True))
val_ds   = BasketballDataset(valid_dir, CLASS_MAPPING, transforms=get_transforms(is_train=False))
test_ds  = BasketballDataset(test_dir,  CLASS_MAPPING, transforms=get_transforms(is_train=False))

train_loader = make_detection_loader(train_ds, batch_size=4, shuffle=True)
val_loader   = make_detection_loader(val_ds,   batch_size=4, shuffle=False)
test_loader  = make_detection_loader(test_ds,  batch_size=4, shuffle=False)

# construct the wrapper, replace the head for 6 classes (5 + background),
# then train with mAP tracked periodically and disk checkpointing on
model.replace_box_predictor(num_classes=6)
model.training_loop(train_loader, val_loader, device,
                    epochs=60, calculate_map_interval=5,
                    checkpoint_path='basketball_best.pt', save_period=10)

# after training, select the best checkpoint by mAP and evaluate on test
model.select_best_by_map(val_loader, device, checkpoint_path='basketball_final.pt')
results = model.evaluate_detections(test_loader, device, return_full=True)
```

See the pipeline notebook for the complete, runnable version.

## Running the image service

Start the Flask service, pointing it at a trained checkpoint:

```bash
python detection_app.py --checkpoint basketball_final.pt --device cpu
```

Then, from another terminal, send it an image:

```bash
python detection_client.py shot.jpg               # prints JSON detections
python detection_client.py shot.jpg --save out.png # also saves the annotated image
```

## Running the video processor

Annotate a video from the command line:

```bash
python detection_video.py game.mp4 --checkpoint basketball_final.pt --output annotated.mp4
```

Useful options:

- `--device cuda` — run on GPU (much faster per frame).
- `--stride N` — run detection every *N*th frame (skipped frames reuse the last detections). Higher stride is faster but the ball's box lags during fast motion.
- `--max-seconds S` — cap processing to the first *S* seconds (a safety valve for long clips).

## Verifying the setup

Before serving, confirm the served architecture matches the trained one (so the checkpoint will load):

```bash
python tests.py
```

A passing result certifies that `load_state_dict` will succeed on your checkpoint.
