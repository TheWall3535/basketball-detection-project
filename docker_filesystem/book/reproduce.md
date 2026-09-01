# Reproducing the Pipeline

The project runs in a **Docker container** — a Jupyter environment with GPU access, PyTorch (CUDA 12.8), and the computer-vision system libraries preinstalled. This page describes how to build the container and run each piece — training, image serving, and video processing — from inside it.

## Environment

The environment is defined by three files at the project root:

- **`Dockerfile`** — starts from a Jupyter base image, installs the OpenCV system libraries (`libgl1-mesa-glx`, `libglib2.0-0`, and others), installs PyTorch and torchvision from the CUDA 12.8 wheel index, then installs the remaining Python packages from `requirements.txt`.
- **`requirements.txt`** — the Python dependencies (Flask, ONNX / ONNX Runtime, OpenCV, torchmetrics, pycocotools, and the rest). PyTorch itself is installed separately in the Dockerfile so it comes from the CUDA build.
- **`docker-compose.yml`** — passes the host GPU into the container, mounts the working directory, sets an 8 GB shared-memory size (needed for data-loading workers), and forwards the Jupyter port.

```{note}
A few packages in `requirements.txt` (for example `timm` and `transformers`) are carried over from earlier work and are **not used** by this project. The basketball pipeline relies on torch / torchvision, torchmetrics, pycocotools, OpenCV, Flask, ONNX / ONNX Runtime, Pillow, numpy, and requests.
```

## Build and start the container

From the project root (where the `Dockerfile` and `docker-compose.yml` live):

```bash
docker compose up --build
```

This builds the image (first run only — subsequent starts are fast) and launches the Jupyter server. The working directory on the host (mounted into the container at `/home/jovyan/work`) holds the code modules, the notebook, the dataset, trained checkpoints, and test images — so everything persists across container restarts.

Open the Jupyter server in a browser (the console prints the URL). **Everything below is run from inside the container** — either in the notebook or in a Jupyter terminal (New → Terminal). Because the service and any client both run inside the container, `localhost` works between them with no additional port forwarding.

## Data

The project uses the [Roboflow basketball-detection dataset](https://universe.roboflow.com/) in **Pascal VOC** format — the same data SwishAI was trained on, which is what makes the benchmark comparison valid. Place its `train/`, `valid/`, and `test/` splits under the mounted working directory so they are visible inside the container.

```{warning}
The dataset's labels are 0–4 (YOLO's convention). Faster R-CNN reserves class 0 for **background**, so the labels are shifted to 1–5 during data loading. Using the raw 0–4 labels would silently train one real class as background — a failure that produces no error, only degraded performance.
```

## Training a model

Training is driven end-to-end from **`basketball_training.ipynb`**, which imports
the modular code from `src/`. Rather than duplicate it here, the notebook is the
source of truth — it is included as its own chapter in this book. In outline, it:

1. **Defines the class mapping** (the five real classes at indices 1–5, with 0
   reserved for background) and builds the three dataset splits, each constructed
   with its own transforms via `get_transforms(is_train=...)`:

   ```python
   from src.utils import BasketballDataset, get_transforms
   from src.models import make_detection_loader, ModifiedFasterRCNN

   train_dataset = BasketballDataset(train_path, CLASS_MAPPING, train_transforms)
   valid_dataset = BasketballDataset(valid_path, CLASS_MAPPING, test_transforms)
   test_dataset  = BasketballDataset(test_path,  CLASS_MAPPING, test_transforms)

   train_loader = make_detection_loader(train_dataset, batch_size=4, shuffle=True)
   val_loader   = make_detection_loader(valid_dataset, batch_size=4, shuffle=False)
   test_loader  = make_detection_loader(test_dataset,  batch_size=4, shuffle=False)
   ```

2. **Constructs the detector** — a torchvision Faster R-CNN with a ResNet-50 FPN
   backbone, transfer-learned from COCO with the top three backbone stages
   trainable — and wraps it in `ModifiedFasterRCNN`, then replaces the box
   predictor for the six-class head:

   ```python
   from torchvision.models.detection import fasterrcnn_resnet50_fpn

   DETECTION_MODEL = fasterrcnn_resnet50_fpn(weights='DEFAULT',
                                             trainable_backbone_layers=3)
   experiment = ModifiedFasterRCNN(DETECTION_MODEL, optimizer=..., ...)
   experiment.replace_box_predictor(num_classes=NUM_CLASSES)   # 5 + background
   ```

3. **Trains** with mAP tracked periodically (as a diagnostic, not the stopping
   signal) and best-checkpoint-to-disk enabled, then **selects the best
   checkpoint by mAP** and **evaluates on the held-out test set**:

   ```python
   experiment.training_loop(train_loader=train_loader, val_loader=val_loader,
                            device=device, calculate_map_interval=5,
                            checkpoint_path='basketball_best.pt', save_period=10)
   experiment.select_best_by_map(val_loader, device)
   results = experiment.evaluate_detections(test_loader, device, return_full=True)
   ```

The notebook runs this as two experiments — a first run, then a warm-started
second run with more patience — and evaluates the final model on the test set.
See the notebook chapter for the complete, runnable code. Training uses the
container's GPU; inference runs comfortably on CPU.

## Running the image service

From a Jupyter terminal inside the container, start the Flask service pointing at a trained checkpoint (use the name of the checkpoint saved after training):

```bash
python detection_app.py --checkpoint basketball_final.pt --device cpu
```

Then, from a second Jupyter terminal (also inside the container), send it an image:

```bash
python detection_client.py shot.jpg               # prints JSON detections
python detection_client.py shot.jpg --save out.png # also saves the annotated image
```

```{note}
The service listens on port 5000 inside the container. Because the client runs inside the container too, it reaches the service at `localhost:5000` directly. To call the service from the **host** instead, forward the port in `docker-compose.yml` (add `"5000:5000"` under `ports`) and rebuild.
```

## Running the video processor

Annotate a video from a Jupyter terminal:

```bash
python detection_video.py game.mp4 --checkpoint basketball_final.pt --output annotated.mp4
```

Useful options:

- `--device cuda` — run on the container's GPU (much faster per frame).
- `--stride N` — run detection every *N*th frame (skipped frames reuse the last detections). Higher stride is faster but the ball's box lags during fast motion.
- `--max-seconds S` — cap processing to the first *S* seconds (a safety valve for long clips).

## Verifying the setup

Before serving, confirm the served architecture matches the trained one (so the checkpoint will load):

```bash
python tests.py
```

A passing result certifies that `load_state_dict` will succeed on your checkpoint.
