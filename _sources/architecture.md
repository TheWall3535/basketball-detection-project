# Code & System Architecture

This page documents how the codebase is organized and the design decisions behind it. The guiding principle throughout: **the notebook conducts experiments against reusable code that lives in an importable `src/` package** — logic is defined once, in the library, so that serving never has to reinvent what training established.

## Project layout

```text
project root/
├── src/                       # the importable library
│   ├── utils.py               # dataset, VOC parsing, transforms, visualization
│   ├── models.py              # the training wrapper + data-loading helpers
│   ├── detection_serving.py   # inference core: build model, predict, draw boxes
│   └── video_processing.py    # run the detector over a video, frame by frame
├── detection_app.py           # entry point: the Flask image service
├── detection_client.py        # entry point: HTTP client for the service
├── detection_video.py         # entry point: command-line video annotator
├── tests.py                   # build-time checks (architecture parity)
├── basketball_training.ipynb  # the notebook that drives experiments
├── Dockerfile, docker-compose.yml, requirements.txt
```

Everything under `src/` is imported (`from src.utils import ...`); the scripts at the root are *run*. Entry points import from `src/`, never the reverse.

## What lives where

| Module | Contents | Role |
|---|---|---|
| `src/utils.py` | `BasketballDataset`, `parse_voc_xml`, `get_transforms`, `visualize_sample` | Data intake and the v2 transform pipelines. |
| `src/models.py` | `ModifiedFasterRCNN`, `EarlyStopping`, `make_detection_loader`, `detection_collate_fn` | The training wrapper (head replacement, freezing, the training loop, the two-mode loss paths, mAP evaluation, mAP-based checkpoint selection) plus loader construction. |
| `src/detection_serving.py` | `build_model`, `DetectionPredictor`, `draw_detections`, `image_to_png_bytes` | The inference core: rebuild the model from a checkpoint, run it, draw detections. No dependency on the training wrapper. |
| `src/video_processing.py` | `process_video` | Reads a video frame by frame, runs the detector on each frame, writes an annotated output. |
| `detection_app.py` | Flask `/predict` endpoint | Image service. Built once at startup, reused per request. |
| `detection_client.py` | HTTP client | POSTs an image to the running service; torch-free (only `requests`). |
| `detection_video.py` | CLI entry point | Argument parsing over `process_video`. |
| `tests.py` | architecture-parity check | Confirms the served model matches the trained one so a checkpoint will load. |

## Three design principles

**Lean deployment.** The serving core (`detection_serving.py`) rebuilds the model from a saved `state_dict` and depends only on the bare torchvision model — not on `ModifiedFasterRCNN` or its optimizer, early stopper, and history. So a deployed service loads only what inference needs, not the full training stack.

**Consistent by construction.** Training and inference share **one** preprocessing definition. The deterministic transform tail in `get_transforms(is_train=False)` — PIL → float tensor in `[0,1]`, with resize and normalization delegated to the model's internal transform — is exactly what `detection_serving.py` applies at inference. The single most dangerous bug in a deployment like this is *train/serve skew*: the model receiving subtly different inputs in production than in training, silently degrading predictions. Sharing the preprocessing prevents drift, and `tests.py` enforces that the architectures match.

**Built to improve.** Because the reusable logic lives in `src/` and the notebook merely orchestrates experiments against it, a researcher can iterate quickly — change a hyperparameter, retrain, re-evaluate — against a stable, tested base. The prototype is a platform for the research that carries it toward a product, not a dead-end artifact.

## The training wrapper

`ModifiedFasterRCNN` (in `src/models.py`) wraps a torchvision Faster R-CNN and owns the training lifecycle:

- **`replace_box_predictor(num_classes)`** swaps the detection head for one sized to six outputs (five classes + background).
- **`training_loop(...)`** runs training with within-epoch progress printing, tracks mAP periodically (`calculate_map_interval`) as a diagnostic without letting it drive stopping, and — importantly for crash-resilience — writes the best checkpoint to disk (`checkpoint_path`) and periodic snapshots (`save_period`).
- **`compute_val_loss`** and **`evaluate_detections`** are the two evaluation paths. They use *different model modes*: a detection model in `train()` mode returns losses (used for the cheap early-stopping signal), while `eval()` mode returns detections (used for the expensive mAP computation). `evaluate_detections(..., return_full=True)` exposes mAP@50 and mAP@75 alongside the strict mAP@[.5:.95].
- **`select_best_by_map`** chooses the final checkpoint by mAP rather than by loss, reflecting that loss and the metric of interest can diverge.

## The two inference surfaces

A single trained checkpoint feeds two ways of consuming the model:

- **Image service** (`detection_app.py`) — a Flask endpoint. POST an image; receive JSON detections (labels, confidences, boxes) by default, or the annotated image when `?annotated=true` is passed. Exposing *per-image* predictions — not just a final annotated video — aids error diagnosis, fosters transparency, and surfaces where the model needs richer data.
- **Video processor** (`detection_video.py` → `src/video_processing.py`) — a command-line tool that reads a video, runs the image detector on each frame, draws the detections, and writes an annotated output video. Because the model is a *per-frame image* detector, video is simply "run the image model on each frame" — it has no memory of motion across frames, which is why a shot fake (a single frame that looks like a shot) can trigger a false detection.

```{note}
The OpenCV/PIL colour-channel bridge matters here: OpenCV reads and writes frames in **BGR**, but the model and `draw_detections` work in **RGB**. `process_video` converts BGR→RGB before inference and RGB→BGR before writing each frame. Skipping that conversion would feed the model colour-swapped images and silently degrade detection.
```

```{note}
Serving video over HTTP for many users would require an asynchronous job pattern (upload → background processing → poll for status → download the result), since a full video takes minutes to process and cannot be handled inside a single request/response cycle. That is deliberately **not** built here — it is deployment plumbing that does not improve the detector — but it is the natural next step for a multi-user product.
```

## The architecture-parity test

Because `detection_serving.build_model` **rebuilds** the architecture from scratch (so serving carries no dependency on the training wrapper), the two definitions of the architecture could drift apart — and if they did, loading a trained checkpoint into the served model would fail, or silently load only part of it.

`tests.py` guards against exactly this. It builds both the training-style and serving-style architectures and compares their `state_dict`s on three axes, strongest last:

1. total parameter count matches,
2. the set of parameter keys matches exactly,
3. every tensor's shape matches per key.

The third is the real guarantee — it is precisely what `load_state_dict` requires — so a passing test certifies that the checkpoint will load correctly. This is the same "verify the thing that can silently break" discipline as the shared-preprocessing choice.

## Deployment approach (current state)

The system currently runs **locally**, inside the project's Docker container, for a single researcher; predictions are **written to disk**. This is appropriate for the model's present maturity — a research instrument, not a public product.

**Not yet implemented**, and required for real use: persisting predictions to **cloud storage** and loading them (via ETL/ELT) into a **relational database**, so that shooting data accumulates and becomes queryable over time. Without persistence there is no "track a player's progress across sessions," which is the core of the eventual value proposition. This is a near-term item on the path to an MVP, not a delivered capability.
