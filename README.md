# Automating the Film Bottleneck in Basketball

A Faster R-CNN object detector for basketball film — recognizing players, the
ball, the hoop, shots, and made shots — built as a challenger to the
[SwishAI](https://github.com/sPappalard/SwishAI) benchmark, with an end-to-end
pipeline from data loading through deployment.

The detector reaches **test mAP@50 of 0.904** (vs. SwishAI's 0.909) and
**mAP@[.5:.95] of 0.605** (vs. 0.62) — competitive with the published benchmark
at roughly **40 training epochs to their 200**. Full analysis, including the
honest account of where the model works and where it breaks, is in the
accompanying Jupyter Book (see *Documentation* below).

## What this is — and isn't

**Delivered:** a working detection model, an image inference service, and a
command-line video processor.

**One step away:** a counting layer on top of the detections would turn them into
automated shooting-percentage analytics (the natural near-term product).

This is a **research prototype**, not a finished product. It works well on clean
scenes — a practice gym with a handful of players, one ball, one hoop — and
degrades on crowded broadcast footage. That limitation is deliberate: the
intended first use case is a single program's *practice film*, which is exactly
the clean setting the model handles.

## Repository layout

```text
.
├── Dockerfile, docker-compose.yml, requirements.txt   # environment
└── docker_filesystem/            # mounted into the container at /home/jovyan/work
    ├── src/                      # the importable library
    │   ├── utils.py              # dataset, VOC parsing, transforms, visualization
    │   ├── models.py             # the training wrapper + data-loading helpers
    │   ├── detection_serving.py  # inference core: build model, predict, draw boxes
    │   └── video_processing.py   # run the detector over a video, frame by frame
    ├── detection_app.py          # entry point: Flask image service
    ├── detection_client.py       # entry point: HTTP client for the service
    ├── detection_video.py        # entry point: command-line video annotator
    ├── tests.py                  # build-time architecture-parity check
    ├── basketball_training.ipynb # the notebook that drives experiments
    └── book/                     # the Jupyter Book source (documentation)
```

Everything under `src/` is imported (`from src.utils import ...`); the scripts at
the top of `docker_filesystem/` are run directly.

## Quickstart

The project runs in a Docker container with GPU access and the CV/ML stack
preinstalled.

```bash
# build and start (first run builds the image; later starts are fast)
docker compose up --build
```

Open the Jupyter server (the console prints the URL). Everything below runs
**inside the container** — from the notebook or a Jupyter terminal.

**Train** — driven from `basketball_training.ipynb`, which imports the modular
code, builds the train/val/test loaders, trains with periodic mAP tracking and
disk checkpointing, selects the best checkpoint by mAP, and evaluates on the
test set.

**Serve an image endpoint:**

```bash
python detection_app.py --checkpoint basketball_run2.pt --device cpu
# from a second terminal:
python detection_client.py shot.jpg --save out.png
```

**Annotate a video:**

```bash
python detection_video.py game.mp4 --checkpoint basketball_final.pt --output annotated.mp4
# useful flags: --device cuda, --stride N, --max-seconds S
```

**Verify the setup** (confirms the served architecture matches the trained one,
so a checkpoint will load):

```bash
python tests.py
```

## Data

The project uses the Roboflow **basketball-detection** dataset (Pascal VOC
format) — the same data SwishAI was trained on, which is what makes the
benchmark comparison valid. It is **not committed** to this repository; download
it from Roboflow and place it under the mounted working directory.

> **Note on labels.** The dataset ships labels 0–4 (YOLO's convention). Faster
> R-CNN reserves class 0 for *background*, so labels are shifted to 1–5 during
> data loading. Using the raw labels would silently train one real class as
> background — a failure that produces no error, only degraded performance.

## Documentation

Full technical documentation is published as a **Jupyter Book** in `book/`,
covering the complete pipeline — data, preprocessing, model architecture and
mathematical foundations, training methodology, results, and the honest
limitations. Build it with:

```bash
pip install "jupyter-book<2"
jupyter-book build book/
# then open book/_build/html/index.html
```

## Design notes

A few decisions worth calling out, documented more fully in the Jupyter Book:

- **Serving is decoupled from training.** The inference core rebuilds the model
  from a saved checkpoint and depends only on the bare torchvision model — not on
  the training wrapper — so a deployed service stays lean.
- **Training and serving share one preprocessing path**, so they cannot drift
  apart (the classic train/serve skew bug). A parity test enforces that the
  architectures match.
- **Challenger architecture.** Faster R-CNN (two-stage) was chosen over YOLO for
  precise detection of small objects like a ball at the rim; inference speed is
  not the barrier to adoption here — trust in accuracy is.

## Not yet built

Named honestly, and on the path to a viable product: automatic shot tallying (the
counting layer), an asynchronous video service for multi-user uploads, and
persistence of predictions to cloud storage and a database so shooting data
accumulates and is queryable over time.
