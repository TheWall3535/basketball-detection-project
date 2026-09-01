# Automating the Film Bottleneck in Basketball

*A Faster R-CNN challenger for basketball object detection, with an honest path from prototype to product.*

## What this is

The bottleneck in basketball analytics is extracting data from film — a manual, labor-bound process that resource-limited programs cannot afford. This project takes one concrete step toward automating it: a computer-vision **object detector** that recognizes players, the basketball, the hoop, shots, and made shots in film.

It builds on [SwishAI](https://github.com/sPappalard/SwishAI), which uses object detection to compute shooting field-goal percentage (FG%) from video. The goal here was to build a **challenger** to SwishAI's YOLO-based detector on the same five-class detection task — and to do so with tooling written from the ground up, so the modeler controls every part of the pipeline and further experimentation is ergonomic.

## What was delivered

- A **Faster R-CNN detector** trained on the basketball-detection dataset, **competitive with SwishAI's published benchmark** (test mAP@50 of 0.904 vs. their 0.909; mAP@[.5:.95] of 0.605 vs. 0.62) — achieved at roughly **40 training epochs to their 200**.
- An **end-to-end pipeline**: a notebook that imports modular code to load data, train, and evaluate a model; a local **image inference service** (upload an image → JSON detections + an annotated image); and a **command-line video processor** (batch a film clip → an annotated video).
- Honest characterization of **where the model works and where it breaks** — including a quantified account of its failure on dense broadcast footage.

## Three altitudes

It helps to be explicit about what exists today versus what lies ahead:

- **Delivered today** — a working detection model that recognizes players, basketballs, shots, and makes when they happen in film.
- **One step away** — a simple counting layer on top turns those detections into automated shot analytics (FG% from film).
- **The horizon** — reducing film to full *sequences* of events (passes, screens, dribbles, shots) to power the next generation of descriptive and predictive basketball analytics.

## How to read this book

- **{doc}`architecture`** — how the code and system are organized: the modular design, the training-versus-serving split, and the two inference surfaces.
- **The pipeline notebook** — the complete workflow from data loading through training, evaluation, and inference, with the real outputs from the training runs.
- **{doc}`reproduce`** — how to install the dependencies and run the pipeline yourself.

```{note}
This is a research prototype, not a finished product. It is usable mainly in clean settings — a practice gym with a handful of players, one ball, and one hoop — and is intended for researchers and stakeholders working directly with the model, not for broad public deployment. The {doc}`architecture` page and the results discussion are candid about the current limitations and the path forward.
```
