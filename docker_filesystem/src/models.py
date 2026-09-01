"""
models.py (detection wrapper) -- basketball final project.

ModifiedFasterRCNN adapted from Assignment 5. The only structural change from
that version is that loader construction no longer goes through an owned
DetectionPreprocessor: transforms now live on the dataset objects themselves
(each of Roboflow's pre-split train/valid/test datasets is constructed with its
own transforms), so the wrapper just wraps a ready dataset in a DataLoader via
the standalone make_detection_loader(), which guarantees the detection
collate_fn. The preprocessor_class / preprocessor / preprocessor_kwargs
constructor arguments and the internal preprocessor-building block are gone.

Everything else -- head replacement, freezing, the two-mode train/eval loss
paths, mAP evaluation, loss-incumbent snapshotting, and mAP-based checkpoint
selection -- is unchanged from the pothole version, because that logic is
dataset-agnostic.
"""



import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
import glob
import ssl
from collections import Counter
import pandas as pd




def detection_collate_fn(batch):
    """Group a batch into (images_tuple, targets_tuple) instead of stacking.
    This is the format GeneralizedRCNN expects, and the only collation that
    works when images differ in size and box counts differ per image."""
    return tuple(zip(*batch))

def make_detection_loader(dataset, **loader_kwargs):
    """DataLoader over a detection dataset, forcing the detection collate_fn
    unless the caller overrides it. Batching knobs pass through."""
    loader_kwargs.setdefault('collate_fn', detection_collate_fn)
    return DataLoader(dataset, **loader_kwargs)

class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Current model
            
        Returns:
            bool: Whether to stop training
        """
        if val_loss != val_loss:   # NaN check
            raise ValueError("val_loss is NaN — training diverged.")
        
        if self.best_loss is None:
            # First epoch
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  → Early stopping: Baseline set (val_loss={val_loss:.4f})")
        elif val_loss > self.best_loss - self.min_delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"  → Early stopping: {self.counter}/{self.patience} "
                      f"(val_loss={val_loss:.4f}, best={self.best_loss:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  ⚠ Early stopping triggered! No improvement for {self.patience} epochs.")    
        else:

            # Improvement
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"  → Early stopping: Improved! (val_loss={val_loss:.4f})")
        
        return self.early_stop

    def restore_best_weights(self, model):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            if self.verbose:
                print(f"  → Restored best weights (val_loss={self.best_loss:.4f})")
        return model

    def reset_counter(self):
        self.counter = 0
        self.early_stop=False
        return self

    def reset_all_state(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None
        return self
        




class ModifiedFasterRCNN:
    def __init__(self,
                 detection_model,               # torchvision Faster R-CNN instance
                 optimizer,                     # optimizer CLASS, e.g. torch.optim.SGD
                 learning_scheduler,            # scheduler CLASS, e.g. ReduceLROnPlateau
                 early_stopper_class,           # EarlyStopping CLASS (built here)
                 early_stopper=None,            # OR inject a ready instance (override)
                 early_stopper_kwargs=None,     # kwargs for building the early stopper
                 loss_combiner=None):           # dict[str,Tensor]->scalar; default sum
        """
        Each wrapper OWNS a fresh set of collaborators. Pass CLASSES (plus
        optional kwargs) and the wrapper constructs its own optimizer,
        scheduler, and early stopper -- so running N experiments means tracking
        N models, not N x (optimizer, stopper) loose objects, and no
        collaborator is ever shared between experiments (which is what makes a
        fresh early stopper per model safe).
 
        A ready early stopper can still be injected via `early_stopper=` as an
        escape hatch; an injected instance wins over class-based construction.
 
        Loaders are built via the standalone make_detection_loader() (transforms
        live on the datasets), so this wrapper no longer owns a preprocessor.
        """
        self.model = detection_model
        self._optimizer = optimizer
        self._learning_scheduler = learning_scheduler
 
        # Early stopper: injected instance wins, else build a fresh one.
        if early_stopper is not None:
            self.early_stopper = early_stopper
        else:
            self.early_stopper = early_stopper_class(**(early_stopper_kwargs or {}))
 
        self.loss_combiner = loss_combiner if loss_combiner is not None \
            else (lambda loss_dict: sum(loss_dict.values()))
 
        self._replaced_predictor = False
        self.mode = None
        self.training_history = []
 
        # candidates for end-of-training mAP selection: every epoch that sets a
        # new best val loss gets snapshotted here.
        self._loss_incumbent_snapshots = []   # list of (epoch, state_dict, val_loss)
 
        # best interval-mAP snapshot -- closes the hole where the best-mAP epoch
        # was never a best-val-loss epoch, so it wouldn't otherwise be a
        # selection candidate. Populated when calculate_map_interval is on.
        self._best_map_snapshot = None        # (epoch, state_dict, val_map) or None
 
    # ------------------------------------------------------------------ #
    # mode handling
    # ------------------------------------------------------------------ #
    def set_model_mode(self, mode):
        """'train' or 'eval'. For detection this changes the model's OUTPUT
        TYPE (losses vs detections), not just BN/dropout behavior."""
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f"mode must be 'train' or 'eval'. Got: {mode}")
        self.mode = mode
        return self
 
    # ------------------------------------------------------------------ #
    # head replacement (box predictor swap)
    # ------------------------------------------------------------------ #
    def replace_box_predictor(self, num_classes):
        """Replace the detection head's final classification + box-regression
        layers, sized to num_classes. num_classes must INCLUDE background
        (e.g. 5 basketball classes + background = 6). in_features is read from
        the existing predictor, never passed in."""
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self._replaced_predictor = True
        return self
 
    # ------------------------------------------------------------------ #
    # freezing (RPN only; backbone handled at construction)
    # ------------------------------------------------------------------ #
    def set_component_trainable(self, component_prefix, trainable):
        """Toggle requires_grad for all params whose name starts with
        component_prefix (e.g. 'rpn', 'roi_heads.box_predictor')."""
        for name, p in self.model.named_parameters():
            if name.startswith(component_prefix):
                p.requires_grad = trainable
        return self
 
    def freeze_rpn(self):
        return self.set_component_trainable('rpn', False)
 
    def unfreeze_rpn(self):
        return self.set_component_trainable('rpn', True)
 
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
 
    # ------------------------------------------------------------------ #
    # optimizer / scheduler
    # ------------------------------------------------------------------ #
    def instantiate_optimizer(self, parameters, **kwargs):
        self.optimizer = self._optimizer(parameters, **kwargs)
        return self
 
    def instantiate_learning_scheduler(self, **kwargs):
        self.learning_scheduler = self._learning_scheduler(self.optimizer, **kwargs)
        return self
 
    # ------------------------------------------------------------------ #
    # dataloader construction (delegates to the standalone loader factory)
    # ------------------------------------------------------------------ #
    def build_loader(self, dataset, is_train, **loader_kwargs):
        """Turn one split into a DataLoader via make_detection_loader. Transforms
        already live on the dataset; this only handles batching + collate_fn.
        is_train is a sensible default for shuffle (train shuffles, eval doesn't)
        unless overridden."""
        loader_kwargs.setdefault('shuffle', is_train)
        return make_detection_loader(dataset, **loader_kwargs)
 
    def build_loaders(self, train_dataset, val_dataset, test_dataset=None,
                      **loader_kwargs):
        """Build train/val (and optional test) loaders in one call. Batching
        kwargs (batch_size, num_workers, ...) apply to all; shuffle is set per
        split (True for train, False for val/test). Returns a dict of loaders."""
        loaders = {
            'train': self.build_loader(train_dataset, is_train=True, **loader_kwargs),
            'val':   self.build_loader(val_dataset,   is_train=False, **loader_kwargs),
        }
        if test_dataset is not None:
            loaders['test'] = self.build_loader(test_dataset, is_train=False,
                                                **loader_kwargs)
        return loaders
 
    # ------------------------------------------------------------------ #
    # device
    # ------------------------------------------------------------------ #
    def to(self, device):
        self.model.to(device)
        return self
 
    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, train_loader, device, epoch=None, epochs=None,
                         print_every=20):
        """Real training pass: train() mode, gradients on. Model returns a loss
        dict; combine and backprop. Returns running average combined loss.
 
        Prints within-epoch progress every `print_every` batches so a slow
        detection epoch is visibly alive rather than apparently hung. Set
        print_every=0 to silence within-epoch output."""
        self.set_model_mode('train')
        running = 0.0
        n = 0
        n_batches = len(train_loader)
        epoch_tag = f"Epoch {epoch + 1}/{epochs} " if epoch is not None else ""
 
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)   # train mode -> losses
            loss = self.loss_combiner(loss_dict)
            loss.backward()
            self.optimizer.step()
 
            running += loss.item()
            n += 1
 
            if print_every and (batch_idx + 1) % print_every == 0:
                # also surface the individual loss components -- useful for
                # spotting e.g. an exploding box-reg loss early
                parts = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items())
                print(f"  {epoch_tag}batch {batch_idx + 1}/{n_batches} "
                      f"| loss {running / n:.4f} | {parts}", flush=True)
 
        return running / max(n, 1)
 
    def training_loop(self,
                      train_loader,
                      val_loader,
                      device,
                      epochs=10,
                      early_stopping='use',
                      calculate_map_interval=None,   # None => no periodic mAP diagnostic
                      print_every=20,                # within-epoch batch print cadence
                      checkpoint_path=None,          # if set, save best weights here to disk
                      save_period=None,              # if set, also save every N epochs to disk
                      **kwargs):
        import time
        import os
 
        optimizer_kwargs = {k.split('__', 1)[1]: v for k, v in kwargs.items()
                            if k.startswith('optimizer__')}
        scheduler_kwargs = {k.split('__', 1)[1]: v for k, v in kwargs.items()
                            if k.startswith('scheduler__')}
 
        self.to(device)
        self.instantiate_optimizer(self.trainable_parameters(), **optimizer_kwargs)
        self.instantiate_learning_scheduler(**scheduler_kwargs)
 
        if early_stopping == 'reset':
            self.early_stopper.reset_all_state()
        elif early_stopping == 'soft':
            self.early_stopper.reset_counter()
 
        n_trainable = sum(p.numel() for p in self.trainable_parameters())
        print("=" * 70)
        print(f"Starting detection training for {epochs} epoch(s)")
        print(f"  trainable params: {n_trainable:,} | "
              f"train batches/epoch: {len(train_loader)} | "
              f"device: {device}")
        print("=" * 70, flush=True)
 
        for epoch in range(epochs):
            t0 = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)
 
            train_loss = self._train_one_epoch(
                train_loader, device, epoch=epoch, epochs=epochs,
                print_every=print_every)
            val_loss = self.compute_val_loss(val_loader, device)
 
            self.learning_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
 
            metrics = {
                'epoch': len(self.training_history),
                'epoch_of_current_run': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
            }
 
            # periodic mAP diagnostic. Does NOT drive early stopping (we monitor
            # loss with high patience), but the best interval-mAP snapshot is
            # kept as an end-of-training selection candidate.
            map_str = ""
            if calculate_map_interval and (epoch + 1) % calculate_map_interval == 0:
                val_map = self.evaluate_detections(val_loader, device)
                metrics['val_map'] = val_map
                map_str = f" | val_mAP {val_map:.4f}"
 
                # keep the best interval-mAP snapshot as a selection candidate,
                # closing the hole where the best-mAP epoch was never a
                # best-val-loss epoch.
                if (self._best_map_snapshot is None
                        or val_map > self._best_map_snapshot[2]):
                    snap = {k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()}
                    self._best_map_snapshot = (epoch, snap, val_map)
                    map_str += "  <- new best val_mAP"
 
            self.training_history.append(metrics)
 
            # snapshot loss-incumbents for end-of-training mAP selection.
            # (the early stopper independently tracks its own best-loss weights)
            is_new_best = (self.early_stopper.best_loss is None
                           or val_loss < self.early_stopper.best_loss)
            star = "  <- new best val_loss" if is_new_best else ""
            if is_new_best:
                self._snapshot_loss_incumbent(epoch, val_loss)
                # persist best-loss weights to disk immediately, so a kernel
                # death never costs more than the current epoch. This is the
                # crash-resilience fix: state lived only in memory before.
                if checkpoint_path is not None:
                    torch.save(self.model.state_dict(), checkpoint_path)
                    star += f" (saved -> {checkpoint_path})"
 
            # periodic every-N-epoch save, independent of best-ness, as a
            # coarser safety net (mirrors SwishAI's save_period).
            if save_period and checkpoint_path is not None \
                    and (epoch + 1) % save_period == 0:
                root, ext = os.path.splitext(checkpoint_path)
                periodic_path = f"{root}_epoch{epoch + 1}{ext or '.pt'}"
                torch.save(self.model.state_dict(), periodic_path)
 
            elapsed = time.time() - t0
            print(f"  epoch {epoch + 1} done in {elapsed:.1f}s | "
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
                  f"{map_str} | lr {current_lr:.2e}{star}", flush=True)
 
            if early_stopping != 'off':
                if self.early_stopper(val_loss, self.model):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}", flush=True)
                    break
 
        print("\n" + "=" * 70)
        print("Training complete.")
        print("=" * 70, flush=True)
        return self
 
    # ------------------------------------------------------------------ #
    # the two evaluation paths (mutually exclusive model modes)
    # ------------------------------------------------------------------ #
    def compute_val_loss(self, val_loader, device):
        """Cheap early-stopping signal. Needs train() mode (to get losses) +
        no_grad() (to not learn). Targets go TO THE MODEL.
 
        NB: train() would normally drift BatchNorm running stats, but the
        detection backbone uses FrozenBatchNorm by default, so there are no
        stats to corrupt. If a trainable-BN backbone is ever used, BN would
        need to be forced to eval here."""
        self.set_model_mode('train')
        running = 0.0
        n = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                running += self.loss_combiner(loss_dict).item()
                n += 1
        return running / max(n, 1)
 
    def evaluate_detections(self, loader, device, return_full=False):
        """Expensive mAP path. Needs eval() mode (to get detections). Model is
        called WITHOUT targets; targets go TO THE METRIC, not the model.
 
        Uses torchmetrics MeanAveragePrecision. Returns COCO-style mAP@[.5:.95]
        by default (a single float); pass return_full=True to get the whole
        result dict, which includes map_50 and map_75 -- the AP@0.5 vs AP@0.75
        split that reads as classification-vs-localization quality.
 
        NB for the SwishAI comparison: SwishAI reports mAP@50. To compare
        apples-to-apples, call with return_full=True and read result['map_50'],
        NOT the default mAP@[.5:.95] (which is always lower).
 
        Format contract (the part that fails silently if wrong):
          predictions: list of dicts with 'boxes', 'scores', 'labels'
          targets:     list of dicts with 'boxes', 'labels'  (NO scores)
        Both boxes are [N,4] xyxy. The metric wants CPU tensors, so predictions
        and targets are moved off-device before being handed over.
        """
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
 
        self.set_model_mode('eval')            # eval() -> model returns detections
        metric = MeanAveragePrecision(box_format='xyxy')
 
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                preds = self.model(images)     # no targets -> list of detection dicts
 
                # predictions: keep only the three keys the metric expects, on CPU
                preds_cpu = [{
                    'boxes': p['boxes'].detach().cpu(),
                    'scores': p['scores'].detach().cpu(),
                    'labels': p['labels'].detach().cpu(),
                } for p in preds]
 
                # targets: boxes + labels only (NO scores), on CPU. Note the
                # boxes may be a tv_tensors.BoundingBoxes subclass; plain-tensor
                # them so the metric doesn't choke on the wrapper.
                targets_cpu = [{
                    'boxes': torch.as_tensor(t['boxes']).detach().cpu(),
                    'labels': t['labels'].detach().cpu(),
                } for t in targets]
 
                metric.update(preds_cpu, targets_cpu)
 
        result = metric.compute()
        if return_full:
            return {k: (v.item() if torch.is_tensor(v) and v.ndim == 0 else v)
                    for k, v in result.items()}
        return result['map'].item()
 
    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, images, device):
        """eval() mode; returns list of detection dicts (boxes/labels/scores)."""
        self.set_model_mode('eval')
        with torch.no_grad():
            images = [img.to(device) for img in images]
            return self.model(images)
 
    # ------------------------------------------------------------------ #
    # end-of-training mAP-based selection over saved loss-incumbents
    # ------------------------------------------------------------------ #
    def _snapshot_loss_incumbent(self, epoch, val_loss):
        snapshot = {k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()}
        self._loss_incumbent_snapshots.append((epoch, snapshot, val_loss))
        return self
 
    def select_best_by_map(self, loader, device, checkpoint_path=None):
        """After training: compute mAP on each candidate snapshot, restore the
        weights of the best. Candidates are the loss-incumbents PLUS the best
        interval-mAP snapshot (if interval mAP was on) -- the latter closes the
        hole where the best-mAP epoch was never a best-val-loss epoch. Ties
        break toward the EARLIER (leaner / less-trained) checkpoint.
 
        If checkpoint_path is given, the selected best weights are also written
        there, so the on-disk artifact matches the selected model.
 
        Depends on evaluate_detections returning a comparable scalar."""
        # assemble candidate pool: (epoch -> state_dict), de-duplicated by epoch
        candidates = {}
        for epoch, state, _ in self._loss_incumbent_snapshots:
            candidates[epoch] = state
        if self._best_map_snapshot is not None:
            m_epoch, m_state, _ = self._best_map_snapshot
            candidates.setdefault(m_epoch, m_state)
 
        if not candidates:
            return self
 
        best_map = None
        best_state = None
        best_epoch = None
        # evaluate in ascending epoch order so strict > keeps the EARLIEST on ties
        for epoch in sorted(candidates):
            self.model.load_state_dict(candidates[epoch])
            score = self.evaluate_detections(loader, device)   # scalar mAP
            if best_map is None or score > best_map:
                best_map = score
                best_state = candidates[epoch]
                best_epoch = epoch
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if checkpoint_path is not None:
                torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Selected checkpoint from epoch {best_epoch} (mAP={best_map:.4f})")
        return self
 
