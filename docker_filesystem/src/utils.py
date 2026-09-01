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

class BasketballDataset(Dataset):
    """
    Custom Dataset for Basketball Object Detection.
    """
    def __init__(self, data_dir, class_mapping, transforms=None):
        self.data_dir = data_dir
        self.class_mapping = class_mapping
        self.transforms = transforms

        self.images = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))

        self.annotations = []
        for img_path in self.images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(self.data_dir, basename + '.xml')
            self.annotations.append(xml_path)

        valid_pairs = [(img, xml) for img, xml in zip(self.images, self.annotations)
                       if os.path.exists(xml)]
        self.images = [pair[0] for pair in valid_pairs]
        self.annotations = [pair[1] for pair in valid_pairs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        xml_path = self.annotations[idx]
        boxes, labels = parse_voc_xml(xml_path, self.class_mapping)

        # Filter invalid boxes (non-positive width/height) before tensor-izing
        valid_boxes = []
        valid_labels = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            if (xmax - xmin) > 1.0 and (ymax - ymin) > 1.0:
                valid_boxes.append(box)
                valid_labels.append(labels[i])

        if len(valid_boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(valid_labels, dtype=torch.int64)

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(
            boxes_t, format='XYXY', canvas_size=(img.height, img.width)
        )
        target['labels'] = labels_t
        target['image_id'] = torch.tensor([idx])
        target['iscrowd'] = torch.zeros((len(boxes_t),), dtype=torch.int64)

        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = F.to_image(img)
            img = F.to_dtype(img, torch.float32, scale=True)

        final_boxes = target['boxes']
        if final_boxes.numel() > 0:
            target['area'] = (final_boxes[:, 3] - final_boxes[:, 1]) * \
                             (final_boxes[:, 2] - final_boxes[:, 0])
        else:
            target['area'] = torch.zeros((0,), dtype=torch.float32)

        return img, target

def parse_voc_xml(xml_file, class_mapping, verbose=False):
    """
    Parse PASCAL VOC format XML annotation file.
    
    Args:
        xml_file: Path to XML annotation file
        class_mapping: Dictionary mapping class names to integers
    
    Returns:
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax]
        labels: List of class labels
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        # Get class name
        class_name = obj.find('name').text.lower()

    
        if verbose:
            print(f"raw name: '{class_name}' -> mapped: {class_mapping.get(class_name)}")
        
        # Map class name to label (default to 1 if not in mapping)
        label = class_mapping.get(class_name, 1)
        
        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    
    return boxes, labels

def visualize_sample(dataset, idx, IDX_TO_CLASS, CLASS_COLORS):
    """
    Visualize an image with bounding boxes.
    """
    img, target = dataset[idx]
    
    # Convert tensor to numpy for visualization
    img_np = img.permute(1, 2, 0).numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    
    # Draw bounding boxes
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        class_name = IDX_TO_CLASS.get(label, 'unknown')
        color = CLASS_COLORS.get(class_name, 'yellow')
        
        rect = patches.Rectangle((xmin, ymin), width, height, 
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin-5, class_name, color=color, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title(f"Sample {idx}: {len(boxes)} object(s) detected")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_transforms(is_train):
    """Return the v2 transform pipeline for detection.

    Train carries light augmentation; eval (val + inference) is deterministic.
    Both end with the same tensor conversion, since the model's internal
    transform handles resize + normalize -- so we deliberately do NOT resize or
    normalize here.
    """
    # shared deterministic tail: PIL/ndarray -> float tensor in [0,1], and keep
    # bounding boxes valid through the pipeline.
    tail = [
        v2.ToImage(),                              # -> tv_tensors.Image
        v2.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
    ]

    if not is_train:
        return v2.Compose(tail)

    # train: light, box-safe augmentation applied BEFORE the tensor tail.
    aug = [
        v2.RandomHorizontalFlip(p=0.5),            # courts are L/R symmetric
        v2.RandomPhotometricDistort(p=0.5),        # brightness/contrast/sat/hue jitter
        v2.ClampBoundingBoxes(),                   # keep boxes inside the image
        v2.SanitizeBoundingBoxes(),                # drop boxes that became degenerate
    ]
    return v2.Compose(aug + tail)