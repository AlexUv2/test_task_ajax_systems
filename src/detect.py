import os
import cv2
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Union
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from tqdm import tqdm
from src.utils import load_yaml
import numpy as np


def detect_images(
    images_dir: str,
    detections_dir: str,
    model_type: str,
    device_id: int = 0,
    confidence_th: float = 0.25,
    iou_th: float = 0.7,
    classes_to_detect: list = None,
):
    detections_dir = Path(detections_dir)
    detections_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = YOLO(model_type)
    model.to(device)
    
    
    for image_path in Path(images_dir).iterdir():
        results = model.predict(
            source=image_path,
            conf=confidence_th,
            iou=iou_th,
            classes=classes_to_detect,
            stream=True,
        )
        
        detections = []
        relust_file_path = detections_dir / f"{image_path.stem}.txt"
        for result in results:
            boxes = result.boxes.xywhn.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().reshape(-1, 1)
            result = np.hstack((classes, boxes))
            result = result.tolist()
            
            for pred in result:
                pred[0] = int(pred[0])
                pred[1:] = list(map(lambda x: round(x, 7), pred[1:]))
            detections.extend(result)
            
        with open(relust_file_path, "w") as f:
            for row in detections:
                f.write(" ".join(map(str, row)) + "\n")
