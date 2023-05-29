import fiftyone as fo
import fiftyone.utils.random as four

from typing import Dict, Union, List, Tuple
from pathlib import Path
from tqdm import tqdm


def parse_yolo_file(yolo_annotation_path: Union[str, Path], classes_mapping: dict) -> List[Dict]:
    
    groups = {}
    for group, class_ids in classes_mapping.items():
        for class_id in class_ids:
            groups[class_id] = group
    
    annotations = []
    with open(yolo_annotation_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            annotation = {
                'bbox': map(float, [x_center, y_center, width, height]),
                'label': groups[int(class_id)]
            }
            annotations.append(annotation)

    return annotations


def get_dataset_from_yolo_detections(
    detections_path: str,
    images_dir: str,
    classes_mapping: dict,
) -> fo.Dataset:
    
    detections_path  = Path(detections_path)
    samples = []
    dataset = fo.Dataset()
    dataset.default_classes = list(classes_mapping.keys())

    for image_path in tqdm(Path(images_dir).iterdir()):
        ann_path = detections_path / f"{image_path.stem}.txt"
        
        detections = []
        sample = fo.Sample(filepath=image_path)

        if ann_path.exists():
            anns = parse_yolo_file(ann_path, classes_mapping)
            for line in anns:
                x, y, w, h = line["bbox"]
                cls = line["label"]
                bbox = [x - w / 2, y - h / 2, w, h]
                
                detections.append(fo.Detection(label=cls, bounding_box=bbox))
                
            sample["detections"] = fo.Detections(detections=detections)
        else:
            sample["detections"] = fo.Detections(detections=[])
        samples.append(sample)
    dataset.add_samples(samples)
    
    return dataset


def split_dataset(dataset: fo.Dataset, train_ratio: float = 0.8) -> Tuple[fo.Dataset, fo.Dataset]:
    return four.random_split(dataset, [train_ratio, 1 - train_ratio], seed=42)


def save_dataset(dataset: fo.Dataset, annotations_dir: str, dataset_name: str):
    
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(exist_ok=True)
    labels_path = annotations_dir / f"{dataset_name}.json"
    
    dataset.export(
        labels_path=labels_path,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="detections",
        overwrite=True,
    )


def create_splits(
    yolo_labels_path: str,
    images_dir: str,
    classes_mapping: Dict,
    dataset_save_dir: str,
    train_split_ratio: float = 0.8,
):
    dataset = get_dataset_from_yolo_detections(
        detections_path=yolo_labels_path,
        images_dir=images_dir,
        classes_mapping=classes_mapping
    )
    
    train_dataset, val_dataset = split_dataset(dataset, train_split_ratio)
    
    for dataset, dataset_name  in zip((train_dataset, val_dataset), ('train', 'val')):
        save_dataset(dataset, dataset_save_dir, dataset_name)
