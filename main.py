import os
import argparse

from src.convert_detections import create_splits
from src.detect import detect_images
from src.view_dataset import view_datasets
from src.utils import load_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_images_dir", type=str, required=True)
    parser.add_argument("--detections_dir", type=str, required=True)
    parser.add_argument("--infer_conf", type=str, default="configs/inference_config.yaml")
    parser.add_argument("--cls_map_path", type=str, default="configs/classes_to_detect.yaml")
    parser.add_argument("--ann_dir", type=str, default="labels")
    parser.add_argument("--train_split_ratio", type=float, default=0.8)
    
    return parser.parse_args()


def main(
    src_images_dir: str,
    detections_dir: str,
    inference_config_path: str,
    classes_mapping_path: str,
    annotations_dir: str = 'labels',
    train_split_ratio: float = 0.8,
):
    
    inference_config = load_yaml(inference_config_path)
    classes_mapping = load_yaml(classes_mapping_path)

    detect_images(
        images_dir=src_images_dir,
        detections_dir=detections_dir,
        model_type=inference_config['model_type'],
        confidence_th=inference_config['conf_th'],
        iou_th=inference_config['iou_th'],
        classes_to_detect=list(classes_mapping['classes'].keys()),
        device_id=inference_config['device_id'],
    )

    create_splits(
        yolo_labels_path=detections_dir,
        images_dir=src_images_dir,
        classes_mapping=classes_mapping['groups'],
        dataset_save_dir=annotations_dir,
        train_split_ratio=train_split_ratio,
    )

    view_datasets(
        data_path=src_images_dir,
        train_coco_path=os.path.join(annotations_dir, 'train.json'),
        val_coco_path=os.path.join(annotations_dir, 'val.json'),
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.src_images_dir,
        args.detections_dir,
        args.infer_conf,
        args.cls_map_path,
        args.ann_dir,
        args.train_split_ratio,
    )
    

