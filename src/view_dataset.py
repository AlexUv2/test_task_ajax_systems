import fiftyone as fo


def view_datasets(
    data_path: str,
    train_coco_path: str,
    val_coco_path: str,
):
    train_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=train_coco_path,
    )
    
    val_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=val_coco_path,
    )
    
    combined_dataset = fo.Dataset()
    
    for sample in train_dataset:
        sample["split"] = "train"
        combined_dataset.add_sample(sample)

    for sample in val_dataset:
        sample["split"] = "val"
        combined_dataset.add_sample(sample)

    session = fo.launch_app(combined_dataset)
    session.wait()
