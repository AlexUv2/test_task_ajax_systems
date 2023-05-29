# Tesk Task Ajax Systems

## Goals:
- Label images with the following classes: [`person`, `car`, `pet`]
- Split labeled images into train and val sets
- Process dataset with FiftyOne


### Classes
Classes for detecting were chosen manualy from model classes. See utils.get_model_classes(). 

# Setup
- Create suitable python environment
- Install requirements. Run `pip install requirements.txt`

# Usage Example
run 
```
python main.py \
    --src_images_dir /path/to/images \
    --detections_dir /path/to/detections \
    --infer_conf configs/inference_config.yaml(optional) \
    --cls_map_path configs/classes_to_detect.yaml(optional) \
    --annotations_dir /path/to/result/annotations \
    --train_split_ratio 0.8(optional)
```

## [Results](https://drive.google.com/file/d/1M_4NvX-mmaBKI932ptQgkn4m-zmmEpTg/view?usp=share_link)

