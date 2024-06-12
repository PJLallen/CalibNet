import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

CLASS_NAMES =["foreground"]

PREDEFINED_SPLITS_DATASET = {
    "come15k_train": ("COME15K/train/imgs_right", "COME15K/annotations/COME15K-Train.json"),
    "come15k_test_e": ("COME15K/COME-E/RGB", "COME15K/annotations/COME15K-Test-E.json"),
    "come15k_test_h": ("COME15K/COME-H/RGB", "COME15K/annotations/COME15K-Test-H.json"), 
    "sip_test": ("SIP/RGB", "SIP/SIP.json"),
    "dsis_test": ("DSIS/RGB", "DSIS/DSIS.json")
}

def register_dataset_instances(name,json_file,image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")

def register_rgbdsis_datasets(root):
    for key,(image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, json_file=os.path.join(root,json_file), image_root=os.path.join(root,image_root))

_root = os.getenv("DETECTRON2_DATASETS", "path/to/dataset/root")

register_rgbdsis_datasets(_root)