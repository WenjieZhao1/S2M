import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import random
# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import torch
import numpy ,cv2
import random, os, json
from detectron2.utils.aug import *
from skimage.measure import regionprops
from detectron2.utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape, random_mirror, normalize, random_scale, \
    center_crop_to_shape, pad_image_to_shape
import pycocotools

__all__ = ["load_voc_instances", "get_score_offline"]

def get_score_offline(img_dir):
    json_file = os.path.join(img_dir, "ood.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        filename = os.path.join(img_dir, "offline_dataset", v["file_name"])
        new_filename = os.path.splitext(filename)[0] + '.png'
        height, width = cv2.imread(new_filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["annotations"]
        objs = []
        obj = {
            "bbox": [annos['bbox'][0], annos['bbox'][1], annos['bbox'][2], annos['bbox'][3]],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 254,
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("OOD_score" + d, lambda d=d: get_score_offline("/path_to_offline_dataset_score"))
    MetadataCatalog.get("OOD_score" + d).set(thing_classes=["OOD"])
balloon_metadata = MetadataCatalog.get("balloon_train")