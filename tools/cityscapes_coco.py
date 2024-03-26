import torch
import numpy
import numpy as np
import random
from PIL import Image
from detectron2.structures import BoxMode

import pycocotools
# from ..detectron2.structures import Instances
from training.cityscapes import Cityscapes
from training.coco import COCO

class CityscapesCocoMix(torch.utils.data.Dataset):
    def __init__(self, split, preprocess, cs_root='', coco_root="", cs_split=None, coco_split=None):

        self._split_name = split
        self.preprocess = preprocess

        if cs_split is None or coco_split is None:
            self.cs_split = split
            self.coco_split = split
        else:
            self.cs_split = cs_split
            self.coco_split = coco_split
        self.city = Cityscapes(root=cs_root, split=self.cs_split)
        self.city_annotation = Cityscapes(root=cs_root, split=self.cs_split, target_type="color")
        self.coco = COCO(root=coco_root, split=self.coco_split,
                         proxy_size=int(len(self.city)))
        self.city_number = len(self.city.images)
        self.ood_number = len(self.coco.images)
        self.train_id_out = self.coco.train_id_out
        self.num_classes = self.city.num_train_ids
        self.mean = self.city.mean
        self.std = self.city.std
        self.void_ind = self.city.ignore_in_eval_ids

    def __getitem__(self, idx):
        city_idx, anomaly_mix_or_not = idx[0], idx[1]
        # city_idx = idx
        """Return raw image, ground truth in PIL format and absolute path of raw image as string"""
        city_image = numpy.array(Image.open(self.city.images[city_idx]).convert('RGB'), dtype=numpy.float64)
        #city_target = numpy.array(Image.open(self.city.targets[city_idx]).convert('L'), dtype=numpy.long)
        city_target = numpy.array(Image.open(self.city_annotation.targets[city_idx]).convert('L'), dtype=numpy.long)
        ood_idx = random.randint(0, self.ood_number-1)
        ood_image = numpy.array(Image.open(self.coco.images[ood_idx]).convert('RGB'), dtype=numpy.float)
        ood_target = numpy.array(Image.open(self.coco.targets[ood_idx]).convert('L'), dtype=numpy.long)
        city_image, city_target, city_mix_image, city_mix_target, \
            ood_image, ood_target = self.preprocess(city_image, city_target, ood_image, ood_target,
                                                    anomaly_mix_or_not=anomaly_mix_or_not)
        record = {}
        record["file_name"] = "City" + str(city_idx) + "Ood" + str(ood_idx)
        record["image_id"] = idx
        record["height"] = city_mix_image.shape[1]
        record["width"] = city_mix_image.shape[2] 
        image = torch.from_numpy(city_mix_image)
        image_ = (image.permute(1,2,0) * numpy.array([0.229, 0.224, 0.225])) + numpy.array([0.485, 0.456, 0.406]) *255
        image_ = image_.to(torch.uint8)
        record["image"] = image_.permute(2,0,1)
        obj = []
        mask = (city_mix_target == 254)
        points = np.argwhere(mask)
        if len(points) > 0:
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            obj = {
                "bbox": [min_x, min_y, max_x, max_y],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 254,
                "segmentation": pycocotools.mask.encode(np.asarray(mask.astype(np.uint8), order="F")),
            }
        record["instances"] = obj
        return record
    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.city.images)

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Cityscapes Split: %s\n' % self.cs_split
        fmt_str += '----Number of images: %d\n' % len(self.city)
        fmt_str += 'COCO Split: %s\n' % self.coco_split
        fmt_str += '----Number of images: %d\n' % len(self.coco)
        return fmt_str.strip()
