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
from cityscape import Cityscapes
from coco_ import COCO
from PIL import Image
import numpy
import random
from detectron2.utils.aug import *
from skimage.measure import regionprops
from detectron2.utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape, random_mirror, normalize, random_scale, \
    center_crop_to_shape, pad_image_to_shape
import pycocotools
__all__ = ["load_voc_instances", "get_OE_dataset_dicts"]

city_dir = '/path_to_cityscapes_dataset/cityscapes'
coco_dir = '/path_to_COCO'
image_mean = numpy.array([0.485, 0.456, 0.406])
image_std = numpy.array([0.229, 0.224, 0.225])
city_image_height = 700
city_image_width = 700
ood_image_height = 700
ood_image_width = 700
ood_train_scale_array = [.25, .5, .5, .75, .1, .125]
image_mean = numpy.array([0.485, 0.456, 0.406])
image_std = numpy.array([0.229, 0.224, 0.225])
import cv2
import torch

class Preprocess(object):
    def __init__(self, ):
        self.img_mean = image_mean
        self.img_std = image_std
        self.city_image_height = city_image_height
        self.city_image_width = city_image_width
        self.ood_image_height = ood_image_height
        self.ood_image_width = ood_image_width
        self.ood_scale_array = ood_train_scale_array

    def inlier_transform(self, img_, gt_):
        img_, gt_ = random_mirror(img_, gt_)
        img_ = normalize(img_, self.img_mean, self.img_std)
        crop_size = (self.city_image_height, self.city_image_width)
        crop_pos = generate_random_crop_pos(img_.shape[:2], crop_size)
        img_, _ = random_crop_pad_to_shape(img_, crop_pos, crop_size, 0)
        gt_, _ = random_crop_pad_to_shape(gt_, crop_pos, crop_size, 255)
        return img_, gt_

    def outlier_transform(self, img_, gt_):
        img_ = normalize(img_, self.img_mean, self.img_std)
        scaled_img_, scaled_gt_, _ = random_scale(img_.copy(), gt_.copy(), self.ood_scale_array)
        if img_.shape[0] > self.ood_image_width or img_.shape[1] > self.ood_image_height:
            img_, gt_ = center_crop_to_shape(img=img_, gt=gt_, size=(self.ood_image_height, self.ood_image_width))
        else:
            img_, _ = pad_image_to_shape(img_, (self.ood_image_height, self.ood_image_width), value=0)
            gt_, _ = pad_image_to_shape(gt_, (self.ood_image_height, self.ood_image_width), value=255)

        if scaled_img_.shape[0] > self.ood_image_width or scaled_img_.shape[1] > self.ood_image_height:
            scaled_img_, scaled_gt_ = center_crop_to_shape(img=scaled_img_, gt=scaled_gt_, size=(self.ood_image_height, self.ood_image_width))
        else:
            scaled_img_, _ = pad_image_to_shape(scaled_img_, (self.ood_image_height, self.ood_image_width), value=0)
            scaled_gt_, _ = pad_image_to_shape(scaled_gt_, (self.ood_image_height, self.ood_image_width), value=255)

        return img_, gt_, scaled_img_, scaled_gt_

    def __call__(self, city_img, city_gt, ood_img, ood_gt, anomaly_mix_or_not):
        city_img, city_gt = self.inlier_transform(city_img, city_gt)
        ood_img, ood_gt, scaled_ood_img, scaled_ood_gt = self.outlier_transform(img_=ood_img, gt_=ood_gt)
        assert ood_img.shape == city_img.shape and ood_img.shape == scaled_ood_img.shape, \
            print("ood_img.shape {}, city_img.shape {}".format(
            ood_img.shape, city_img.shape
        ))

        assert city_gt.shape == ood_gt.shape, print("ood_img.shape {}, city_img.shape {}".format(
            ood_gt.shape, city_gt.shape
        ))

        if anomaly_mix_or_not:
            city_mix_img, city_mix_gt = self.mix_object(current_labeled_image=city_img.copy(),
                                                        current_labeled_mask=city_gt.copy(),
                                                        cut_object_image=scaled_ood_img,
                                                        cut_object_mask=scaled_ood_gt)
        else:
            city_mix_img = numpy.zeros_like(city_img)
            city_mix_gt = numpy.zeros_like(city_gt)

        return city_img.transpose(2, 0, 1), city_gt, city_mix_img.transpose(2, 0, 1), city_mix_gt, \
            ood_img.transpose(2, 0, 1), ood_gt

    def mix_object(self, current_labeled_image=None, current_labeled_mask=None,
                   cut_object_image=None, cut_object_mask=None):

        train_id_out = 254
        cut_object_mask[cut_object_mask == train_id_out] = 254

        mask = cut_object_mask == 254

        ood_mask = numpy.expand_dims(mask, axis=2)
        ood_boxes = self.extract_bboxes(ood_mask)
        ood_boxes = ood_boxes[0, :]
        y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
        cut_object_mask = cut_object_mask[y1:y2, x1:x2]
        cut_object_image = cut_object_image[y1:y2, x1:x2, :]
        idx = numpy.transpose(numpy.repeat(numpy.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

        h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
        h_end_point = h_start_point + cut_object_mask.shape[0]
        w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
        w_end_point = w_start_point + cut_object_mask.shape[1]

        current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][numpy.where(idx == 254)] = \
            cut_object_image[numpy.where(idx == 254)]

        current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][numpy.where(cut_object_mask == 254)] = \
            cut_object_mask[numpy.where(cut_object_mask == 254)]

        return current_labeled_image, current_labeled_mask

    @staticmethod
    def extract_bboxes(mask):
        boxes = numpy.zeros([mask.shape[-1], 4], dtype=numpy.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = numpy.where(numpy.any(m, axis=0))[0]
            vertical_indicies = numpy.where(numpy.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = numpy.array([y1, x1, y2, x2])
        return boxes.astype(numpy.int32)

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
        #####
        self.city_annotation = Cityscapes(root=cs_root, split=self.cs_split, target_type="color")
        ####
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
        record["annotations"] = obj
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
    
def get_OE_dataset_dicts(city_dir, coco_dir, split_):
    train_dataset = CityscapesCocoMix(split='train', preprocess=Preprocess(),
                                    cs_root=city_dir, coco_root=coco_dir)
    split_ratio = 0.9
    total_samples = len(train_dataset)
    train_samples = int(total_samples * split_ratio)
    test_samples = total_samples - train_samples
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_samples, test_samples])
    print("train_dataset: {}, test_dataset: {}".format(len(train_dataset), len(test_dataset)))
    return train_dataset
    
def register_dataset():
    for d in ["train", "val"]:
        DatasetCatalog.register("OE_dataset_" + d,  lambda x = city_dir, y=coco_dir, d=d: get_OE_dataset_dicts(x,y,d))
        MetadataCatalog.get("OE_dataset" + d).set(thing_classes=["OOD"])
    balloon_metadata = MetadataCatalog.get("OE_dataset_train")