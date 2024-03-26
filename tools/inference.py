#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from datasets.validation.val_score import *
from utils.pyt_utils import eval_ood_measure
from utils.img_utils import Compose, Normalize, ToTensor
from tqdm import tqdm
import torchvision
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    build_OE_detection_train_loader,
)
from model.network import Network
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from config import config
import numpy as np
logger = logging.getLogger("detectron2")
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import warnings
warnings.filterwarnings("ignore")
import numpy
import json
from sklearn.metrics import confusion_matrix, f1_score
sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")  ########Modify the sam model path.#############
def calculate_miou(y_true, y_pred, num_classes):
    ious = []
    f1_scores = []
    for i in range(len(y_true)):
        intersection = torch.logical_and(y_true[i]==1,y_pred[i]==1).sum().item()
        union = torch.logical_or(y_true[i]==1,y_pred[i]==1).sum().item()
        if union == 0:
            iou = 0
            f1_score = 0
        else:
            iou = intersection / union
            f1_score = (2 * intersection) / (intersection + union)
        ious.append(iou)
        f1_scores.append(f1_score)
    return np.mean(ious), numpy.mean(f1_scores)
def get_iou(anomaly, ood_gts,dataset_name,step, save_name=None):
    if isinstance(ood_gts, numpy.ndarray):
        ood_gts = torch.from_numpy(ood_gts)
    if isinstance(anomaly, numpy.ndarray):
        anomaly = torch.from_numpy(anomaly)
    i=0
    ious=[]
    Best_miou = 0
    Best_threshold=0
    threshold_list = []
    f1_scores = []
    num_set=0
    for threshold in np.arange(0,1+step, step):
        num_set += 1
        threshold_list.append(threshold)
        anomaly_mask = torch.zeros_like(anomaly)
        anomaly_mask[anomaly > threshold] = 1
        iou, f1_score = calculate_miou(ood_gts, anomaly_mask, 2)
        if iou > Best_miou:
            Best_miou = iou
            Best_threshold = threshold
        ious.append(iou)
        f1_scores.append(f1_score)   
    f1_score = np.mean(f1_scores)
    area = np.trapz(ious, threshold_list)
    save = {"ious":ious,"threshold_list":threshold_list,"area":area,"Best_miou":Best_miou,"Best_threshold":Best_threshold,"f1_scores":f1_score}
    save = {dataset_name:save}
    if save_name is None:
        save_name = 'save.json'
    if os.path.isfile(save_name) and os.path.getsize(save_name) > 0:
        with open(save_name, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    data.update(save)
    with open(save_name, 'w') as f:
        json.dump(data, f, indent=4)
    return Best_miou,Best_threshold,area,f1_score, ious, threshold_list

def compute_anomaly_score(score, mode='energy'):
    score = score.squeeze()[:19]
    if mode == 'energy':
        anomaly_score = -(1. * torch.logsumexp(score, dim=0))

    elif mode == 'entropy':
        prob = torch.softmax(score, dim=0)
        anomaly_score = -torch.sum(prob * torch.log(prob), dim=0) / torch.log(torch.tensor(19.))
    else:
        raise NotImplementedError

    # regular gaussian smoothing
    anomaly_score = anomaly_score.unsqueeze(0)
    anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(anomaly_score)
    anomaly_score = anomaly_score.squeeze(0)
    return anomaly_score
    
def get_anomaly_detector(ckpt_path):
    """
    Get Network Architecture based on arguments provided
    """
    ckpt_name = ckpt_path
    model = Network(config.num_classes)
    state_dict = torch.load(ckpt_name)
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    return model

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax,score):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1))
    ax.text(x0, y0 - 5, f'Score: {score:.2f}', color='green', fontsize=10)  

def do_test_score(cfg, model,dataset,dataset_name):
    sam.to("cuda")
    predictor = SamPredictor(sam)
    test_set = dataset
    stride = len(test_set)
    e_record = min((1) * stride, len(test_set))
    shred_list = list(range(0 * stride, e_record))
    tbar = tqdm(shred_list, ncols=80, leave=True, miniters=1)
    model.eval()
    ood_gts_list = []
    anomaly_list = []
    with torch.no_grad():
        num_set = 0
        ious = []
        anomaly_list = []
        num_set += 1
        for idx in tbar:
            input= test_set[idx]
            output = model(input)
            ood_gts_list.append(np.expand_dims(np.array(input[0]['label']), 0))
            image= input[0]['ori_image']
            image_array = np.array(image)
            if output[0]['instances'].scores.numel() == 0 :
                anomaly = np.zeros((image_array.shape[0], image_array.shape[1]))
                anomaly_list.append(np.expand_dims(anomaly, 0))
                continue
            scores = output[0]['instances'].scores
            draw = ImageDraw.Draw(image)
            plt.imshow(image)
            predictor.set_image(image_array)
            prompt = []
            prompt_score = []
            for i, box in enumerate(output[0]['instances'].pred_boxes):
                if 1:
                    prompt.append(box) 
                    prompt_score.append(scores[i].cpu())
            if len(prompt) == 0:
                anomaly = np.zeros((image_array.shape[0], image_array.shape[1]))
                anomaly_list.append(np.expand_dims(anomaly, 0))
                continue
            prompt = torch.stack(prompt)
            transformed_boxes = predictor.transform.apply_boxes_torch(prompt, image_array.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            anomaly = torch.zeros(image_array.shape[0],image_array.shape[1])
            for mask, score in zip(masks,prompt_score):
                anomaly[mask[0].cpu().numpy() ==True ] = score #       score
            anomaly_list.append(np.expand_dims(anomaly, 0))
        ood_gts = np.array(ood_gts_list)
        anomaly = np.array(anomaly_list)
        step = 0.01 
        Best_miou,Best_threshold,area,f1_score,ious,real_threshold_list = get_iou(anomaly, ood_gts,dataset_name,step)
    print("################Final result################")
    print("Area",round(area,4),"f1_score:",round(f1_score,4),"Best_miou:",round(Best_miou,4))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    ) 
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        val_root='../val/'     ########Modify the dataset path.#############
        fishyscapes_static_score = val_score(root=os.path.join(val_root,'fishyscapes/Static'))
        fishyscapes_static_LandF = val_score(root=os.path.join(val_root,'fishyscapes/LostAndFound'))
        SMYIC_anomaly = SMIYC(root=os.path.join(val_root,'segment_me'), split='road_anomaly')
        SMYIC_obstacle = SMIYC(root=os.path.join(val_root,'segment_me'), split='road_obstacle')
        Road_anomaly = RoadAnomaly(root=os.path.join(val_root,'road_anomaly'))
        print('##########Static##########')
        do_test_score(cfg, model,fishyscapes_static_score,dataset_name='Fishyscapes_static')
        print('##########Static End##########\n\n')
        print('##########L&F##########')
        do_test_score(cfg, model,fishyscapes_static_LandF,dataset_name='Fishyscapes_ls')
        print('##########L&F End##########\n\n')

        print('##########SMIYC Anomaly##########')
        do_test_score(cfg, model,SMYIC_anomaly,dataset_name='segment_me_anomaly')
        print('##########SMIYC Anomaly End##########\n\n')

        print('##########SMIYC Obstacle##########')
        do_test_score(cfg, model,SMYIC_obstacle,dataset_name='segment_me_obstacle')
        print('##########SMIYC Obstacle End##########\n\n')

        print('##########Road Anomaly##########')
        do_test_score(cfg, model,Road_anomaly,dataset_name='road_anomaly')
        print('##########Road Anomaly End##########\n\n')



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
