import os
import torch
from PIL import Image
from collections import namedtuple
##############################################
from PIL import Image, UnidentifiedImageError
import numpy as np
import re
import json

def extract_number_from_filename(file_path):
    match = re.search(r'\d+.npz$', file_path)
    if match:
        return int(match.group()[:-4])
    return 0 

def extract_number_from_filename_SM(filename):
    # Use regular expression to find the numbers in the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0
def extract_number_from_filename_SM_OB_IMG(filename):
    # Use regular expression to find the numbers in the filename
    match = re.search(r'validation_(\d+).webp', filename)
    if match:
        return int(match.group(1))
    return 0
def extract_number_from_filename_SM_OB_LABEL(filename):
    # Use regular expression to find the numbers in the filename
    match = re.search(r'validation_(\d+)_labels_semantic.png', filename)
    if match:
        return int(match.group(1))
    return 0
class val_score(torch.utils.data.Dataset):
    def __init__(self, root):
        """Load all filenames."""
        self.root = root
        self.images = []  # list of all raw input images
        self.score = []
        self.labels = []  # list of all ground truth TrainIds images
        filenames = os.listdir(os.path.join(root, 'score'))
        filenames = sorted(filenames, key=extract_number_from_filename)
        # root = os.path.join(root, 'original')
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.npz':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", 'image_' + f_name)
                filename_base_score = os.path.join("score", f_name)
                filename_base_labels = os.path.join("labels", 'image_' + f_name)
                self.images.append(os.path.join(root, filename_base_img + '.png'))
                self.score.append(os.path.join(root, filename_base_score + '.npz'))
                self.labels.append(os.path.join(root, filename_base_labels + '.png'))
        # self.score = sorted(self.score)
    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)
    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        label = Image.open(self.labels[i]).convert('L')
        data = np.load(self.score[i])
        score = data['arr_0']  # 
        score = torch.from_numpy(score).cuda(non_blocking=True)
        score = score.unsqueeze(0).expand(3, -1, -1)
        group = [{'ori_image': image, 'ori_image_path':self.images[i],'image': score, 'label': label}]
        return group

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

class SMIYC(torch.utils.data.Dataset):
    train_id_in = 0
    train_id_out = 1

    def __init__(self, root, split):
        """Load all filenames."""
        self.images = []  # list of all raw input images
        self.score = []  # list of all ground truth TrainIds images
        self.labels = []  # list of all ground truth TrainIds images
        if split == "road_anomaly":
            self.root = os.path.join(root, "dataset_AnomalyTrack")
        elif split == "road_obstacle":
            self.root = os.path.join(root, "dataset_ObstacleTrack")
        else:
            raise FileNotFoundError("there is no subset with name {} in target dataset".format(split))
        idx = 0
        for file_prototype in os.listdir(os.path.join(self.root, "labels_masks")):
            if "color" in file_prototype:
                continue
            prefix = "validation" if split == "road_anomaly" else "validation_"
            suffix = ".jpg" if split == "road_anomaly" else ".webp"
            img_name = prefix + ''.join(filter(str.isdigit, file_prototype)) + suffix
            self.images.append(os.path.join(self.root, "images", img_name))
            self.labels.append(os.path.join(self.root, "labels_masks", file_prototype))
            self.score.append(os.path.join(self.root, "score", str(idx) + ".npz"))
            # self.score.append(os.path.join(self.root, "score_pebal", str(idx) + ".npz"))
            idx += 1
        if split == "road_anomaly":
            self.images = sorted(self.images, key=lambda x: os.path.basename(x))
            self.labels = sorted(self.labels, key=lambda x: extract_number_from_filename_SM(os.path.basename(x)))
        elif split == "road_obstacle":
            self.images = sorted(self.images, key=extract_number_from_filename_SM_OB_IMG)
            self.labels = sorted(self.labels, key=extract_number_from_filename_SM_OB_LABEL)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        try:
            image = Image.open(self.images[i]).convert('RGB')
        except UnidentifiedImageError:
            print('please install the webp with cmd: pip install webp'
                  'and re-install pillow with cmd: pip install --upgrade --force-reinstall Pillow')
            from PIL import features
            assert features.check_module('webp'), "webp error."
    
        label = Image.open(self.labels[i]).convert('L')
        data = np.load(self.score[i])
        score = data['arr_0']
        score = torch.from_numpy(score).cuda(non_blocking=True)
        score = score.unsqueeze(0).expand(3, -1, -1)
        group = [{'ori_image': image,'ori_image_path':self.images[i], 'image': score, 'label': label}]
        return group
    
class RoadAnomaly(torch.utils.data.Dataset):
    def __init__(self, root="/home/yu/yu_ssd/road_anomaly"):
        """Load all filenames."""
        self.root = root
        self.images = []  # list of all raw input images
        self.labels = []  # list of all ground truth TrainIds images
        self.score = []
        filenames = os.listdir(os.path.join(root, 'original'))
        idx=0
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.jpg':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", f_name)
                filename_base_labels = os.path.join("labels", f_name)
                self.images.append(os.path.join(self.root, filename_base_img + '.jpg'))
                self.labels.append(os.path.join(self.root, filename_base_labels + '.png'))
                self.score.append(os.path.join(self.root,"score",str(idx) + '.npz'))
                # self.score.append(os.path.join(self.root,"score_pebal",str(idx) + '.npz'))
                idx+=1
        self.images = sorted(self.images)
        self.labels = sorted(self.labels)
    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        label = Image.open(self.labels[i]).convert('L')
        data = np.load(self.score[i])
        score = data['arr_0'] 
        score = torch.from_numpy(score).cuda(non_blocking=True)
        score = score.unsqueeze(0).expand(3, -1, -1)
        group = [{'ori_image': image,'ori_image_path':self.images[i], 'image': score, 'label': label}]
        return group

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road anomaly Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()
    
class input_(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.images = os.listdir(root)
        # root = os.path.join(root, 'original')
        # self.score = sorted(self.score)
    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)
    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        name = self.images[i]
        path = os.path.join(self.root,name)
        ori_image = Image.open(path).convert('RGB')
        shortened_filenames = name[:4]
        name = os.path.splitext(name)[0]
        if self.transform is not None:
            image, target = self.transform(ori_image, ori_image)
        return ori_image, image, shortened_filenames, name
    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

