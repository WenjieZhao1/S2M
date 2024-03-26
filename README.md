## Preparation
Download the [checkpoint](https://drive.google.com/file/d/1r0U2yQYHBcqjxs162h5RAaDwDyMpEQqw/view?usp=drive_link) file and put it in `./tools`.

Download checkpoint file of [SAM-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put it in `./tools`.

Download [validation dataset](https://drive.google.com/file/d/1IbD_zl5MecMCEj6ozGh40FE5u-MaxPLh/view?usp=sharing). It should look like this:
```
${PROJECT_ROOT}
 -- val
     -- fishyscapes
         ...
     -- road_anomaly
         ...
     -- segment_me
         ...
```

Download [train set](https://drive.google.com/file/d/1k25FpVP4pG3ER3eXsR-go_iprZMEdEae/view?usp=sharing).It should look like this: 
```
 -- train_dataset
     -- offline_dataset
         ...
     -- offline_dataset_score
         ...
     -- offline_dataset_score_view
         ...
     -- ood.json
```


## Install the environment
Please create a environment with pytoch == 2.0.1 and install package in the requirements.txt. 
Then install detectron2 with our S2M by following:
1. Get out of S2M folder.
2. Install environment of S2M by follow.
```
python -m pip install -e S2M
```
## Training
Set the detail of training in `configs/OE/OE.yaml`.
```
cd ./tools
python3 plain_train_net.py   --config-file ../configs/OE/OE.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025
```
## Evaluation
Set the path of dataset in ./tools/inference.py line 256.
```
cd ./tools
python3 inference.py   --config-file ../configs/OE/OE.yaml   --eval-only MODEL.WEIGHTS /path_to/model.pth
```
## Acknowledgement
Our project is implemented base on the following projects. We really appreciate their excellent open-source works!
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [RPL](https://github.com/yyliu01/RPL)

## Citation

If our work has been helpful to you, we would greatly appreciate a citation.
```
@article{zhao2023segment,
  title={Segment Every Out-of-Distribution Object},
  author={Zhao, Wenjie and Li, Jia and Dong, Xin and Xiang, Yu and Guo, Yunhui},
  journal={arXiv preprint arXiv:2311.16516},
  year={2023}
}
```