# Segment Every Out-of-Distribution Object
**S**core **T**o **M**ask (**S2M**) is a simple and efficienty way to utilize anomaly score from current mainstream methods and improve their performance. Experiments demonstrate that S2M outperforms the state-of-the-art by approximately 20\% in IoU and 40\% in mean F1 score, on average.

![](/docs/final.png)

> [**Segment Every Out-of-Distribution Object**](https://arxiv.org/abs/2311.16516v3)            
> [Wenjie Zhao](https://www.linkedin.com/in/wenjie-zhao-7290b4298/), [Jia Li](https://github.com/LONZARK/), [Xin Dong](https://simonxin.com/), [Yu Xiang](https://yuxng.github.io/), [Yunhui Guo](https://yunhuiguo.github.io/)     
> UT Dallas, Harvard          
> CVPR 2024

[[`arxiv`](https://arxiv.org/abs/2311.16516v3)] [[`bibtex`](#citation)] 

## Features
* We propose S2M, a simple and general pipeline to generate the precise mask for OoD objects.
* It eliminates the need to manually choose an optimal threshold for generating segmentation masks.
* Our method is general and independent of particular anomaly scores, prompt generators, or promptable segmentation models.
* S2M didn't produce any mask on the ID picture.

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