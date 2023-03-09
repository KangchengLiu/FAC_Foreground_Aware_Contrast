We release our PointNet++ and MinkowskiEngine UNet models pretrained with our proposed FAC with the hope that other researchers might also benefit from these pretrained backbones. Due to license issue, models pretrained on Waymo cannot be released. For PointnetMSG and Spconv-UNet models, we encourage the researchers to train by themselves using the provided script.

We first provide PointNet++ models with different sizes.
| network | epochs | batch-size | ScanNet Det with VoteNet | url | args |
|-------------------|---------------------|---------------------|--------------------|--------------------|--------------------|
| PointNet++-1x | 150 | 2048 | 61.9 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EWpv8MbdgjtJsSKFgxsj2mYBxKq5vNgCE48dwu5y0Bggvg?e=lM713a) | [config](./configs/point_within_format.yaml) |
| PointNet++-2x | 200 | 4096 | 63.3 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EX7BJH6P8a9Erk9gUTlstUIB0GFNoOTSgqI9DxA9fV5gGQ?e=Cik00b) | [config](./configs/point_within_format.yaml) |
| PointNet++-3x | 150 | 4096 | 64.1 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EfGgJ6QWiu1Jk6t8c1IfKj4BF80lvygNYxpTAWyHORdYgg?e=1dVTZF) | [config](./configs/point_within_format.yaml) |
| PointNet++-4x | 100 | 4096 | 63.8 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EUUuqGq-XjhMkySHovezBbQBTiL4fP3jEXlkIP7sxenx0g?e=BhpT5t) | [config](./configs/point_within_format.yaml) |

The ScanNet detection evaluation metric is mAP at IOU=0.25. You need to change the scale parameter in the config files accordingly.

We provide the joint training results here, with different epochs. We use epoch 400 to generate the results reported in the paper.

| Backbone | epochs | batch-size | url | args |
|-------------------|-------------------|---------------------|--------------------|--------------------|
| PointNet++ & MinkowskiEngine UNet | 300 | 3192 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EYp4uV3TcxBBm7EysOUyNKQBEyjCfWeeB6SezZ27T_Ktqg?e=bM90lJ) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 400 | 1024 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/Ed-6xnMQ4jVDmr3XbpHGrNABEID0akAyfrWkTAI0mINSHg?e=nGjlot) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 500 | 1024 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EbMWdobnuHFCommOT0icvDYBMGhSYF3Db7PQFbXZTedscA?e=rijn27) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 600 | 1024 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EeODVMuNtNNLpQvnvP-YlQEBBdk2mknwZdzT8e-9Wc71Ig?e=AE0hsp) | [config](./configs/point_vox_template.yaml) |
| PointNet++ & MinkowskiEngine UNet | 700 | 1024 | [model](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EVk66W67EvFBuyw3BpK4swIBUtAzf-B0-fmjmMGW3MHZvg?e=psTH6c) | [config](./configs/point_vox_template.yaml) |

# Running the unsupervised pre-training

## Requirements
You can use the requirements.txt to setup the environment.
First download the git-repo and install the pointnet modules:
```
git clone --recursive https://github.com/facebookresearch/DepthContrast.git 
cd pointnet2
python setup.py install
```
Then install all other packages:
```
pip install -r requirements.txt
```
or
```
conda install --file requirements.txt
```

For voxel representation, you have to install MinkowskiEngine. Please see [here](https://github.com/chrischoy/SpatioTemporalSegmentation) on how to install it.

For the lidar point cloud pretraining, we use models from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). It should be in the third_party folder. To install OpenPCDet, you need to install [spconv](https://github.com/traveller59/spconv), which is a bit difficult to install and may not be compatible with MinkowskiEngine. Thus, we suggest you use a different conda environment for lidar point cloud pretraining. 

## Singlenode training

To experiment with it on one GPU and debugging, you can do:
```
python main.py /path/to/cfg/file
```

For the actual training, please use the distributed trainer. 
For multi-gpu training in one node, you can run:
```
python main.py /path/to/cfg_file --multiprocessing-distributed --world-size 1 --rank 0 --ngpus number_of_gpus
```
To run it with just one gpu, just set the --ngpus to 1.
For submitting it to a slurm node, you can use ./scripts/pretrain_node1.sh. For hyper-parameter tuning, please change the config files.

## Multinode training
Distributed training is available via Slurm. We provide several [SBATCH scripts](./scripts) to reproduce our results.
For example, to train FAC on 4 nodes and 32 GPUs with a batch size of 1024 run:
```
sbatch ./scripts/pretrain_node4.sh /path/to/cfg_file
```
Note that you might need to remove the copyright header from the sbatch file to launch it.

