# Weakly supervised instance segmentation on S3DIS and ScanNet


## Abstract of this work

 This work presents a general and simple framework to tackle point clouds understanding when labels are limited. The first contribution is that we have done extensive methodology comparisons of traditional and learnt 3D descriptors for the task of weakly supervised 3D scene understanding, and validated that our adapted traditional PFH-based 3D descriptors show excellent generalization ability across different domains. The second contribution is that we proposed a learning-based region merging strategy based on the affinity provided by both the traditional/learnt 3D descriptors and learnt semantics. The merging process takes both low-level geometric and high-level semantic feature correlations into consideration.  Experimental results demonstrate that our framework has the best performance among the three most important weakly supervised point clouds understanding tasks including semantic segmentation, instance segmentation, and object detection.

## News

* **20 November 2022**: All main Codes and models are released!


## Code structure
Our RM3D adapts the structure of the codebase [Mix3D](https://github.com/kumuji/mix3d) which provides a highly modularized framework for 3D Semantic Segmentation based on the MinkowskiEngine.

```
├── rm3d
│   ├── main_instance_segmentation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- RM3D modules
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
```

### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.10.6
cuda: 11.6
```
You can set up a conda environment as follows
```
conda create --name=RM3D python=3.10.6
conda activate RM3D

conda update -n base -c defaults conda
conda install openblas-devel -c anaconda

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

pip install ninja==1.10.2.3
pip install pytorch-lightning fire imageio tqdm wandb python-dotenv pyviz3d scipy plyfile scikit-learn trimesh loguru albumentations volumentations

pip install antlr4-python3-runtime==4.8
pip install black==21.4b2
pip install omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

cd third_party/pointnet2 && python setup.py install
```

### Data preprocessing :hammer:
After installing the dependencies, we preprocess the datasets.

#### ScanNet 
First, we apply Felzenswalb and Huttenlocher's Graph Based Image Segmentation algorithm to the test scenes using the default parameters.
Please refer to the [original repository](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for details.
Put the resulting segmentations in `./data/raw/scannet_test_segments`.
```
python datasets/preprocessing/scannet_preprocessing.py preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="../../data/processed/scannet" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
```

#### S3DIS
The S3DIS dataset contains some smalls bugs which we initially fixed manually. We will soon release a preprocessing script which directly preprocesses the original dataset. For the time being, please follow the instructions [here](https://github.com/JonasSchult/RM3D/issues/8#issuecomment-1279535948) to fix the dataset manually. Afterwards, call the preprocessing script as follows:

```
python datasets/preprocessing/s3dis_preprocessing.py preprocess \
--data_dir="PATH_TO_Stanford3dDataset_v1.2" \
--save_dir="../../data/processed/s3dis"
```

<!-- #### STPLS3D
```
python datasets/preprocessing/stpls3d_preprocessing.py preprocess \
--data_dir="PATH_TO_STPLS3D" \
--save_dir="../../data/processed/stpls3d"
``` -->

### Training and testing 
Train RM3D on the ScanNet dataset:
```bash
python main_instance_segmentation.py
```
Please refer to the [config scripts](https://github.com/JonasSchult/RM3D/tree/main/scripts) (for example [here](https://github.com/JonasSchult/RM3D/blob/main/scripts/scannet/scannet_val.sh#L15)) for detailed instructions how to reproduce our results.
In the simplest case the inference command looks as follows:
```bash
python main_instance_segmentation.py \
general.checkpoint='PATH_TO_CHECKPOINT.ckpt' \
general.train_mode=false
```

## Trained checkpoints :floppy_disk:
We provide detailed scores and network configurations with trained checkpoints.

### [S3DIS](http://buildingparser.stanford.edu/dataset.html) (pretrained on ScanNet train+val)
Following PointGroup, HAIS and SoftGroup, we finetune a model pretrained on ScanNet. Here we provided the models trained with 1% labels. More circumstances with diverse labeling percentage will be provided. Please stay tuned. 

| Dataset | AP_50 | Config | Checkpoint :floppy_disk: 
|:-:|:-:|:-:|:-:|
| Area 1 | 54.9 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://hkustgz-my.sharepoint.com/:u:/g/personal/kangchengliu_hkust-gz_edu_cn/EfPaO83QkO5Ol2qwMUSjZKkBRibz3TuhSYjk6WpbaIYOdA?e=jsXaoO) 
| Area 2 | 53.6 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://hkustgz-my.sharepoint.com/:u:/g/personal/kangchengliu_hkust-gz_edu_cn/Ea8BbjsXAdFJiDxQacvqLCwB2QX5z98NMxVR4bKuT92E7w?e=9dzHtO) 
| Area 3 | 51.7 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://hkustgz-my.sharepoint.com/:u:/g/personal/kangchengliu_hkust-gz_edu_cn/Eb8GEtMUChZGnxqb6p1uMyIBQhEyV0QVpVMalMTGqYH-Wg?e=PJoJ3x) 
| Area 4 | 58.9 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://hkustgz-my.sharepoint.com/:u:/g/personal/kangchengliu_hkust-gz_edu_cn/Ee-KpKkJL6JOjJlorhR22hQBvL2iht8lODIYzeV9ho0vWQ?e=Wz6cAc) 
| Area 5 | 57.6 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://hkustgz-my.sharepoint.com/:u:/g/personal/kangchengliu_hkust-gz_edu_cn/Eah0XjyqKQ5BhqyeYMdWPMEBSeeGOHYgBztR_GXEtEgOuw?e=Maz57q) 
| Area 6 | 56.2 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://hkustgz-my.sharepoint.com/:u:/g/personal/kangchengliu_hkust-gz_edu_cn/ER9xo_OxSeJLsj2SJJNOGDgBcEajYDt4oWAMMreEbIeAmA?e=aayUTS) 



### [ScanNet v2](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap)
 Here we provided the models trained with 1% labels on ScanNet v2. More circumstances with diverse labeling percentage will be provided. Please stay tuned. 


| Dataset  | AP_50 | Config | Models :floppy_disk:
|:-:|:-:|:-:|:-:|
| ScanNet val  | 55.7 | [config](scripts/scannet/scannet_val.sh) | [checkpoint](https://entuedu-my.sharepoint.com/:u:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EUZb4ZB-XAdFhECafEd3euQBZxN8bnqCmjjkKF2-Fq2z4w?e=yuzCwX) 


