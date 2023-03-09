## S3DIS dataset

1\) Download the [S3DIS](http://buildingparser.stanford.edu/dataset.html#Download) dataset

2\) Put the ``Stanford3dDataset_v1.2.zip`` to ``SoftGroup/dataset/s3dis/`` folder and unzip

3\) Preprocess data
```
cd SoftGroup/dataset/s3dis
bash prepare_data.sh
```

After running the script the folder structure should look like below
```
SoftGroup
├── dataset
│   ├── s3dis
│   │   ├── Stanford3dDataset_v1.2
│   │   ├── preprocess
│   │   ├── preprocess_sample
│   │   ├── val_gt
```

## ScanNet v2 dataset

1\) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

2\) Put the downloaded ``scans`` and ``scans_test`` folder as follows.

```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

3\) Split and preprocess data
```
cd SoftGroup/dataset/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.
```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## STPLS3D dataset
1\) Download the [STPLS3D](https://www.stpls3d.com/) dataset

2\) Put ``Synthetic_v3_InstanceSegmentation.zip`` to ``dataset/stpls3d`` and unzip

3\) Preprocess data
```
cd SoftGroup/dataset/stpls3d
bash prepare_data.sh
```

## SemanticKITTI dataset
1\) Download the [SemanticKITTI](http://www.semantic-kitti.org/dataset.html) dataset

2\) Unzip the downloaded data and put the ``sequences`` to ``dataset/kitti``

3\) The data structure should be as follows:
```
SoftGroup
├── dataset
│   ├── kitti
│   │   ├── sequences
│   │   |   |── 00
│   │   |   |   ├── calib.txt
│   │   |   |   ├── labels
│   │   |   |   ├── poses.txt
│   │   |   |   ├── times.txt
│   │   |   |   ├── velodyne
│   │   |   |── ...
│   │   |   |── 21
```
