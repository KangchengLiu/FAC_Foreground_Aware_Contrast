
# Weakly supervised semantic segmentation on S3DIS and ScanNet


# Installation

This implementation has been tested on Ubuntu 18.04. For installation of the KpConv-based semantic segmentation, details are provided in [INSTALL.md](./INSTALL.md).

# Experiments

## Preparing Dataset

Processed S3DIS dataset can be downloaded <a href="https://www.dropbox.com/sh/hedb6yilh9v7fw4/AACrljL1t7SzOdx8WNPrLK84a?dl=0">here</a>. Download the dataset and move it to `Data/`.

## S3DIS Training

Simply run the following script to start the training:

        python train_S3DIS.py

## ScanNet Training


1. Download the Scannet dataset through the [official webcite](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation).

2. Extract the subcloud labels.

3. Modify the dataset path in `Scannet_subcloud.py` And start training:

```
python training_mprm.py
```
You can also specify the model saving path and change parameters in `training_mprm.py`.

4. You can plot the training details by speficy the saved model path and run:
```
python plot_convergence_mprm.py
```

5. Generate the pseudo label by running:
```
python generate_pseudo_label.py
```

6. You can choose to post-process the pseudo-label by running:
```
python crf_postprocessing.py
```


7. Finally, use the pseudo label to train a segmentation network by running:
```
python training_segmentation.py
```

## Pretrained Models

The pre-trained models for the tasks of semantic segmentation are provided when tested with 1% labels at [checkpoint](https://entuedu-my.sharepoint.com/:f:/g/personal/kangcheng_liu_staff_main_ntu_edu_sg/EsmqJYbnUVFKoy8ysRUEbtcBft8ghHF8TAfWXl19Mu9sng?e=RbG1Ht).

More circumstances with diverse labeling percentage will be released. Please stay tuned.

## Testing

To test the trained model, please run the script:

        python test_models.py

# If you find our work helpful, please feel free to give a star to this repo!

