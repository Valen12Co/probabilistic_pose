# probabilistic_pose

This is a secondary repo for the semester project `Effect of probabilistic pose estimator on downstream tasks` conducted during Fall 2024. Please look at `https://github.com/vita-epfl/downstream_performance_comparison/` for the framework implementation. In this repo, we build on the work of TIC-TAC [1] for the Human pose. The code is based on the code from the following Git `https://github.com/vita-epfl/TIC-TAC/tree/main/HumanPose`.
The aim was to extend the code to work for the MS_COCO dataset. Furthermore, we added a model that predicts if the joint is present in the image (visible or occluded counts as present). This enables us to change the covariance matrix accordingly and enable the full model to handle images where not necessarily all keypoints are present. We still added a small constraint to consider an image: it must have a sufficient bounding box size and at least one visible keypoint.

## Code structure

Tha main code structure is the following:
```
├── cached
│   ├── Stacked_HG_ValidationImageNames.txt
├── code
│   ├── utils
│   │   ├── pose.py
│   │   ├── tic.py
│   ├── models
│   │   ├── auxiliary
│   │   ├── joint_prediction
│   │   ├── stacked_hourglass
│   │   ├── vit_pose
├── main.py
├── dataloader.py
├── dataloader_CPU.py
├── load_CPU.py
├── loss.py
├── config.py
├── configuration.yml
```

## How to run the code?
The `dataloader.py` and `dataloader_CPU.py` are the data loader for the various datasets. If you want to just load the first time the data and save it, you could run  `python load_CPU.py`. If you wish to directly use the models you can simply run `python main.py`. Be careful to clearly understand the details of `configuration.yml`.

## Dataset

The Coco dataset has been imported in the RCP clusters. You can find it under the `datasets/MS_COCO_rwx`. It has the following structure:
```
datasets/MS_COCO_rwx
├── annotations
├── cached
│   ├── coco_full_train_not_all_joints.npy
│   ├── coco_full_validation_not_all_joints.npy
├── images
```

cached and annotations are the not treated data directely taken from the MS_COCO dataset website: `https://cocodataset.org/#download`. There are many files under cached, but the two main ones are the one used for the last training made with this code. It contains all the dataset where at least one human is in the image. Following this we apply all_joints or bbox_size etc...

## Configuration.yml file clarification

`configuration.yml` contains five main variables that are important:
* model: you can choose either Hourglass or ViTPose
* dataset/load: you choose which dataset you want to use. The code was tested with mpii and coco.
* precached: for each dataset: decided if you use the pre-saved model of the data, you will pass the load_xxx functions. The first run it should be set to False.
* all_joints: For coco only. If true you run with the coco dataset where all joints are visible. If False, you run with at least one visible ground truth and a minimal bbox size. Note that this is valid for the case where you set preloading to True. If you set it to False you may need to add the functions.
* preloading: For coco only. If True directly bypass the load_coco and create_coco files. You will en up with a none filtered datset, where your preferences set with all_joints are then applied.
  
## References
[1] SHUKLA, Megh, SALZMANN, Mathieu, et ALAHI, Alexandre. TIC-TAC: A Framework for Improved Covariance Estimation in Deep Heteroscedastic Regression. In: Proceedings of the 41st International Conference on Machine Learning (ICML) 2024. 2024.
