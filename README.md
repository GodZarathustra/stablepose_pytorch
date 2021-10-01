# StablePose
StablePose: Learning 6D Object Poses from Geometrically Stable Patches

CVPR 2021

Created by Yifei Shi, Junwen Huang, Xin Xu, Yifan Zhang and Kai Xu

This repository includes:  
* lib: the core Python library for networks and loss  
  ** lib/loss_*dataset.py: symmetrynet loss caculation for respective dataset  
  ** lib/network_*dataset.py: network architecture for the respective dataset  

* datasets: the dataloader and training/testing lists  
  ** datasets/tless/dataset.py: the training dataloader for tless dataset  
  ** datasets/tless/dataset_eval.py: the evaluation dataloader for tless dataset  
  ** datasets/tless/dataset_config/*.txt: training and testing splits for tless dataset

  ** datasets/shapenet/dataset.py: the training dataloader for shapnet dataset  
  ** datasets/shapenet/dataset_eval.py: the evaluation dataloader for shapnet dataset  
  ** datasets/shapenet/dataset_config/*.txt: training and testing splits for shapenet dataset 

  ** datasets/linemod/dataset.py: the training dataloader for linemod dataset  
  ** datasets/linemod/dataset_eval.py: the evaluation dataloader for linemod dataset  
  ** datasets/linemod/dataset_config/*.txt: training and testing splits for linemod dataset

  ** datasets/nocs/dataset.py: the training dataloader for nocs dataset  
  ** datasets/nocs/dataset_eval.py: the evaluation dataloader for nocs dataset  
  ** datasets/nocs/dataset_config/*.txt: training and testing splits for nocs dataset

To train StablePose on T-LESS dataset, run
```
python train_tless.py -tless
```
To train StablePose on NOCS-REAL275 dataset, run 
```
python train_nocs.py
```
To train StablePose on LMO dataset, run 
```
python train_lmo.py
```


To evaluate instace-level datasets: T-LESS and LMO, use the code here https://github.com/thodan/bop_toolkit.  

To test StablePose on T-LESS dataset, run 
```
python test_tless.py
```   
To test/evaluate StablePose on LMO dataset, run 
```
python test_lmo.py
```
The above scripts will create the required csv files in https://github.com/thodan/bop_toolkit.



To test/evaluate StablePose on NOCS-REAL275 dataset, run 
```
python test_nocs.py
```

## Pretrained model & data download
The pretrained models and data can be found at [here](https://pan.baidu.com/s/1cbRP2dgv4opkgmk8UUVbWQ) (baidu yunpan, password: qde4) and [here](https://pan.baidu.com/s/1q6wM21l5IM2zs8KsmDRzIg) (baidu yunpan, password: cqqx).
