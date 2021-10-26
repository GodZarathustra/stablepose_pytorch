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
  ** datasets/tless/dataset_test.py: the evaluation dataloader for tless dataset  
  ** datasets/tless/dataset_config/*.txt: training and testing splits for tless dataset

  ** datasets/linemod/dataset_lmo.py: the training dataloader for linemod dataset  
  ** datasets/linemod/dataset_lmo_test.py: the evaluation dataloader for linemod dataset  
  ** datasets/linemod/dataset_config/*.txt: training and testing splits for linemod dataset

  ** datasets/nocs/dataset_nocs.py: the training dataloader for nocs dataset  
  ** datasets/nocs/dataset_nocs_eval.py: the evaluation dataloader for nocs dataset  
  ** datasets/nocs/dataset_config/*.txt: training and testing splits for nocs dataset

To train StablePose on T-LESS dataset, run
```
python train_tless.py
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
The detection results lmo can be found at [this link](https://drive.google.com/file/d/1Tde_jPLxsi-KeYi0gI8qAC1-xmo7kFzL/view?usp=sharing).


The detection results tless can be found at [this link](https://drive.google.com/file/d/1cDHdfyGourdJoPJlHODyoWkTkYoVa7sm/view?usp=sharing).


The trained model for lmo can be found at [this link](https://drive.google.com/file/d/1r-RlnVrOseu9gmeG8WbptwsrWcnEkcDp/view?usp=sharing).


The trained model for tless can be found at [this link](https://drive.google.com/file/d/1jdVSwBJduUpd7hv2_jrN__ZNjpiGsakO/view?usp=sharing).

Note for simplicity and fair comparison, we combine all the categories in a single model for both instance-level and category-level in this repo, wich is different from the implementation in our paper and brings performance decline. 

For full training and testing dataset: [here](https://pan.baidu.com/s/1q6wM21l5IM2zs8KsmDRzIg) (baidu yunpan, password: cqqx).
