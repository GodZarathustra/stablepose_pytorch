# Code, data and pretrained models for object detection and segmentation (MaskRCNN)

To run the code, you should first download the data from [this link](https://drive.google.com/file/d/1-zOHOR7PC6Oe5SgmsA1Ss6MFA9selfME/view?usp=sharing) and the pretrained models from [this link](https://drive.google.com/drive/folders/1vqeRZBF4EKrFkRAS7mBbyV04CBEek9sE?usp=sharing).

To perform object detection and segmentation on T-LESS dataset, run
```
python maskrcnn_test_tless.py
```
To perform object detection and segmentation on LMO dataset, run
```
python maskrcnn_test_lmo.py
```

The code above will quantitively evaluate the results and generate two folders that contain the masks of the detected objects (segmentation_pred and mask_visib_pred).

