import os
import numpy as np
import torch
from PIL import Image

import torchvision
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

from visualize_maskrcnn_predictions import *
import matplotlib.pyplot as plt
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        
        self.imgs = []
        self.sem_masks = []
        self.ins_masks = []

        self.scenes = list(sorted(os.listdir(root)))
        for scene in self.scenes:
            if os.path.exists(os.path.join(root, scene, 'mask_visib_pred')):
                shutil.rmtree(os.path.join(root, scene, 'mask_visib_pred'))
            
            if os.path.exists(os.path.join(root, scene, 'segmentation_pred')):
                shutil.rmtree(os.path.join(root, scene, 'segmentation_pred'))
            
            if os.path.exists(os.path.join(root, scene, 'segmentation_pred_vis')):
                shutil.rmtree(os.path.join(root, scene, 'segmentation_pred_vis'))
            
            os.mkdir(os.path.join(root, scene, 'mask_visib_pred'))
            os.mkdir(os.path.join(root, scene, 'segmentation_pred'))
            os.mkdir(os.path.join(root, scene, 'segmentation_pred_vis'))

            scene_pred_path = os.path.join(root, scene, 'mask_visib_pred', 'scene_pred.txt')
            if os.path.exists(scene_pred_path) and os.path.getsize(scene_pred_path) > 0:
                os.remove(scene_pred_path)

        for scene in self.scenes:
            scene_dir = os.path.join(root, scene)
            scene_imgs = list(sorted(os.listdir(os.path.join(scene_dir,"rgb"))))
            scene_sem_masks = [0]*len(scene_imgs)
            scene_ins_masks = [0]*len(scene_imgs)
            for i in range(len(scene_imgs)):
                name = scene_imgs[i]
                scene_imgs[i] = os.path.join(scene_dir,"rgb",name)
                scene_sem_masks[i] = os.path.join(scene_dir,"mask_visib_processed",name.split('.')[0]+'_sem.png')
                scene_ins_masks[i] = os.path.join(scene_dir,"mask_visib_processed",name.split('.')[0]+'_ins.png')
                if not os.path.exists(scene_sem_masks[i]) or not os.path.exists(scene_ins_masks[i]):
                    print('warning', scene_sem_masks[i], scene_ins_masks[i])
            self.imgs += scene_imgs
            self.sem_masks += scene_sem_masks
            self.ins_masks += scene_ins_masks
            
        print('self.scenes',len(self.scenes))
        print('self.imgs',len(self.imgs))
        print('self.sem_masks',len(self.sem_masks))
        print('self.ins_masks',len(self.ins_masks))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        sem_mask_path = self.sem_masks[idx]
        ins_mask_path = self.ins_masks[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ins_mask_path)
        sem_mask = Image.open(sem_mask_path)

        mask = np.array(mask)
        sem_mask = np.array(sem_mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        sem_label = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if (xmax - xmin) <= 2:
                xmax = xmax + 2
            if (ymax - ymin) <= 2:
                ymax = ymax + 2
            boxes.append([xmin, ymin, xmax, ymax])
            sem_label.append(sem_mask[pos[0][0],pos[1][0]])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        for i in range(len(labels)):
            labels[i] = int(sem_label[i])
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        L = self.sem_masks[idx].split('/')
        mask_pred_folder = mask_pred_path = os.path.join(self.root, L[-3], 'mask_visib_pred')
        if not os.path.exists(mask_pred_folder):
            os.mkdir(mask_pred_folder)
        mask_pred_path = os.path.join(mask_pred_folder, L[-1].split('_')[0])
        scene_pred_path = os.path.join(mask_pred_folder, 'scene_pred.txt')
        
        target["mask_pred_path"] = mask_pred_path
        target["scene_pred_path"] = scene_pred_path
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_dir = '/data2/yifeis/pose/stablepose_maskrcnn_data_release/T-LESS/test_primesense'
    dataset_test = PennFudanDataset(data_dir, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=3, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    num_classes = 100
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load('./log_tless/network.pth'))
    model.to(device)

    print('evaluate...')
    evaluate(model, data_loader_test, device=device)

    print('output...')
    model.cpu()
    model.eval()
    for i, data in enumerate(data_loader_test, 0):
        img, target = data
        for j in range(len(img)):
            result, output, top_pred = predict(img[j], model, target[j])
    
    print("That's it!")
    
if __name__ == "__main__":
    main()
