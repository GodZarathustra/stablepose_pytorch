import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision

def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions["labels"]
    boxes = predictions['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions["masks"].ge(0.5).mul(255).byte().numpy()
    labels = predictions["labels"]

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite


def overlay_keypoints(image, predictions):
    kps = predictions["keypoints"]
    scores = predictions["keypoints_scores"]
    kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
    for region in kps:
        image = vis_keypoints(image, region.transpose((1, 0)))
    return image

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


CATEGORIES = """BACKGROUND
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
""".split("\n")

class PersonKeypoints(object):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)


def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions['boxes']

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image
    
    
def predict(img, model, target):
    img = img.permute(1, 2, 0)
    cv_img = np.array(img)[:, :, :]
    img_tensor = torchvision.transforms.functional.to_tensor(cv_img)
    with torch.no_grad():
        #output = model([img_tensor.cuda()])
        output = model([img_tensor])
    
    top_predictions = select_top_predictions(output[0], 0.6)
    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
    
    # for visualization
    result = cv_img.copy()
    result = overlay_boxes(result, top_predictions)
    if 'masks' in top_predictions:
        result = overlay_mask(result, top_predictions)
    #if 'keypoints' in top_predictions:
    #    result = overlay_keypoints(result, top_predictions)
    result = overlay_class_names(result, top_predictions)

    f = open(target['scene_pred_path'],'a')
    for i in range(len(output[0]['labels'])):
        if output[0]['scores'][i] < 0.01:
            continue
        mask_c = output[0]['masks'][i,0,:,:]*255
        mask_c[mask_c>50] = 255
        mask_c[mask_c<=50] = 0
        box_c = output[0]['boxes'][i].numpy()
        label_c = output[0]['labels'][i].numpy()
        mask_dir = target['mask_pred_path']+'_'+str(i).zfill(6)+'.png'
        print('write to',mask_dir)
        cv2.imwrite(mask_dir, mask_c.numpy().astype(np.uint8))
        img_name = mask_dir.split('/')[-1]
        f.write('%s %f %f %f %f %d\n'%(img_name,box_c[0],box_c[1],box_c[2],box_c[3],label_c))
    f.close()

    return result, output, top_predictions
