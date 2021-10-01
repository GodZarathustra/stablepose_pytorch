from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn
import random
import math
import torch.backends.cudnn as cudnn
import pytorch3d
from torch.autograd import Variable
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.transforms import (
    quaternion_apply,
    quaternion_invert,
    quaternion_to_matrix,
    matrix_to_quaternion,
    Rotate,
    transform3d
)
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def loss_calculation(pred_r, pred_t, pred_a, target_rt, points, idx,target, model_points):
    rot_obj_idx = [0, 1, 3, 5]
    nosym_obj_idx = [2, 4, 5]
    num_point_mesh = model_points.shape[1]
    bs = 1
    num_p,_ = pred_t.size()
    target_rt = target_rt.reshape(1, 12)
    target_r_ = target_rt[0, :9].reshape(3, 3)
    target_t_ = target_rt[0, 9:].reshape(-1, 3)
    # target_trans = target_s.contiguous().view(-1,4,4)
    target_r = target_r_.reshape(1, 3, 3).repeat(bs * num_p, 1, 1).contiguous().view(bs * num_p, 3, 3)
    target_t = target_t_.reshape(1, 1, 3).repeat(bs * num_p, 1, 1).contiguous().view(bs * num_p, 1, 3)
    num_pt = points.shape[1]

    model_axis = torch.FloatTensor([0,1,0]).cuda().view(-1,3)

    mean_points = torch.mean(points, dim=1).view(bs, 1, 3)
    dis_c = torch.norm((mean_points + pred_t - target_t), dim=2).view(-1)
    if idx in rot_obj_idx:
        target_axis = torch.mm(model_axis, target_r.view(3, 3))
        target_axis = F.normalize(target_axis, p=2, dim=1).view(bs, -1)
        base = quaternion_to_matrix(pred_a.view(bs,4))
        base = base.contiguous().transpose(2, 1).contiguous()
        pred_axis = torch.mm(model_axis, base.view(3, 3))
        pred_axis = F.normalize(pred_axis, p=2, dim=1).view(bs, -1)
        cos_loss = 1-torch.nn.functional.cosine_similarity(pred_axis, target_axis)
        pred = torch.add(torch.bmm(model_points,base),pred_t+mean_points)
        dis_model = torch.mean(torch.mean(torch.norm((pred - target), dim=2), dim=1))
        loss = 0.8*cos_loss+0.2*dis_model
    elif idx in nosym_obj_idx:
        model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p,
                                                                                               num_point_mesh,
                                                                                               3)
        target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
        pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
        base = quaternion_to_matrix(pred_r.contiguous().view(bs, 4))
        base = base.contiguous().transpose(2, 1).contiguous()
        pred = torch.add(torch.bmm(model_points, base), mean_points + pred_t)
        dis_model = torch.mean(torch.mean(torch.norm((pred - target), dim=2), dim=1))
        loss = dis_model

    loss = loss+dis_c
    r_pred = base
    t_pred = pred_t + mean_points

    return loss, loss, loss, dis_c, r_pred, t_pred, pred #, new_points.detach(), new_target.detach()

def label_trans(input):
    if input.shape[0] == 3:
        label = torch.tensor([1, 1, 1])
    if input.shape[0] == 2:
        if input.equal(torch.tensor([0, 1])) or input.equal(torch.tensor([1, 0])):
            label = torch.tensor([1, 1, 0])

        if input.equal(torch.tensor([0, 2])) or input.equal(torch.tensor([2, 0])):
            label = torch.tensor([1, 0, 1])

        if input.equal(torch.tensor([1, 2])) or input.equal(torch.tensor([2, 1])):
            label = torch.tensor([0, 1, 1])
    if input.shape[0] == 1:
        if input.equal(torch.tensor([0])):
            label = torch.tensor([1, 0, 0])

        if input.equal(torch.tensor([1])):
            label = torch.tensor([0, 1, 0])

        if input.equal(torch.tensor([2])):
            label = torch.tensor([0, 0, 1])
    return label

class Loss(_Loss):

    def __init__(self, num_points_mesh):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        # self.rot_list = rot_list
        # self.ref_list = ref_list
        # self.nosym_list = nosym_list
        # self.sym_list = sym_list

    def forward(self, pred_r, pred_t,pred_a, target_rt,points,idx, target,model_pt, ):

        return loss_calculation(pred_r, pred_t, pred_a, target_rt, points,idx,target,model_pt)

