from torch.nn.modules.loss import _Loss
import torch
import numpy as np
import math
from torch.autograd import Variable
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from scipy.optimize import linear_sum_assignment

def loss_calculation(pred_r, pred_t, choose, target_rt,target_trans, idx, points,
                      num_point_mesh,rot_list,ref_list,
                     nosym_list,target,model_points,model_info):

    bs, num_p, _ = pred_t.size()
    target_rt = target_rt.reshape(1, 12)
    target_r_ = target_rt[0, :9].reshape(3, 3)
    target_t_ = target_rt[0, 9:].reshape(-1, 3)
    target_r = target_r_.reshape(1, 3, 3).repeat(bs * num_p, 1, 1).contiguous().view(bs * num_p, 3, 3)
    target_t = target_t_.reshape(1, 1, 3).repeat(bs * num_p, 1, 1).contiguous().view(bs * num_p, 1, 3)
    num_pt = points.shape[1]

    mean_points = torch.mean(points, dim=1).view(bs, 1, 3)
    dis_c = torch.norm((mean_points + pred_t - target_t), dim=2).view(-1)
    if idx[0].item() in rot_list:
        pred_axis = (pred_r/torch.norm(pred_r, dim=1)).contiguous().view(bs,-1)
        target_quater = matrix_to_quaternion(target_r)
        rot_ang = torch.acos(target_quater[0,0])
        target_axis = (target_quater[0,1:]/torch.sin(rot_ang)).contiguous().view(bs,-1)
        target_axis = target_axis/torch.norm(target_axis,dim=1)
        cos_theta = 1-torch.nn.functional.cosine_similarity(pred_axis, target_axis)
        loss = cos_theta
        theta = torch.tensor(math.pi / 2).cuda()
        cos_theta = torch.cos(theta).view(1,1)
        base = quaternion_to_matrix(torch.cat([cos_theta,pred_axis*torch.sin(theta)],dim=1))
        base = base.contiguous().transpose(2, 1).contiguous()
        pred = torch.add(torch.bmm(model_points,base),pred_t+mean_points)
    else:
        model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p,
                                                                                               num_point_mesh,
                                                                                               3)
        target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
        pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
        points = points.contiguous().view(bs * num_pt, 1, 3)

        if idx[0].item() in nosym_list:
            base = quaternion_to_matrix(pred_r.contiguous().view(bs,4))
            base = base.contiguous().transpose(2, 1).contiguous()
            pred = torch.add(torch.bmm(model_points, base), mean_points + pred_t)
            dis_model = torch.mean(torch.mean(torch.norm((pred - target), dim=2), dim=1))
            loss = dis_model

        if idx[0].item() in ref_list:
            pred_r = pred_r.contiguous().view(3,4)
            base = quaternion_to_matrix(pred_r).contiguous()
            base = base.contiguous().transpose(2, 1).contiguous()

            sym_trans_ls = model_info["symmetries_discrete"]
            cost_matrix = torch.zeros(3, len(sym_trans_ls))
            for i in range(len(sym_trans_ls)):
                for j in range(base.shape[0]):
                    pred = torch.add(torch.bmm(model_points, base[j,:].view(bs,3,3)), mean_points + pred_t)
                    sym_trans = sym_trans_ls[i]
                    sym_trans = Variable(torch.Tensor(sym_trans).view(1, 4, 4)).cuda()
                    ones = Variable(torch.from_numpy(np.ones((bs, num_point_mesh, 1)).astype(np.float32))).cuda()
                    model_points_ = torch.cat((model_points, ones), 2).permute(0, 2, 1)
                    target_pt_ = torch.bmm(target_trans, torch.bmm(sym_trans, model_points_)).permute(0, 2, 1)[:, :, :3]
                    sym_dis = torch.mean(torch.mean(torch.norm((pred - target_pt_), dim=2), dim=1))
                    cost_matrix[j, i] = sym_dis
            row_id_, col_id_ = linear_sum_assignment(cost_matrix.detach().numpy())
            if len(sym_trans_ls) > 1:
                corr = np.array([row_id_, col_id_]).T
                ordered_id = corr[corr[:, 1].argsort()]
                row_id = ordered_id[:, 0]
                col_id = ordered_id[:, 1]
            else:
                row_id = row_id_
                col_id = col_id_
            target_id = label_trans(torch.tensor(row_id)).cuda().float()
            id_loss = torch.nn.BCELoss()
            num_loss = id_loss(choose[0], target_id)
            dis_sum = 0
            for k in range(row_id.shape[0]):
                dis_sum += cost_matrix[row_id_[k], col_id_[k]]
            model_loss = dis_sum/row_id.shape[0]

            loss = 0.5*num_loss + 0.5*model_loss.cuda()

    loss = loss+dis_c

    r_pred = base
    t_pred = pred_t + mean_points
    if idx[0].item() in ref_list:
        pred = torch.add(torch.bmm(model_points.repeat(3, 1, 1), r_pred), t_pred)

    return loss, dis_c, r_pred, t_pred, pred

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

    def __init__(self, num_points_mesh, sym_list, rot_list, ref_list, nosym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.rot_list = rot_list
        self.ref_list = ref_list
        self.nosym_list = nosym_list
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target_rt, target_trans,idx, points,
             target, model_pt,model_info):

        return loss_calculation(pred_r, pred_t, pred_c, target_rt,target_trans, idx, points,
                                self.num_pt_mesh,
                               self.rot_list,self.ref_list,
                                self.nosym_list,target,model_pt,model_info)


