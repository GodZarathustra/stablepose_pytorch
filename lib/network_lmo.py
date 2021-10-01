import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from lib.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch.nn.functional as F
# from lib.pspnet_patchfusion import PSPNet
torch.backends.cudnn.enabled = False

class PatchFeat(nn.Module):
    def __init__(self):
        super(PatchFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1_x = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2_x = nn.BatchNorm1d(128)

        self.conv5 = torch.nn.Conv1d(128, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        in_channel = 128
        self.sa1 = PointNetSetAbstractionMsg(128, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[64, 64, 64], [64, 64, 128], [64, 96, 128]])  #(1,320,512)
        self.sa2 = PointNetSetAbstractionMsg(64, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])  #(1,640,128)
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)  #(1,1024,1)
        self.ap_patch = torch.nn.AdaptiveMaxPool1d(1)
        # self.num_points = num_points

    def forward(self,x, num_points):
        pointfeat_0 = self.bn1_x(F.relu(self.conv1(x)))
        pointfeat_1 = self.bn2_x(F.relu(self.conv2(pointfeat_0)))

        xyz = x
        l1_xyz, l1_points = self.sa1(xyz, pointfeat_1)  # (1,320,512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (1,640,128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (1,1024,1)

        ap_x = l3_points.view(-1, 1024, 1).repeat(1, 1, num_points)
        # return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 256 + 512=896
        return ap_x

class PatchNet(nn.Module):
    def __init__(self,num_obj):
        super(PatchNet, self).__init__()
        # self.cnn = ModifiedResnet()
        self.feat = PatchFeat()
        self.global_pool = nn.MaxPool1d(2000)
        self.rot_obj_idx = []
        self.ref_obj_idx = [6-1,7-1]
        self.nosym_oobj_idx = [0,1,2,3,4,7]
        self.out_dim = len(self.rot_obj_idx)*3+len(self.ref_obj_idx)*12+len(self.nosym_oobj_idx)*4

        self.rl_r = nn.Sequential(
            torch.nn.Conv1d(2048 * 2, 2048, 1),
            # nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            # nn.Dropout(0.5),
            torch.nn.Conv1d(2048, 1024, 1),
            # nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            # nn.Dropout(0.4),
            torch.nn.Conv1d(1024, 512, 1),
            # nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )
        self.rl_t = nn.Sequential(
            torch.nn.Conv1d(2048*2, 2048, 1),
            # nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            # nn.Dropout(0.5),
            torch.nn.Conv1d(2048, 1024, 1),
            # nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            # nn.Dropout(0.4),
            torch.nn.Conv1d(1024, 512, 1),
            # nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        self.fc_r = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            # nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.out_dim)
        )  # quaternion

        self.fc_choose = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
            torch.nn.Sigmoid()
        )  # quaternion

        self.fc_t = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_obj * 3)
        )  # translation

        self.rl_maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.rl_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.num_obj = num_obj

    def forward(self, img, x, choose, patch_ls, obj):
        # out_img, im_emb = self.cnn(img)
        bs = x.shape[0]
        num_points = x.shape[1]

        x = x.transpose(2, 1).contiguous()

        global_feat = self.feat(x, num_points)
        ap_global_feat = self.global_pool(global_feat)

        r_list = []
        t_list = []
        feat_list = []
        for i in range(len(patch_ls)):
            patch_dix = patch_ls[i][0]
            num_patch_points = patch_dix.shape[0]
            patch_x = x[:, :, patch_ls[i][0]]
            # choose_emb = patch_dix.view(1,1,-1).repeat(1, di, 1).cuda()
            # emb_patch = torch.gather(emb_global, 2, choose_emb).contiguous()
            patch_feat = self.feat(patch_x, patch_dix.shape[0]) #(982)
            global_feat_ = ap_global_feat.repeat(1, 1, num_patch_points) #(982)
            # global_emb = im_emb.repeat(1, 1, num_patch_points) #(512)
            fused_feat = torch.cat([global_feat_, patch_feat], 1)  # (896*2+512=2304)(bs,c,num_pt)
            avg_feat = self.rl_avgpool(fused_feat).view(bs, 1, 1, -1)
            # feat_list.append(fused_feat)

            # rx = self.conv_r(fused_feat).contiguous().view(bs, 1, 1, -1) #(1,1,1,512)
            # tx = self.conv_t(fused_feat).contiguous().view(bs, 1, 1, -1) #(1,1,1,512)
            r_list.append(avg_feat)
            t_list.append(avg_feat)

        r_map = torch.cat(r_list, 2)
        t_map = torch.cat(t_list, 2)

        c_feature = avg_feat.shape[3]

        r_map_x = r_map.contiguous().view(bs, 1, -1, c_feature).repeat(1, len(r_list), 1, 1)
        r_map_y = r_map.contiguous().view(bs, -1, 1, c_feature).repeat(1, 1, len(r_list), 1)

        t_map_x = t_map.contiguous().view(bs, 1, -1, c_feature).repeat(1, len(r_list), 1, 1)
        t_map_y = t_map.contiguous().view(bs, -1, 1, c_feature).repeat(1, 1, len(r_list), 1)

        rl_map_r = torch.cat([r_map_x, r_map_y], 3) #(bs,n_patch,n_patch,512*2)
        rl_map_t = torch.cat([t_map_x, t_map_y], 3) #(bs,n_patch,n_patch,512*2)

        rx = rl_map_r.contiguous().view(bs, c_feature*2, -1)
        tx = rl_map_t.contiguous().view(bs, c_feature*2, -1)

        if rx.shape[2] == 1:
            rx = rx.repeat(1, 1, 2)
            tx = tx.repeat(1, 1, 2)

        rx = self.rl_r(rx).contiguous().view(bs, -1)
        tx = self.rl_t(tx).contiguous().view(bs, -1)

        choose_r = self.fc_choose(rx).contiguous().view(bs, 3)
        rx = self.fc_r(rx).view(bs, self.out_dim)
        tx = self.fc_t(tx).view(bs, self.num_obj, 3, -1)

        b = 0
        st_id, end_id = get_idx(obj[b])
        out_rx = rx[0,st_id:end_id].contiguous().view(bs, -1)
        out_tx = torch.index_select(tx[b], 0, obj[b]).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, choose_r


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)  # translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx


rot_obj_idx = []
ref_obj_idx = [6-1,7-1]
nosym_oobj_idx = [0,1,2,3,4,7]

def get_idx(obj_id):
    st_idx = 0
    for i in range(obj_id):
        if i in nosym_oobj_idx:
            st_idx += 4
        elif i in rot_obj_idx:
            st_idx += 3
        else:
            st_idx += 4 * 3
    if obj_id in nosym_oobj_idx:
        end_idx = st_idx + 4
    elif obj_id in rot_obj_idx:
        end_idx = st_idx + 3
    else:
        end_idx = st_idx + 4 * 3
    return st_idx, end_idx