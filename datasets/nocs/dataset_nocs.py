import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import random
from cv2 import cv2
import _pickle as cPickle
import open3d as o3d
import numpy.ma as ma
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
import heapq


class Dataset(data.Dataset):
    def __init__(self, mode, root, add_noise, num_pt, num_cates, count, cate_id):
        self.root = root
        self.add_noise = add_noise
        self.mode = mode
        self.num_pt = num_pt
        self.num_cates = num_cates
        self.back_root = '{0}/train2017/'.format(self.root)

        self.cate_id = cate_id

        self.obj_list = {}
        self.obj_name_list = {}

        if self.mode == 'train':
            for tmp_cate_id in range(1, self.num_cates + 1):
                self.obj_name_list[tmp_cate_id] = os.listdir('{0}/data_list/train/{1}/'.format(self.root, tmp_cate_id))
                self.obj_list[tmp_cate_id] = {}

                for item in self.obj_name_list[tmp_cate_id]:
                    print(tmp_cate_id, item)
                    self.obj_list[tmp_cate_id][item] = []

                    input_file = open('{0}/data_list/train/{1}/{2}/list.txt'.format(self.root, tmp_cate_id, item), 'r')
                    while 1:
                        input_line = input_file.readline()
                        if not input_line:
                            break
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        self.obj_list[tmp_cate_id][item].append('{0}/data_ls/{1}'.format(self.root, input_line))
                    input_file.close()

        self.real_obj_list = {}
        self.real_obj_name_list = {}

        for tmp_cate_id in range(1, self.num_cates + 1):
            self.real_obj_name_list[tmp_cate_id] = os.listdir(
                '{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, tmp_cate_id))
            self.real_obj_list[tmp_cate_id] = {}

            for item in self.real_obj_name_list[tmp_cate_id]:
                print(tmp_cate_id, item)
                self.real_obj_list[tmp_cate_id][item] = []

                input_file = open(
                    '{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, tmp_cate_id, item), 'r')

                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    self.real_obj_list[tmp_cate_id][item].append('{0}/data_ls/{1}'.format(self.root, input_line))
                input_file.close()


        self.mesh = []
        proj_dir = os.getcwd()
        input_file = open(proj_dir+'/datasets/nocs/sphere.xyz', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            self.mesh.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        self.mesh = np.array(self.mesh) * 0.6

        self.cam_cx_1 = 322.52500
        self.cam_cy_1 = 244.11084
        self.cam_fx_1 = 591.01250
        self.cam_fy_1 = 590.16775

        self.cam_cx_2 = 319.5
        self.cam_cy_2 = 239.5
        self.cam_fx_2 = 577.5
        self.cam_fy_2 = 577.5

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
        self.length = count

    def divide_scale(self, scale, pts):
        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts

    def get_anchor_box(self, ori_bbox):
        bbox = ori_bbox
        limit = np.array(search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1

        small_range = [1, 3]

        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale

    def change_to_scale(self, scale, cloud_fr, cloud_to=None):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        if cloud_to:
            cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to

    def enlarge_bbox(self, target):

        limit = np.array(search_fit(target))
        longest = max(limit[1] - limit[0], limit[3] - limit[2], limit[5] - limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1] - limit[0])
        scale2 = longest / (limit[3] - limit[2])
        scale3 = longest / (limit[5] - limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def get_pose(self, choose_frame, choose_obj):
        has_pose = []
        pose = {}
        if self.mode == "train":
            input_file = open('{0}_pose.txt'.format(choose_frame.replace("data_ls/", "data_pose/")), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1:
                    idx = int(input_line[0])
                    has_pose.append(idx)
                    pose[idx] = []
                    for i in range(4):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        pose[idx].append(
                            [float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
            input_file.close()
        if self.mode == "val":
            with open('{0}/data_ls/gts/real_test/results_real_test_{1}_{2}.pkl'.format(self.root,
                                                                                       choose_frame.split("/")[-2],
                                                                                       choose_frame.split("/")[-1]),
                      'rb') as f:
                nocs_data = cPickle.load(f)
            for idx in range(nocs_data['gt_RTs'].shape[0]):
                idx = idx + 1
                pose[idx] = nocs_data['gt_RTs'][idx - 1]
                pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1
                pose[idx] = z_180_RT @ pose[idx]
                pose[idx][:3, 3] = pose[idx][:3, 3] * 1000

        input_file = open('{0}_meta.txt'.format(choose_frame), 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            if input_line[-1] == choose_obj:
                ans = pose[int(input_line[0])]
                ans_idx = int(input_line[0])
                break
        input_file.close()

        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t, ans_idx

    def backproject(self, depth, intrinsics, instance_mask):

        intrinsics_inv = np.linalg.inv(intrinsics)
        image_shape = depth.shape
        width = image_shape[1]
        height = image_shape[0]

        x = np.arange(width)
        y = np.arange(height)

        non_zero_mask = (depth > 0)
        final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

        idxs = np.where(final_instance_mask)
        grid = np.array([idxs[1], idxs[0]])

        length = grid.shape[1]
        ones = np.ones([1, length])
        uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

        xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
        xyz = np.transpose(xyz)  # [num_pixel, 3]

        z = depth[idxs[0], idxs[1]]

        pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]

        return pts, idxs

    def get_frame(self, choose_frame, choose_obj, syn_or_real):
        syn_or_real = False

        img = np.array(Image.open('{0}_color.png'.format(choose_frame)))
        depth = np.array(self.load_depth('{0}_depth.png'.format(choose_frame)))

        # get RT from the pose.txt
        target_r, target_t, idx = self.get_pose(choose_frame, choose_obj)

        # get cloud from rgbd input
        if syn_or_real:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        cam_scale = 1.0
        intrinsics = np.array([[cam_fx, 0, cam_cx],
                               [0., cam_fy, cam_cy],
                               [0., 0., 1.]])

        # import scipy.misc
        mask_im = cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 2]
        mask_im = np.array(mask_im)
        patch_file = '{0}{1}{2}_{3}_segmentation.png'.format(choose_frame[:-4], 'process/', choose_frame[-4:], str(idx))
        patch_label = cv2.imread(patch_file)[:, :, 2]
        choose_file = '{0}{1}{2}_{3}_choose.list'.format(choose_frame[:-4], 'process/', choose_frame[-4:], str(idx))
        choose_ls = []
        stable_ls = []
        try:
            with open(choose_file) as f:
                data = f.readlines()
            if len(data) > 1:
                for ids in data:
                    choose_id = ids[:-1].split(',')[:-1]
                    stable = float(ids[:-1].split(',')[-1])
                    choose_ls.append([int(x) for x in choose_id])
                    stable_ls.append(stable)
            else:
                if data[0] != '0':
                    choose_id = data[0].split(',')[:-1]
                    stable = float(data[0].split(',')[-1])
                    choose_ls.append([int(x) for x in choose_id])
                    stable_ls.append(stable)
                else:
                    stable_ls.append(0)
        except:
            print('-choose list file not exist')
            stable_ls.append(0)
            choose_ls = []
            data = ['0']

        tmp_mask = (mask_im == idx)
        # patch_mask = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_seg = ma.getmaskarray(ma.masked_not_equal(patch_label, 0))
        mask_patch = mask_seg * mask_depth
        mask_label = tmp_mask * mask_depth
        mask = mask_label * mask_patch

        pts_ori, idxs = self.backproject(depth, intrinsics, mask_label)

        translation = target_t/1000
        rotation = target_r
        # pts_ = pts_ori - translation
        # pts = pts_ @ rotation
        # pts = pts / 1000
        # target_t = target_t / 1000

        choose = mask_label.flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            if len(choose) == 0:
                print(0)
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')


        patch_masked = patch_label.flatten()[choose][:, np.newaxis].astype(np.float32)
        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = depth_masked / 1000
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

        # get the mesh of normalized model
        mesh_path = '{0}/obj_models/real_{1}/{2}.obj'.format(self.root, self.mode, choose_obj)
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # mesh = np.asarray(mesh.vertices)
        verts, faces, _ = load_obj(mesh_path)
        model_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        mesh = sample_points_from_meshes(model_mesh, 2000)
        target = np.dot(mesh[0], target_r.T) + translation

        patches = patch_masked.astype(int)
        num_patch = np.max(patches)
        num_list = []
        patch_list = patches.reshape(-1).tolist()
        for n in range(1, num_patch + 1):  # ordered num of point in each tless(from patch_1 to patch_n)
            num = str(patch_list).count(str(n))
            num_list.append(num)

        num_list_new = []
        patch_id_list_all = []
        for m in num_list:  # select patchs that num of points > 100
            if m > 100:
                num_list_new.append(m)
                patch_id_list_all.append(num_list.index(m) + 1)

        choose_patch = []
        all_list = [i for i in range(0, 2000)]
        if data[0] != '0':
            patch_id_list = choose_ls[stable_ls.index(max(stable_ls))]
            for k in patch_id_list:
                patch_idx = []
                for m in range(cloud.shape[0]):
                    if patch_masked[m] == k:
                        patch_idx.append(m)
                if len(patch_idx) >= 128:
                    choose_patch.append(np.array(patch_idx))
                else:
                    choose_patch.append(np.array(all_list))
        else:
            choose_patch.append(np.array(all_list))
        if not choose_patch:
            choose_patch.append(np.array(all_list))

        mesh = mesh.detach().data.numpy().reshape(2000,3)
        model_axis = np.array([0.0,1.0,0.0])
        target_rt = np.append(target_r.T, translation).reshape(1, 12)
        return [cloud,  # observed point cloud  #pts,  # clouds trans to mesh coordinate
                mesh,  # mesh from the normalized model
                target_rt,
                target,
                choose_patch]# mesh trans to camera(clouds) coordinate


    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr
        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale

    def __getitem__(self, index):
        # syn_or_real = (random.randint(1, 20) < 15)
        syn_or_real = False  # only use the real dataset
        if self.mode == 'val':
            syn_or_real = False

        while 1:
            try:
        # cate_id_ = self.cate_id
                cate_id = random.choice([1, 2, 3, 4, 5, 6])
                # cate_id = 5
                choose_obj = random.sample(self.real_obj_name_list[cate_id], 1)[0]
                choose_frame = random.sample(self.real_obj_list[cate_id][choose_obj], 1)  # only choose one image randomly

                cloud, mesh, target_rt, target,choose_patch = self.get_frame(choose_frame[0], choose_obj,
                                                                                           syn_or_real)

                break

            except:
                continue

        class_gt = np.array([cate_id - 1])

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(mesh.astype(np.float32)), \
               torch.from_numpy(target_rt.astype(np.float32)), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.LongTensor(class_gt), \
               choose_patch


    def __len__(self):
        return self.length


border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def get_2dbbox(cloud, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale):
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt

    if ((rmax - rmin) in border_list) and ((cmax - cmin) in border_list):
        return rmin, rmax, cmin, cmax
    else:
        return 0


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]

