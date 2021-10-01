import torch.utils as utils
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from datasets.nocs.dataset_nocs_eval import Dataset
from lib.network_nocs import PatchNet
from lib.loss_nocs import Loss
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nocs')
parser.add_argument('--dataset_root', type=str, default='/data2/yifeis/pose/data_release/NOCS-REAL275-additional/',
                    help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--lr', default=0.00005, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.01, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.001, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max numbesr of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='model_nocs.pth', help='resume PoseNet model')#pose_model_7_0.1031530118677765.pth,pose_model_18_0.10158068922534585.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

proj_dir = os.getcwd()+'/'
torch.set_num_threads(8)

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'nocs':
        opt.num_objects = 6  # number of object classes in the dataset
        opt.num_points = 2000  # number of points on the input pointcloud
        opt.outf = '/home/lthpc/yifeis/pose/stablepose_new/trained_models/nocs'#proj_dir + 'trained_models/nocs/'  # folder to save trained models
        opt.log_dir = proj_dir + 'experiments/logs/nocs/'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    else:
        print('Unknown dataset')
        return

    estimator = PatchNet()
    estimator = estimator.cuda()


    total_params = sum(p.numel() for p in estimator.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    dataset = Dataset('train', opt.dataset_root, False, opt.num_points, 6, 5000, 1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, 6, 6000, 2)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    # opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\n'.format(
            len(dataset), len(test_dataset), 1))

    criterion = Loss(2000)
    st_time = time.time()
    estimator.eval()

    r_pred1_ls = []
    r_pred2_ls = []
    r_gt1_ls = []
    r_gt2_ls = []
    t_pred_ls = []
    t_gt_ls = []
    cate_ls = []
    t_error_ls = []
    t_error_ls2 = []
    iou_ls = []
    count = 0
    datapath_ls = []
    modelpath_ls = []
    fit_before = []
    fit_after = []
    r_refine_ls = []
    t_refine_ls = []
    # dic =  np.load('/home/lthpc/yifeis/pose/pose_est_tless_3d/tools/nocs/test_results2.npy',allow_pickle=True)
    with torch.no_grad():
        for j, data in enumerate(testdataloader, 0):
            points, model_points, target_rt, target_pt, idx, choose_patchs,datapath,model_path = data

            points, target_rt, target_pt, model_points = Variable(points).cuda(), \
                                                         Variable(target_rt).cuda(), \
                                                         Variable(target_pt).cuda(), \
                                                         Variable(model_points).cuda()

            pred_r, pred_t, pred_a = estimator(points, choose_patchs)
            loss, dis, norm_loss, patch_loss, r_pred, t_pred, pred = criterion(pred_r, pred_t, pred_a, target_rt,
                                                                            points, idx,
                                                                            target_pt, model_points,
                                                                            )
            sym_list = [1,2,4]
            obj_cate = idx.item()+1
            if obj_cate in sym_list:
                sym=1
            else:
                sym=0
            print('sym==', sym,'loss_all:', loss.item(), 'loss_c:', patch_loss.item(), 'loss_model:',
                  loss.item() - patch_loss.item())

            gt_rt = target_rt.detach().cpu().numpy().reshape(-1)
            r_gt1 = gt_rt[:9].reshape(3,3)
            r_gt2 = gt_rt[:9].reshape(3, 3).T

            r_pr1 = r_pred.contiguous().view(3,3)
            r_pr1 = r_pr1.detach().cpu().numpy()
            r_pr2 = r_pred.contiguous().transpose(2, 1).contiguous()
            r_pr2 = r_pr2.detach().cpu().numpy()

            t_gt = gt_rt[9:]
            t_pr = t_pred.detach().cpu().numpy().reshape(-1)

            r_pred1_ls.append(r_pr1)
            r_pred2_ls.append(r_pr2)
            r_gt1_ls.append(r_gt1)
            r_gt2_ls.append(r_gt2)
            t_pred_ls.append(t_pr)
            t_gt_ls.append(t_gt)
            cate_ls.append(obj_cate)

            ############## t_error and iou
            target = target_pt.detach().cpu().numpy().reshape(-1, 3)
            pred = pred.detach().cpu().numpy().reshape(-1,3)
            t_error = np.linalg.norm(t_pr - t_gt)
            t_error_ls.append(t_error)
            datapath_ls.append(datapath[0][0])
            modelpath_ls.append(model_path[0])
            # iou = knn(pred, target, 0.087)
            iou=0
            iou_ls.append(iou)
            if iou > 0.25:
                count += 1

            ########### ICP
            view_point = points.detach().cpu().numpy().reshape(-1, 3)
            model = model_points.view(-1, 3).detach().cpu().numpy()
            r_array = np.array(r_pr2).reshape(3, 3)
            t_array = np.array(t_pr).reshape(3, 1)
            attach = np.array([0, 0, 0, 1]).reshape(1, 4)
            trans_init = np.append(r_array, t_array, axis=1)
            trans_init = np.append(trans_init, attach, axis=0)

            transform,fit_af,fit_bf = icp(model,pred,view_point,trans_init)
            fit_after.append(fit_af)
            fit_before.append(fit_bf)

            r_pred = transform[:3, :3]
            t_pred = transform[:3, 3].reshape(3)
            pred_icp = np.add(model @ r_pred.T,t_pred)
            r_refine_ls.append(r_pred)
            t_refine_ls.append(t_pred)
            t_error2 = np.linalg.norm(t_pred - t_gt)
            t_error_ls2.append(t_error2)
            print('predicting frame ',j)


        mean_iou = np.mean(iou_ls)
        mean_t_error1 = np.mean(t_error_ls)
        mean_t_error2 = np.mean(t_error_ls2)
        print('fit before:',np.mean(fit_before))
        print('fit_after:',np.mean(fit_after))
        print('mean_t_error1:',mean_t_error1)
        print('mean_t_error2:', mean_t_error2)
        print(mean_iou)
        print(count)
        print(j)
        dic = {'r_pred1':r_pred1_ls,'r_pred2':r_pred2_ls,'r_gt1':r_gt1_ls,'r_gt2':r_gt2_ls,'t_gt':t_gt_ls,'t_pred':t_pred_ls,'r_refine':r_refine_ls,'t_refine':t_refine_ls,'obj_id':cate_ls}
        dic_visual = {'r_pred': r_pred2_ls,'r_gt': r_gt2_ls, 't_gt': t_gt_ls,'t_pred': t_pred_ls, 'r_refine':r_refine_ls,'t_refine':t_refine_ls,'obj_id': cate_ls, 'frame': datapath_ls,'model':modelpath_ls}
        np.save('test_results_visual.npy',dic_visual)
        np.save('test_results.npy', dic)


def displayPoint(data,target,view,title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams['axes.unicode_minus'] = False
    while len(data[0]) > 20000:
        print("too much point")
        exit()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r', marker='.')
    ax.scatter3D(target[:, 0], target[:, 1], target[:, 2], c='b', marker='.')
    ax.scatter3D(view[:, 0], view[:, 1], view[:, 2], c='g', marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()


def knn(pcd_a,pcd_b,thresh):
    in_count = 0
    for i in range(500):
            dis = min(np.linalg.norm(pcd_b-pcd_a[i].reshape(1,3).repeat(500,0),axis=1))
            if dis<thresh:
                in_count+=1
    frac = in_count/pcd_a.shape[0]
    return frac

synset_names = ['BG',
                'bottle',
                'bowl',
                'camera',
                'can',
                'laptop',
                'mug'
                ]

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    if RT_1 is None or RT_2 is None:
        return 10000,10000
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        return 10000,10000

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    if synset_names[class_id] in ['bottle', 'can', 'bowl']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] == 'mug' and handle_visibility==0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2)
    result = np.array([theta, shift])

    return result

def icp(model,pred,view_point,trans_init):
    model_max_x = np.max(model[:, 0]) - np.min(model[:, 0])
    model_max_y = np.max(model[:, 1]) - np.min(model[:, 1])
    model_max_z = np.max(model[:, 2]) - np.min(model[:, 2])
    model_d = max([model_max_x, model_max_y, model_max_z])
    mindis = 0.1 * model_d

    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(model)
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred)
    pcd_view = o3d.geometry.PointCloud()
    pcd_view.points = o3d.utility.Vector3dVector(view_point)

    evaluation = o3d.registration.evaluate_registration(pcd_model, pcd_view, mindis, trans_init)
    fitness = evaluation.fitness
    inlier_rmse = evaluation.inlier_rmse
    fit_bf = fitness

    reg_p2p = o3d.registration.registration_icp(pcd_model, pcd_view, mindis, trans_init,
                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                o3d.registration.ICPConvergenceCriteria(
                                                    max_iteration=3000))
    fit_af=reg_p2p.fitness
    transform = reg_p2p.transformation
    print(reg_p2p)

    return transform,fit_af,fit_bf


if __name__ == '__main__':
    main()

