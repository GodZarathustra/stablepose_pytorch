import torch.utils as utils
import argparse
import os
import random
import time
import numpy as np
import torch
import sys
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.dataset_lmo_test import PoseDataset as PoseDataset_lmo
from lib.network_lmo import PatchNet, PoseRefineNet
from lib.loss_tless import Loss
import pandas as pd
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lmo')
parser.add_argument('--dataset_root', type=str, default='/data2/yifeis/pose/stablepose_data_release/lmo',
                    help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=64, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.005, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.02, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='model_lmo.pth', help='resume PoseNet model')##pose_model_23_0.2027222095310724.pth  pose_model_10_0.19870004712375647.pth  pose_model_33_0.19804978409920596.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
proj_dir = os.getcwd()+'/'
cls_id_ls = [1,5,6,8,9,10,11,12]

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.dataset == 'lmo':
        opt.num_objects = 8
        opt.num_points = 2000
        opt.outf = proj_dir + 'trained_models/lmo/'
        opt.log_dir = proj_dir + 'experiments/logs/lmo/'
        opt.repeat_epoch = 2
    else:
        print('Unknown dataset')
        return

    estimator = PatchNet(num_obj=opt.num_objects)
    # estimator = nn.DataParallel(estimator)
    estimator = estimator.cuda()
    total_params = sum(p.numel() for p in estimator.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # print(estimator)
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    refiner.cuda()
    # utils.print_network(estimator)
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = False  # True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


    dataset = PoseDataset_lmo('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
                                      opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                             pin_memory=True)

    test_dataset = PoseDataset_lmo('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers,
                                                 pin_memory=True)

    opt.sym_list = dataset.get_sym_list()
    nosym_list = dataset.get_nosym_list()
    rot_list = dataset.get_rot_list()
    ref_list = dataset.get_ref_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list, rot_list, ref_list, nosym_list)

    best_test = np.Inf

    st_time = time.time()

    test_dis = 0.0
    test_count = 0
    estimator.eval()
    refiner.eval()

    result_list = []
    scene_id_ls = []
    im_id_ls = []
    obj_id_ls = []
    r_ls = []
    score_ls = []
    t_ls = []
    time_ls = []
    dis_ls = []
    occ_ls = []
    fit_before = []
    rmse_before = []
    fit_after = []
    rmse_after = []
    with torch.no_grad():
        for j, data in enumerate(testdataloader, 0):
            # points, choose, img, target, idx, choose_patchs = data
            points, choose, img, target_rt, target_trans, idx, \
            choose_patchs, target_pt, model_points, normals, model_info, model_axis, scene_id, im_id = data
            # if idx[0].item() in nosym_list:
            #     continue
            points, choose, img, target_rt, target_trans, idx, \
            target_pt, model_points, normals, model_axis = Variable(points).cuda(), \
                                                           Variable(choose).cuda(), \
                                                           Variable(img).cuda(), \
                                                           Variable(target_rt).cuda(), \
                                                           Variable(target_trans).cuda(), \
                                                           Variable(idx).cuda(), \
                                                           Variable(target_pt).cuda(), \
                                                           Variable(model_points).cuda(), \
                                                           Variable(normals).cuda(), \
                                                           Variable(model_axis).cuda()

            pred_r, pred_t, choose_r = estimator(img, points, choose, choose_patchs, idx)

            loss,dis, r_pred, t_pred, pred = criterion(pred_r, pred_t, choose_r, target_rt,
                                                                            target_trans, idx, points,
                                                                            target_pt, model_points,
                                                                            model_info)

            obj_id = cls_id_ls[idx]

            if idx[0].item() not in ref_list:
                scene_id = scene_id.numpy()[0]
                im_id = im_id.numpy()[0]

                r_pred = r_pred.detach().cpu().numpy().T.reshape(9).tolist()
                r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                t_pred = t_pred.view(3).detach().cpu().numpy().reshape(3) * 1000
                t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')
                score = 1
                # occ = occ.numpy()[0]
                dis = dis.detach().cpu().numpy()
                pred = pred.view(-1, 3).detach().cpu().numpy()
                view_point = points.detach().cpu().numpy().reshape(-1, 3)
                target = target_pt.view(-1, 3).detach().cpu().numpy()
                model = model_points.view(-1, 3).detach().cpu().numpy()
                r_array = np.array(r_pred).reshape(3, 3)
                t_array = np.array(t_pred).reshape(3, 1) / 1000
                attach = np.array([0, 0, 0, 1]).reshape(1, 4)
                trans_init = np.append(r_array, t_array, axis=1)
                trans_init = np.append(trans_init, attach, axis=0)

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
                fit_before.append(fitness)
                score = fitness
                rmse_before.append(inlier_rmse)
                # print('dis=', dis, 'fitness=', fitness, 'inlier_rmse=', inlier_rmse)

                reg_p2p = o3d.registration.registration_icp(pcd_model, pcd_view, mindis, trans_init,
                                                            o3d.registration.TransformationEstimationPointToPoint(),
                                                            o3d.registration.ICPConvergenceCriteria(max_iteration=3000))
                fit_after.append(reg_p2p.fitness)
                score = reg_p2p.fitness
                rmse_after.append(reg_p2p.inlier_rmse)
                print(reg_p2p)
                print('scene_id:', scene_id, ' im_id:', im_id, ' obj_id:', obj_id, 'score:', score, 'dis:',
                      dis.item())
                transform = reg_p2p.transformation

                r_pred = transform[:3, :3].reshape(9).tolist()
                r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                t_pred = transform[:3, 3].reshape(3) * 1000
                t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')

                scene_id_ls.append(scene_id)
                im_id_ls.append(im_id)
                obj_id_ls.append(obj_id)
                r_ls.append(r_pred_s)
                t_ls.append(t_pred_s)
                score_ls.append(score)
                time_ls.append(-1)
                dis_ls.append(dis)
            else:
                scene_id = scene_id.numpy()[0]
                im_id = im_id.numpy()[0]
                # choose_r = choose_r.detach().cpu().numpy().reshape(-1)
                r_pred = r_pred.detach().cpu().numpy()
                t_pred = t_pred.view(3).detach().cpu().numpy().reshape(3) * 1000
                r_pred_ = np.transpose(r_pred, [0, 2, 1])
                pred_ = pred.view(3, -1, 3).detach().cpu().numpy()
                dis = dis.detach().cpu().numpy()
                fit_ls = []
                for r_id in range(3):
                    r_pred = r_pred_[r_id, :, :]
                    r_pred = r_pred.reshape(9).tolist()
                    r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                    t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')
                    score = 1
                    pred = pred_[r_id, :, :]
                    view_point = points.detach().cpu().numpy().reshape(-1, 3)
                    model = model_points.view(-1, 3).detach().cpu().numpy()
                    r_array = np.array(r_pred).reshape(3, 3)
                    t_array = np.array(t_pred).reshape(3, 1) / 1000
                    attach = np.array([0, 0, 0, 1]).reshape(1, 4)
                    trans_init = np.append(r_array, t_array, axis=1)
                    trans_init = np.append(trans_init, attach, axis=0)

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
                    fit_before.append(fitness)
                    score = fitness
                    rmse_before.append(inlier_rmse)
                    # print('dis=', dis, 'fitness=', fitness, 'inlier_rmse=', inlier_rmse)

                    reg_p2p = o3d.registration.registration_icp(pcd_model, pcd_view, mindis, trans_init,
                                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                                o3d.registration.ICPConvergenceCriteria(
                                                                    max_iteration=3000))
                    fit_after.append(reg_p2p.fitness)
                    score = reg_p2p.fitness
                    rmse_after.append(reg_p2p.inlier_rmse)
                    print(reg_p2p)
                    print('scene_id:', scene_id, ' im_id:', im_id, ' obj_id:', obj_id, 'score:', score, 'dis:',
                          dis.item())
                    transform = reg_p2p.transformation

                    r_pred = transform[:3, :3].reshape(9).tolist()
                    r_pred_s = str(r_pred)[1:-1].replace(',', ' ')
                    t_pred = transform[:3, 3].reshape(3) * 1000
                    t_pred_s = str(t_pred.tolist())[1:-1].replace(',', ' ')
                    fit_ls.append(reg_p2p.fitness)

                    scene_id_ls.append(scene_id)
                    im_id_ls.append(im_id)
                    obj_id_ls.append(obj_id)
                    r_ls.append(r_pred_s)
                    t_ls.append(t_pred_s)
                    score_ls.append(score)
                    time_ls.append(-1)
                    dis_ls.append(dis)

        dataframe = pd.DataFrame({'scene_id': scene_id_ls, 'im_id': im_id_ls, 'obj_id': obj_id_ls, 'score': score_ls,
                                  'R': r_ls, 't': t_ls, 'time': time_ls})
        dataframe.to_csv("./lmo-test.csv", index=False, sep=',')
        print('mean_fitness_before=', np.mean(fit_before))
        print('mean_rmse_before=', np.mean(rmse_before))
        print('mean_fitness_after=', np.mean(fit_after))
        print('mean_rmse_after=', np.mean(rmse_after))
        print('mean_dis=', np.mean(dis_ls))


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


if __name__ == '__main__':
    main()
