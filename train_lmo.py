import torch.utils as utils
import argparse
import os
import random
import time
import numpy as np
import torch

import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.dataset_lmo import PoseDataset as PoseDataset_linemod
from lib.network_lmo import PatchNet, PoseRefineNet
from lib.loss_tless import Loss
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lmo')
parser.add_argument('--dataset_root', type=str, default='/data2/yifeis/pose/stablepose_data_release/lmo/',
                    help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.01, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.001, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max numbesr of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')#pose_model_2_193909.25539978288.pth
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

    if opt.dataset == 'lmo':
        opt.num_objects = 8
        opt.num_points = 2000
        opt.outf = proj_dir +'trained_models/lmo/'
        if os.path.exists(opt.outf)==False:
            os.makedirs(opt.outf)
        opt.log_dir =  proj_dir +'experiments/logs/lmo/'
        if os.path.exists(opt.log_dir)==False:
            os.makedirs(opt.log_dir)
        opt.repeat_epoch = 2
    else:
        print('Unknown dataset')
        return
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    estimator = PatchNet(num_obj=opt.num_objects)
    # estimator = torch.nn.DataParallel(estimator)
    estimator = estimator.cuda()
    # estimator = torch.nn.parallel.DistributedDataParallel(estimator,find_unused_parameters=True)

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
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=0.01)


    dataset = PoseDataset_linemod('train', opt.num_points, False, opt.dataset_root, opt.noise_trans,
                                      opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers,
                                             pin_memory=True)

    test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                                 pin_memory=True)

    opt.sym_list = dataset.get_sym_list()
    nosym_list = dataset.get_nosym_list()
    rot_list = dataset.get_rot_list()
    ref_list = dataset.get_ref_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list,rot_list,ref_list,nosym_list)

    best_test = np.Inf
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        estimator.train()
        optimizer.zero_grad()
        for rep in range(opt.repeat_epoch):
            torch.cuda.empty_cache()
            mean_epoch_loss = 0
            mean_epoch_diss = 0
            train_loss_avg = 0
            train_dis_avg = 0.0
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target_rt,target_trans, idx, \
                choose_patchs,target_pt,model_points,normals,model_info,model_axis,_,_ = data

                points, choose, img, target_rt, target_trans,idx,\
                target_pt, model_points,normals,model_axis = Variable(points).cuda(), \
                                                 Variable(choose).cuda(), \
                                                 Variable(img).cuda(), \
                                                 Variable(target_rt).cuda(), \
                                                 Variable(target_trans).cuda(),\
                                                 Variable(idx).cuda(), \
                                                 Variable(target_pt).cuda(),\
                                                 Variable(model_points).cuda(),\
                                                 Variable(normals).cuda(),\
                                                 Variable(model_axis).cuda()

                pred_r, pred_t, pred_choose = estimator(img, points, choose, choose_patchs, idx)

                loss, dis, r_pred, t_pred, _ = criterion(pred_r, pred_t, pred_choose, target_rt,
                                                            target_trans, idx, points,
                                                            target_pt,model_points,
                                                            model_info)

                loss.backward()

                mean_epoch_loss += loss.item()

                train_loss_avg += loss.item()
                train_dis_avg += dis.item()

                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info(
                        'Train time {0} Epoch {1} Batch {2} Frame {3}  idx:{4} Avg_loss:{5} Avg_dis:{6}'.format(
                            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                            int(train_count / opt.batch_size), train_count, idx, train_loss_avg / opt.batch_size,
                                                                                 train_dis_avg / opt.batch_size))

                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0
                    train_loss_avg = 0
                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

            mean_epoch_diss = mean_epoch_diss / train_count
            mean_epoch_loss = mean_epoch_loss / train_count
            print('mean epoch loss:', mean_epoch_loss, 'mean epoch dis:', mean_epoch_diss)

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_loss = 0.0
        test_count = 0
        estimator.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target_rt, target_trans, idx, \
            choose_patchs, target_pt, model_points, normals, model_info, model_axis, _, _ = data

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

            pred_r, pred_t, pred_choose = estimator(img, points, choose, choose_patchs, idx)

            loss, dis, r_pred, t_pred, _ = criterion(pred_r, pred_t, pred_choose, target_rt,
                                                                            target_trans, idx, points,
                                                                            target_pt, model_points,
                                                                            model_info)

            logger.info('Test time {0} Test Frame No.{1} idx:{2} loss:{3} dis:{4}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, idx, loss.item(),
                dis.item()))

            test_count += 1
            test_loss += loss.item()
            test_dis += dis.item()

        test_dis = test_dis / test_count
        test_loss = test_loss / test_count

        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2} avg loss: {3}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis, test_loss))

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
