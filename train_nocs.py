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
from datasets.nocs.dataset_nocs import Dataset
from lib.network_nocs import PatchNet
from lib.loss_nocs import Loss

from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nocs')
parser.add_argument('--dataset_root', type=str, default='/data2/yifeis/pose/data_release/NOCS-REAL275-additional/',
                    help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of data loading workers')
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
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')#pose_model_2_193909.25539978288.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

proj_dir = os.getcwd()
torch.set_num_threads(32)

def main():
    # opt.manualSeed = random.randint(1, 10000)
    # random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'nocs':
        opt.num_objects = 6  # number of object classes in the dataset
        opt.num_points = 2000  # number of points on the input pointcloud
        opt.outf = proj_dir + '/trained_models/nocs/'  # folder to save trained models
        if os.path.exists(opt.outf)==False:
            os.makedirs(opt.outf)
        opt.log_dir = proj_dir + '/experiments/logs/nocs/'  # folder to save logs
        if os.path.exists(opt.log_dir)==False:
            os.makedirs(opt.log_dir)
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    else:
        print('Unknown dataset')
        return

    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    estimator = PatchNet()
    # estimator = torch.nn.DataParallel(estimator)
    estimator = estimator.cuda()
    # estimator = torch.nn.parallel.DistributedDataParallel(estimator,find_unused_parameters=True)

    total_params = sum(p.numel() for p in estimator.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=0.01)

    dataset = Dataset('train', opt.dataset_root, False, opt.num_points, 6, 5000, 1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, 6, 1000, 2)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    # opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\n'.format(
            len(dataset), len(test_dataset), 1))

    criterion = Loss(2000)
    # criterion_refine = Loss_refine(opt.num_points_mesh)

    best_test = np.Inf
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        train_patch_avg = 0.0
        train_norm_avg = 0.0

        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, model_points, target_rt, target_pt,idx,choose_patchs = data

                points, target_rt, target_pt, model_points = Variable(points).cuda(), \
                                                 Variable(target_rt).cuda(), \
                                                 Variable(target_pt).cuda(),\
                                                 Variable(model_points).cuda()

                pred_r, pred_t, pred_a = estimator(points, choose_patchs)
                loss, dis, norm_loss, patch_loss, r_pred, t_pred, _ = criterion(pred_r, pred_t, pred_a,target_rt,
                                                                                points, idx,
                                                                                target_pt, model_points,
                                                                                )
                loss.backward()

                torch.cuda.empty_cache()

                train_dis_avg += dis.item()
                train_patch_avg += patch_loss.item()
                train_norm_avg += norm_loss.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info(
                        'Train time {0} Epoch {1} Batch {2} Frame {3}  Avg_dis:{4} Avg_norm:{5} Avg_patch:{6}'.format(
                            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                            int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size,
                                                                            train_norm_avg / opt.batch_size,
                                                                            train_patch_avg / opt.batch_size,
                                                                            ))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0
                    train_norm_avg = 0
                    train_patch_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_patch = 0.0
        test_norm = 0.0
        test_count = 0
        estimator.eval()
        # refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, model_points, target_rt, target_pt, idx, choose_patchs = data

            points, target_rt, target_pt, model_points = Variable(points).cuda(), \
                                                         Variable(target_rt).cuda(), \
                                                         Variable(target_pt).cuda(), \
                                                         Variable(model_points).cuda()

            pred_r, pred_t, pred_a = estimator(points, choose_patchs)
            loss, dis, norm_loss, patch_loss, r_pred, t_pred, _ = criterion(pred_r, pred_t, pred_a, target_rt,
                                                                            points, idx,
                                                                            target_pt, model_points,
                                                                            )

            test_dis += dis.item()
            test_norm += norm_loss.item()
            test_patch += patch_loss.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2} norm_loss:{3} patch_loss:{4}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis, norm_loss,
                patch_loss))

            test_count += 1

        test_dis = test_dis / test_count
        test_norm = test_norm / test_count
        test_patch = test_patch / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2} avg norm: {3} avg tless: {4}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis, test_norm, test_patch))
        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

            # opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print(
                '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
                    len(dataset), len(test_dataset), opt.num_points_mesh))

            criterion = Loss(opt.num_points_mesh)
            # criterion_refine = Loss_refine(opt.num_points_mesh)

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
