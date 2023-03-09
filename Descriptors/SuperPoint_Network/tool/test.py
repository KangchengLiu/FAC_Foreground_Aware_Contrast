"""
NOTE: 
"""

import os
#import sys
import time
import random
import numpy as np
import logging
import argparse
import importlib

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler

from util import dataset, transform, config
import yaml
from util.util import AverageMeter, intersectionAndUnionGPU

import torchnet as tnt
import metrics
from metrics import *
from util.util import get_components, perfect_prediction, relax_edge_binary
from util.util import perfect_prediction_base0
from util.util import prediction2ply_seg

from util.util import write_spg
from graphs import compute_sp_graph_yj

from provider import *

from util.S3DIS_dataset import create_s3dis_datasets, my_collate
from util.VKITTI_dataset import create_vkitti_datasets, my_collate_vkitti
from util.SCANNET_dataset import create_scannet_datasets, my_collate_scannet
from util.SCANNET_dataset_v1 import create_scannet_datasets_v1, my_collate_scannet_v1

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Superpoint Generation')
    parser.add_argument('--config', type=str, default=None, required=True, help='config file')
    parser.add_argument('--model_path', type=str, default=None, required=True, help='save path')
    parser.add_argument('--save_folder', type=str, default=None, required=True, help='save_folder path')
    parser.add_argument('--epoch', type=int, default=None, required=True, help='corresponding to the train_epoch_xx.pth')

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["model_path"] = args.model_path
    cfg["save_folder"] = args.save_folder
    cfg["epoch"] = args.epoch
    
    print("#"*20)
    print("Parameters:")
    for ky in cfg.keys():
        print('key: {} -> {}'.format(ky, cfg[ky]))
    print("#"*20)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def worker_init_fn(worker_id):
    random.seed(args["manual_seed"] + worker_id)

def init():
    global args, logger
    args = get_parser()

    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args["test_gpu"])
    logger.info(args)


def main():
    init()
    
    MODEL = importlib.import_module(args["arch"])  # import network module
    logger.info("load {}.py success!".format(args["arch"]))

    model = MODEL.Network(cfg=args)
    
    total = sum([param.nelement() for param in model.parameters()])
    logger.info("Number of params: %.2fM" % (total/1e6))

    
    if args["sync_bn"]:
        from util.util import convert_to_syncbn
        convert_to_syncbn(model) #, patch_replication_callback(model)

    criterion = nn.CrossEntropyLoss(ignore_index=args["ignore_label"]).cuda()
    if args['re_xyz_loss'] == 'mse':
        criterion_re_xyz = nn.MSELoss().cuda()
    else:
        print('re_xyz_loss type error')
        exit()
    
    if args['re_label_loss'] == 'cel':
        criterion_re_label = nn.CrossEntropyLoss(ignore_index=args["ignore_label"]).cuda()
    elif args['re_label_loss'] == 'mse':
        criterion_re_label = nn.MSELoss().cuda()
    else:
        print('re_label_loss type error')
        exit()

    if args['re_sp_loss'] == 'cel':
        criterion_re_sp = nn.CrossEntropyLoss(ignore_index=args["ignore_label"]).cuda()
    elif args['re_sp_loss'] == 'mse':
        criterion_re_sp = nn.MSELoss().cuda()
    else:
        print('re_label_loss type error')
        exit()

    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args["classes"]))
    logger.info(model)
    model = model.cuda()
    
    model_path = os.path.join(args["model_path"], "train_epoch_{}.pth".format(args["epoch"]))
    if os.path.isfile(model_path):
        logger.info("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    

    if args["data_name"] == 's3dis':
        train_data, test_data = create_s3dis_datasets(args, logger)
        my_collate_s = my_collate
    elif args['data_name'] == 'vkitti':
        train_data, test_data = create_vkitti_datasets(args, logger)
        my_collate_s = my_collate_vkitti
    elif args['data_name'] == 'scannet':
        train_data, test_data = create_scannet_datasets(args, logger)
        my_collate_s = my_collate_scannet
    elif args['data_name'] == 'scannet_v1':
        train_data, test_data = create_scannet_datasets_v1(args, logger)
        my_collate_s = my_collate_scannet_v1
    else:
        print("data_name error")
        exit()
   
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=args["test_batch_size"],
                                               shuffle=False,
                                               num_workers=8,
                                               collate_fn=my_collate_s,
                                               pin_memory=True,
                                               drop_last=False)

    test(test_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, args['epoch'])

def label2one_hot(labels, C=13):
    b, n = labels.shape
    labels = torch.unsqueeze(labels, dim=1)
    one_hot = torch.zeros(b, C, n, dtype=torch.long).cuda()         # create black
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
    return target.type(torch.float32)

def binarys(points, dep):
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    if dep == 0:
        ax = 1
    else:
        ax = 0
    left, right = coord_min[ax], coord_max[ax]
    half_n = int(points.shape[0] / 2.0)
    pidx1, pidx2 = None, None
    while (left + 1e-8 < right):
        mid = (left + right) / 2.0 
        tmp1 = np.where((points[:, ax] <= mid))[0]
        if tmp1.size <= int(half_n*1.1) and tmp1.size >= int(half_n*0.9):
            tmp2 = np.where((points[:, ax] > mid))[0]
            pidx1 = tmp1
            pidx2 = tmp2
            break
        elif tmp1.size < int(half_n*0.9):
            left = mid
        else:
            right = mid
    assert (points.shape[0] == pidx1.size + pidx2.size)
    return pidx1, pidx2

def dfs(points, points_index, th, dep):
    ret = []
    if points.shape[0] <= th:
        ret.append(points_index)
        return ret

    t1, t2 = binarys(points, dep)
    p1, p2 = points[t1, :], points[t2, :]
    i1, i2 = points_index[t1], points_index[t2]
    r1 = dfs(p1, i1, th, dep+1)
    for val in r1:
        ret.append(val)
    r2 = dfs(p2, i2, th, dep+1)
    for val in r2:
        ret.append(val)
    return ret

def split_data(points, nei, gt, th=100000):
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    
    points_idx = np.arange(points.shape[0])
    pidx_list = dfs(points, points_idx, th, 0)
    pts, neis, gts, indexs = [], [], [], []
    for val in pidx_list:
        pts.append(points[val, :])
        neis.append(nei[val, :, :])
        gts.append(gt[:, val])
        indexs.append(val)
    return pts, neis, gts, indexs
    
def test(test_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    loss_re_xyz_meter = AverageMeter()
    loss_re_label_meter = AverageMeter()
    loss_re_sp_meter = AverageMeter()
    loss_semantic_meter = AverageMeter()

    loss_meter = AverageMeter()
    
    BR_meter = tnt.meter.AverageValueMeter()
    BP_meter = tnt.meter.AverageValueMeter()
    confusion_matrix = metrics.ConfusionMatrix(args['classes'])

    model.eval()
    end = time.time()
    max_iter = 1 * len(test_loader)
    cnt_room, cnt_sp, cnt_sp_act = 0, 0, 0
    cnt_sp_std = 0.
    for i, (fname, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz) in enumerate(test_loader): 
        logger.info('name: {}'.format(fname[0]))
        # fname: file name
        # edg_source: 1
        # edg_target: 1
        # is_transition: 1
        # labels: 1 x n x 14            torch.IntTensor
        # objects: n                    torch.LongTensor
        # clouds: (1*n) x 6 x 20        torch.FloatTensor
        # clouds_global: 1 x n x 7      torch.FloatTensor
        # xyz: 1 x n x 3                torch.FloatTensor
        # gt: 1 x n                     torch.LongTensor
        data_time.update(time.time() - end)
        if args['data_name'] in ['scannet_v1', 'semanticposs']:
            gt = labels[:, :, :].argmax(axis=2)
        else:
            gt = labels[:, :, 1:].argmax(axis=2)
        input = torch.cat((xyz, clouds_global), dim=2)
        logger.info('xyz: {}'.format(xyz.numpy().shape))    # 1 x n x 3
        # th = 500000
        th = 450000
        if input.size(1) >= th:
            inps, neis, gts, indexs = split_data(np.squeeze(input.numpy(), axis=0), clouds.numpy(), gt.numpy(), th)
            n = 0
            mov_n = 0   # number of points move
            mov_m = 0   # number of clusters move
            all_c_idx = np.array([], dtype=np.int32)
            all_c2p_idx = np.zeros((1, input.size(1), args['near_clusters2point']), dtype=np.int32)   # 1 x n x nc2p
            all_c2p_idx_base = np.zeros((1, input.size(1), args['near_clusters2point']), dtype=np.int32)   # 1 x n x nc2p
            all_output = np.zeros((1, input.size(1), args['classes']), dtype=np.float32)                  # 1 x n x nclass
            all_rec_xyz = np.zeros((1, input.size(1), 3), dtype=np.float32)                               # 1 x n x 3
            all_rec_label = np.zeros((1, input.size(1), args['classes']), dtype=np.float32)               # 1 x n x nclass
            all_fea_dist = np.zeros((1, input.size(1), args['near_clusters2point']), dtype=np.float32)    # 1 x n x nc2p
            for j in range(len(inps)):
                n = n + inps[j].shape[0]
                s_input = torch.from_numpy(inps[j]).cuda(non_blocking=True).unsqueeze(0)
                s_gt = torch.from_numpy(gts[j]).cuda(non_blocking=True)
                s_clouds = torch.from_numpy(neis[j]).cuda(non_blocking=True)
                s_onehot_label = label2one_hot(s_gt, args['classes']) 
                logger.info('s_input: {} {}'.format(s_input.size(), s_input.type()))
                logger.info('s_gt: {} {}'.format(s_gt.size(), s_gt.type()))
                logger.info('s_clouds: {} {}'.format(s_clouds.size(), s_clouds.type()))
                logger.info('s_onehot_label: {} {}'.format(s_onehot_label.size(), s_onehot_label.type()))
                with torch.no_grad():
                    _, c_idx, c2p_idx, c2p_idx_base, output, rec_xyz, rec_label, fea_dist, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot = model(s_input, s_clouds.contiguous(), s_onehot_label, s_gt.unsqueeze(-1))
                
                c_idx = c_idx.cpu().numpy()                 # 1 x m'         val: 0,1,2,...,n'-1
                c_idx += mov_n
                
                all_c_idx = np.concatenate((all_c_idx, c_idx), axis=1) if all_c_idx.size else c_idx
                logger.info('c_idx: {}'.format(c_idx.shape, np.max(c_idx)))
                logger.info('mov_n: {}'.format(mov_n))
            
                c2p_idx = c2p_idx.cpu().numpy()             # 1 x n' x nc2p  val: 0,1,2,...,n'-1
                c2p_idx += mov_n
                all_c2p_idx[0, indexs[j], :] = c2p_idx
                logger.info('c2p_idx: {} {}'.format(c2p_idx.shape, np.max(c2p_idx)))
                logger.info('mov_n: {}'.format(mov_n))
            
                c2p_idx_base = c2p_idx_base.cpu().numpy()   # 1 x n' x nc2p  val: 0,1,2,...,m'-1
                c2p_idx_base += mov_m
                all_c2p_idx_base[0, indexs[j], :] = c2p_idx_base
                logger.info('c2p_idx_base: {}'.format(c2p_idx_base.shape))
                logger.info('mov_m: {}'.format(mov_m))

                output = output.detach().cpu().numpy()      # 1 x nclass x n
                logger.info('output: {}'.format(output.shape))
                all_output[0, indexs[j], :] = output.transpose((0, 2, 1))
            
                rec_xyz = rec_xyz.detach().cpu().numpy()    # 1 x 3 x n
                logger.info('rec_xyz: {}'.format(rec_xyz.shape))
                all_rec_xyz[0, indexs[j], :] = rec_xyz.transpose((0, 2, 1))
            
                rec_label = rec_label.detach().cpu().numpy()    # 1 x nclass x n
                logger.info('rec_label: {}'.format(rec_label.shape))
                all_rec_label[0, indexs[j], :] = rec_label.transpose((0, 2, 1))


                fea_dist = fea_dist.detach().cpu().numpy()  # 1 x n' x nc2p
                all_fea_dist[0, indexs[j], :] = fea_dist
                logger.info('fea_dist: {}'.format(fea_dist.shape))
            
            
                mov_n += inps[j].shape[0]   # number of points move
                mov_m += c_idx.shape[1]   # number of clusters move
            logger.info('n: {}'.format(n))
            logger.info('all_c_idx: {}'.format(all_c_idx.shape))                    # 1 x m
            logger.info('all_c2p_idx: {}'.format(all_c2p_idx.shape))                # 1 x n x nc2p
            logger.info('all_c2p_idx_base: {}'.format(all_c2p_idx_base.shape))      # 1 x n x nc2p
            all_output = all_output.transpose((0, 2, 1))            # 1 x n x nclass -> 1 x nclass x n
            logger.info('all_output: {}'.format(all_output.shape))
            all_rec_xyz = all_rec_xyz.transpose((0, 2, 1))          # 1 x n x 3 -> 1 x 3 x n
            logger.info('all_rec_xyz: {}'.format(all_rec_xyz.shape))                # 1 x 3 x n
            all_rec_label = all_rec_label.transpose((0, 2, 1))      # 1 x n x nclass -> 1 x nclass x n
            logger.info('all_rec_label: {}'.format(all_rec_label.shape))            # 1 x nclass x n
            logger.info('all_fea_dist: {}'.format(all_fea_dist.shape))              # 1 x n x nc2p
            
            gt = gt.cuda(non_blocking=True)
            onehot_label = label2one_hot(gt, args['classes'])

        else:
            input = input.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            onehot_label = label2one_hot(gt, args['classes'])
            clouds = clouds.cuda(non_blocking=True)
            with torch.no_grad():
                spout, c_idx, c2p_idx, c2p_idx_base, output, rec_xyz, rec_label, fea_dist, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot = model(input, clouds.contiguous(), onehot_label, gt.unsqueeze(-1))
            # ---------------- superpoint realted ------------------
            # spout:        b x n x nc2p
            # c_idx:        b x m           in 0,1,2,...,n
            # c2p_idx:      b x n x nc2p    in 0,1,2,...,n
            # c2p_idx_base: b x n x nc2p    in 0,1,2,...,m-1
            # ---------------- semantic related --------------------
            # output:       b x classes x n
            all_c_idx = c_idx.cpu().numpy()
            all_c2p_idx = c2p_idx.cpu().numpy()
            all_c2p_idx_base = c2p_idx_base.cpu().numpy()
            all_output = output.detach().cpu().numpy()
            all_rec_xyz = rec_xyz.detach().cpu().numpy()
            all_rec_label = rec_label.detach().cpu().numpy()
            all_fea_dist = fea_dist.detach().cpu().numpy()
            
            input = input.cpu()
            gt = gt.cpu()

        cnt_sp += all_c_idx.shape[1]
        cnt_room += 1
      
        spout = all_fea_dist
        if gt.shape[-1] == 1:
            gt = gt[:, 0]  # for cls
        re_xyz_loss = args['w_re_xyz_loss'] * criterion_re_xyz(torch.from_numpy(all_rec_xyz), input[:, :, :3].transpose(1,2).contiguous())
        if args['re_label_loss'] == 'cel':
            re_label_loss = args['w_re_label_loss'] * criterion_re_label(torch.from_numpy(all_rec_label), gt.cpu())
        elif args['re_label_loss'] == 'mse':
            re_label_loss = args['w_re_label_loss'] * criterion_re_label(torch.from_numpy(all_rec_label), onehot_label.cpu())

        if args['re_sp_loss'] == 'cel':
            re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab)
        elif args['re_sp_loss'] == 'mse':
            re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab_onehot)

        loss = re_xyz_loss + re_label_loss + re_sp_loss
        if args['use_semantic_loss']:
            semantic_loss = criterion(torch.from_numpy(all_output), gt)
            loss = loss + args['w_semantic_loss'] * semantic_loss

        for bid in range(args['test_batch_size']):
            txyz = input[bid, :, :3].cpu().numpy()

            tpred = all_output[bid, :, :].transpose((0, 1)) # nclass x n -> n x nclass
            tpred = np.argmax(tpred, axis=1)
            tgt = gt[bid, :].cpu().numpy()
            trgb = input[bid, :, 5:8].cpu().numpy()
            tedg_source = edg_source[bid]
            tedg_target = edg_target[bid]
            tis_transition = is_transition[bid].numpy()
            spout_ = spout[bid, :, :]    # 1 x n x nc2p -> n x nc2p
            c2p_ = all_c2p_idx[bid, :, :] # 1 x n x nc2p -> n x nc2p

            init_center = all_c_idx[bid, :]     # 1 x m -> m
            pt_center_index = all_c2p_idx_base[bid, :, :]   # 1 x n x nc2p -> n x nc2p
            # pred_components, pred_in_component = get_components(init_center, pt_center_index, spout_, getmax=True, trick=False, logger=logger)
            pred_components, pred_in_component = get_components(init_center, pt_center_index, spout_, getmax=True, trick=True)
            pred_components = [x[0] for x in pred_components]
            cnt_sp_act += len(pred_components)

            pred_transition = pred_in_component[tedg_source] != pred_in_component[tedg_target]
            if args['data_name'] in ['scannet_v1', 'semanticposs']:
                full_pred = perfect_prediction_base0(pred_components, pred_in_component, labels[bid, :, :].numpy())
            else:
                full_pred = perfect_prediction(pred_components, pred_in_component, labels[bid, :, :].numpy())

            if args['data_name'] in ['scannet_v1', 'semanticposs']:
                confusion_matrix.count_predicted_batch(labels[bid, :, :].numpy(), full_pred)
            else:
                confusion_matrix.count_predicted_batch(labels[bid, :, 1:].numpy(), full_pred)

            if np.sum(tis_transition) > 0:
                BR_meter.add((tis_transition.sum()) * compute_boundary_recall(tis_transition, 
                            relax_edge_binary(pred_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance'])),
                            n=tis_transition.sum())
                BP_meter.add((pred_transition.sum()) * compute_boundary_precision(
                            relax_edge_binary(tis_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance']), 
                            pred_transition),n=pred_transition.sum())
        
        loss_re_xyz_meter.update(re_xyz_loss.item(), input.size(0))
        loss_re_label_meter.update(re_label_loss.item(), input.size(0))
        if args['use_semantic_loss']:
            loss_semantic_meter.update(semantic_loss.item(), input.size(0))

        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(test_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if args['use_semantic_loss']:
            logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                            'LS_re_label {loss_re_label_meter.val:.4f} '
                            'LS_seg_label {loss_semantic_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f}'.format(epoch+1, args["epochs"], i + 1, len(test_loader),
                                                              batch_time=batch_time, data_time=data_time,
                                                              remain_time=remain_time,
                                                              loss_re_xyz_meter=loss_re_xyz_meter,
                                                              loss_re_label_meter=loss_re_label_meter,
                                                              loss_semantic_meter=loss_semantic_meter,
                                                              loss_meter=loss_meter))
        else:
            logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                            'LS_re_label {loss_re_label_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f}'.format(epoch+1, args["epochs"], i + 1, len(test_loader),
                                                              batch_time=batch_time, data_time=data_time,
                                                              remain_time=remain_time,
                                                              loss_re_xyz_meter=loss_re_xyz_meter,
                                                              loss_re_label_meter=loss_re_label_meter,
                                                              loss_meter=loss_meter))

    asa = confusion_matrix.get_overall_accuracy()
    br = BR_meter.value()[0]
    bp = BP_meter.value()[0]
    logger.info('Train result at epoch [{}/{}]: ASA/BR/BP {:.4f}/{:.4f}/{:.4f}'.format(epoch+1, args['epochs'], asa, br, bp))
    logger.info('cnt_room: {} cnt_sp: {} avg_sp: {}'.format(cnt_room, cnt_sp, 1.*cnt_sp/cnt_room))
    logger.info('cnt_sp_act: {} avg_sp_act: {}'.format(cnt_sp_act, 1.*cnt_sp_act/cnt_room))

if __name__ == '__main__':
    main()
