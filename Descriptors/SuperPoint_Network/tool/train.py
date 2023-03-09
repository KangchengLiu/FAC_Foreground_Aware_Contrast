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
from tensorboardX import SummaryWriter

from util import dataset, transform, config
import yaml
from util.util import AverageMeter, intersectionAndUnionGPU
from util.util import simple_write_ply

import torchnet as tnt
import metrics
from metrics import *
from util.util import get_components, perfect_prediction, relax_edge_binary
from util.util import perfect_prediction_base0
from util.util import perfect_prediction_singlelabel

from util.S3DIS_dataset import create_s3dis_datasets, my_collate
from util.VKITTI_dataset import create_vkitti_datasets, my_collate_vkitti
from util.SCANNET_dataset import create_scannet_datasets, my_collate_scannet
from util.SCANNET_dataset_v1 import create_scannet_datasets_v1, my_collate_scannet_v1

from losses import SEAL_loss

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Superpoint Generation')
    parser.add_argument('--config', type=str, default=None, required=True, help='config file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='save path')
    parser.add_argument('--model_path', type=str, default=None, required=True, help='save path')
    parser.add_argument('--weight', type=str, default=None, help='weight')
    parser.add_argument('--resume', type=str, default=None, help='resume')

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["save_path"] = args.save_path
    cfg["model_path"] = args.model_path
    cfg["weight"] = args.weight
    cfg["resume"] = args.resume
    
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
    global args, logger, writer
    args = get_parser()

    logger = get_logger()
    writer = SummaryWriter(os.path.join(args["save_path"], "events"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args["train_gpu"])
    if args["manual_seed"] is not None:
        logger.info("manual_seed is exist")
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.manual_seed(args["manual_seed"])
        np.random.seed(args["manual_seed"])
        torch.manual_seed(args["manual_seed"])
        torch.cuda.manual_seed_all(args["manual_seed"])
    logger.info(args)

def main():
    init()
    
    MODEL = importlib.import_module(args["arch"])
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

    if args['opts'] == 'sgd':
        logger.info('using sgd!!!')
        optimizer = torch.optim.SGD(model.parameters(), lr=args["base_lr"], momentum=args["momentum"], weight_decay=args["weight_decay"])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args["step_epoch"], gamma=args["multiplier"])
    elif args['opts'] == 'adam':
        logger.info('using adam!!!')
        optimizer = torch.optim.Adam(model.parameters(), lr=args['base_lr'], betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=args['weight_decay'])
        scheduler = None
    elif args['opts'] == 'adam_ws':
        logger.info('using adam!!!')
        optimizer = torch.optim.Adam(model.parameters(), lr=args['base_lr'], betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=args['weight_decay'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args["step_epoch"], gamma=args["multiplier"])

    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args["classes"]))
    logger.info(model)
    model = model.cuda()
    
    if args["sync_bn"]:
        from lib.sync_bn import patch_replication_callback
        patch_replication_callback(model)
    
    if args["weight"] is not None:
        if os.path.isfile(args["weight"]):
            logger.info("=> loading weight '{}'".format(args["weight"]))
            checkpoint = torch.load(args["weight"])
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args["weight"]))
        else:
            logger.info("=> no weight found at '{}'".format(args["weight"]))

    if args["resume"] is not None:
        resume_path = os.path.join(args['model_path'], args['resume'])
        if os.path.isfile(resume_path):
            logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda())
            args["start_epoch"] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args["resume"], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(resume_path))
    
    
    train_transform = transform.Compose([transform.ToTensor()])

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


    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=args["train_batch_size"], shuffle=True,
    #                                            num_workers=args["train_workers"], pin_memory=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["train_batch_size"],
                                               shuffle=True,
                                               num_workers=args["train_workers"],
                                               collate_fn=my_collate_s,
                                               #pin_memory=True,
                                               drop_last=True)

    val_loader = None
    if args["evaluate"]:
        pass
        # val_transform = transform.Compose([transform.ToTensor()])
        # val_data = dataset.PointData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        # val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size_val, shuffle=False, num_workers=args.train_workers, pin_memory=True)
    
    for epoch in range(args["start_epoch"], args["epochs"]):
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, optimizer, epoch)
        epoch_log = epoch + 1
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if epoch_log % args["save_freq"] == 0:
            filename = os.path.join(args["save_path"], 'saved_model/train_epoch_{}.pth'.format(str(epoch_log)))
            logger.info('Saving checkpoint to: ' + filename)
            if scheduler is not None:
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)
            else:
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        
        if args["evaluate"]:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

        if scheduler is not None:
            scheduler.step()

def label2one_hot(labels, C=13):
    b, n = labels.shape
    labels = torch.unsqueeze(labels, dim=1)
    one_hot = torch.zeros(b, C, n, dtype=torch.long).cuda()         # create black
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
    return target.type(torch.float32)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train(train_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_re_xyz_meter = AverageMeter()
    loss_re_label_meter = AverageMeter()
    loss_re_sp_meter = AverageMeter()
    loss_semantic_meter = AverageMeter()
    loss_seal_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    gt_meter = AverageMeter()
    
    BR_meter = tnt.meter.AverageValueMeter()
    BP_meter = tnt.meter.AverageValueMeter()
    confusion_matrix = metrics.ConfusionMatrix(args['classes'])

    model.train()
    end = time.time()
    max_iter = args["epochs"] * len(train_loader)
    print('$'*10)
    
    # is_transition: bn*5
    for i, (fname, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz) in enumerate(train_loader): 
        data_time.update(time.time() - end)
        if args['data_name'] in ['scannet_v1', 'semanticposs']:
            gt = labels[:, :, :].argmax(axis=2)
        else:
            gt = labels[:, :, 1:].argmax(axis=2)
        
        input = torch.cat((xyz, clouds_global), dim=2)
        input = input.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        onehot_label = label2one_hot(gt, args['classes'])
        clouds = clouds.cuda(non_blocking=True)
        # logger.info('fname: {} input: {}'.format(fname[0], input[0, 0, :]))
        # if i == 10: break
        # continue
        # clouds:               bn x 6 x 20
        # clouds_global:        bn x 7
        # edg_source len:       b
        # edg_target len:       b
        # is_transition len:    b
        # labels:               b x n x 14    n=(14999+1)
        # objects:              bn
        # clouds:               bn x fea_dim x 20
        # clouds_global:        b x n x 7
        # xyz:                  b x n x 3 
        # onehot_label:         b x classes x n
        # input:                b x n x 10
        # gt:                   b x n

        spout, c_idx, c2p_idx, c2p_idx_base, output, rec_xyz, rec_label, fea_dist, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot = model(input, clouds, onehot_label, gt.unsqueeze(-1))
        # ---------------- superpoint realted ------------------
        # spout:        b x n x nc2p
        # c_idx:        b x m           in 0,1,2,...,n
        # c2p_idx:      b x n x nc2p    in 0,1,2,...,n
        # c2p_idx_base: b x n x nc2p    in 0,1,2,...,m-1
        # ---------------- semantic related --------------------
        # output:       b x classes x n
        # p_fea:        b x n x c
        # sp_pred_lab:  b x classes x m
        # sp_pseudo_lab:b x classes x m
        # ------------------- debug ----------------------------
        # ret-asso_matrix: b x m x n    ok
        # ret-sp_nei_cnt: b x m x 1     ok
        # asso_matrix, sp_nei_cnt = ret
        
        if gt.shape[-1] == 1:
            gt = gt[:, 0]  # for cls

        re_xyz_loss = args['w_re_xyz_loss'] * criterion_re_xyz(rec_xyz, input[:, :, :3].transpose(1,2).contiguous())
        if args['re_label_loss'] == 'cel':
            re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label, gt)
        elif args['re_label_loss'] == 'mse':
            re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label, onehot_label)

        if args['re_sp_loss'] == 'cel':
            re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab)
        elif args['re_sp_loss'] == 'mse':
            re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab_onehot)


        loss = re_xyz_loss + re_label_loss + re_sp_loss
        if args['use_semantic_loss'] == 1:
            semantic_loss = criterion(output, gt)
            loss = loss + args['w_semantic_loss'] * semantic_loss
        

        # calcuate superpoint metric
        for bid in range(args['train_batch_size']):
            txyz = input[bid, :, :3].cpu().numpy()
            tpfea = p_fea[bid, :, :]
            tedg_source = edg_source[bid]
            tedg_target = edg_target[bid]
            tis_transition = is_transition[bid].numpy()

            tobjects = objects[bid*(args['num_point']+1):(bid+1)*(args['num_point']+1)].numpy()
            spout_ = spout[bid, :, :].detach().cpu().numpy()
            init_center = c_idx[bid, :].cpu().numpy()
            pt_center_index = c2p_idx_base[bid, :, :].cpu().numpy()
            pred_components, pred_in_component = get_components(init_center, pt_center_index, spout_, getmax=True)
            pred_components = [x[0] for x in pred_components]
            
            pred_transition = pred_in_component[tedg_source] != pred_in_component[tedg_target]
            if args['data_name'] in ['scannet_v1', 'semanticposs']:
                full_pred = perfect_prediction_base0(pred_components, pred_in_component, labels[bid, :, :].numpy())
            else:
                full_pred = perfect_prediction(pred_components, pred_in_component, labels[bid, :, :].numpy())
            
            if args['use_semantic_loss'] == 2:
                if bid == 0:
                    # pred_components: (m, x) number of superpoints, each superpoint save the idx of its points
                    # pred_in_component: (n,) where n is the number of points
                    seal_loss_a, seal_loss_b = SEAL_loss(args['num_point']+1, pred_components, pred_in_component, tobjects, tedg_source, tedg_target, tis_transition, tpfea, transition_factor=5)
                else:
                    tmpa, tmpb = SEAL_loss(args['num_point']+1, pred_components, pred_in_component, tobjects, tedg_source, tedg_target, tis_transition, tpfea, transition_factor=5)
                    seal_loss_a = seal_loss_a + tmpa
                    seal_loss_b = seal_loss_b + tmpb


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
        
        if args['use_semantic_loss'] == 2:
            seal_loss = args['w_semantic_loss'] * (seal_loss_a + seal_loss_b) / args['train_batch_size']
            loss = loss + seal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        intersection, union, gt = intersectionAndUnionGPU(output, gt, args["classes"], args["ignore_label"])
        intersection, union, gt = intersection.cpu().numpy(), union.cpu().numpy(), gt.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), gt_meter.update(gt)
       
        accuracy = sum(intersection_meter.val) / (sum(gt_meter.val) + 1e-10)
        loss_re_xyz_meter.update(re_xyz_loss.item(), input.size(0))
        loss_re_label_meter.update(re_label_loss.item(), input.size(0))
        loss_re_sp_meter.update(re_sp_loss.item(), input.size(0))

        if args['use_semantic_loss'] == 1:
            loss_semantic_meter.update(semantic_loss.item(), input.size(0))
        if args['use_semantic_loss'] == 2:
            loss_seal_meter.update(seal_loss.item(), input.size(0))
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        

        now_lr = get_lr(optimizer)
        if args['use_semantic_loss'] == 1:
            if (i + 1) % args["print_freq"] == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                            'LS_re_label {loss_re_label_meter.val:.4f} '
                            'LS_seg_label {loss_semantic_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '
                            'lr {lr:.6f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch+1, args["epochs"], i + 1, len(train_loader),
                                                              batch_time=batch_time, data_time=data_time,
                                                              remain_time=remain_time,
                                                              loss_re_xyz_meter=loss_re_xyz_meter,
                                                              loss_re_label_meter=loss_re_label_meter,
                                                              loss_semantic_meter=loss_semantic_meter,
                                                              loss_meter=loss_meter,
                                                              lr=now_lr,
                                                              accuracy=accuracy))
        elif args['use_semantic_loss'] == 2:
            if (i + 1) % args["print_freq"] == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                            'LS_re_label {loss_re_label_meter.val:.4f} '
                            'LS_seal_label {loss_seal_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '
                            'lr {lr:.6f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch+1, args["epochs"], i + 1, len(train_loader),
                                                              batch_time=batch_time, data_time=data_time,
                                                              remain_time=remain_time,
                                                              loss_re_xyz_meter=loss_re_xyz_meter,
                                                              loss_re_label_meter=loss_re_label_meter,
                                                              loss_seal_meter=loss_seal_meter,
                                                              loss_meter=loss_meter,
                                                              lr=now_lr,
                                                              accuracy=accuracy))
        else:
            if (i + 1) % args["print_freq"] == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                            'LS_re_label {loss_re_label_meter.val:.4f} '
                            'LS_re_sp {loss_re_sp_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '
                            'lr {lr:.6f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch+1, args["epochs"], i + 1, len(train_loader),
                                                              batch_time=batch_time, data_time=data_time,
                                                              remain_time=remain_time,
                                                              loss_re_xyz_meter=loss_re_xyz_meter,
                                                              loss_re_label_meter=loss_re_label_meter,
                                                              loss_re_sp_meter=loss_re_sp_meter,
                                                              loss_meter=loss_meter,
                                                              lr=now_lr,
                                                              accuracy=accuracy))
        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (gt + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    asa = confusion_matrix.get_overall_accuracy()
    br = BR_meter.value()[0]
    bp = BP_meter.value()[0]
    logger.info('Train result at epoch [{}/{}]: ASA/BR/BP {:.4f}/{:.4f}/{:.4f}'.format(
                epoch+1, args['epochs'], asa, br, bp))
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (gt_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = np.sum(intersection_meter.sum) / (np.sum(gt_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(
                epoch+1, args["epochs"], mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        output = model(input)
        loss = criterion(output, target)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args["classes"], args["ignore_label"])
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args["print_freq"] == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = np.sum(intersection_meter.sum) / (np.sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args["classes"]):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc

if __name__ == '__main__':
    main()
