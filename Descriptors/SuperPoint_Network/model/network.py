#!/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from lib.pointops.functions import pointops
from util import pt_util
from modules import LPE_stn
from modules import init_fea, compute_final_assignments, calc_sp_fea
from modules import learn_SLIC_calc_v1_new

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.debug = cfg['for_debug']
        self.clusters = int(cfg['num_point']*cfg['rate'])  # m=int(n*rate)=120, rate=0.008
        self.rate = cfg['rate']
        self.np = cfg['num_point']
        self.use_xyz = cfg['use_xyz']               # True
        self.use_softmax = cfg['use_softmax']       # True
        self.use_norm = cfg['use_norm']             # True
        self.nsample = cfg['nsample']               # knn: 20
        self.fea_dim = cfg['fea_dim']
        self.nc2p = cfg['near_clusters2point']      # 6
        self.np2c = cfg['near_points2cluster']      # 100
        self.classes = cfg['classes']
        self.drop = cfg['dropout']

        add = 0
        if self.use_xyz: add = 3

        self.backbone= LPE_stn(input_channels=2, cfg=cfg)

        self.learn_SLIC_calc_1 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[64, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[64, 16, 16],
                            bn=True, use_xyz=self.use_xyz, use_softmax=self.use_softmax, use_norm=self.use_norm)

        self.learn_SLIC_calc_2 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[64, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[64, 16, 16],
                            bn=True, use_xyz=self.use_xyz, use_softmax=self.use_softmax, use_norm=self.use_norm)

        self.learn_SLIC_calc_3 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[64, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[64, 16, 16],
                            bn=True, use_xyz=self.use_xyz, use_softmax=self.use_softmax, use_norm=self.use_norm)

        self.learn_SLIC_calc_4 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[64, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[64, 16, 16],
                            bn=True, use_xyz=self.use_xyz, use_softmax=self.use_softmax, use_norm=self.use_norm, last=True)

        self.mlp = nn.Sequential(pt_util.SharedMLP_1d([64, 64], bn=True),
                                 pt_util.SharedMLP_1d([64, self.classes], activation=None))
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        fea = pc[..., 3:].contiguous()
        return xyz, fea

    def label2one_hot(self, labels, C=13):
        b, n = labels.shape
        labels = torch.unsqueeze(labels, dim=1)
        one_hot = torch.zeros(b, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)

    def forward(self, pointcloud: torch.cuda.FloatTensor, clouds_knn, onehot_label=None, label=None):
        # pointcloud: b x n x (3+c)
        # clouds_knn: bn x 6 x 20
        # onehot_label: b x c x n
        # label: b x n x 1
        
        batch_size, num_points, _ = pointcloud.size()

        xyz, clouds_global = self._break_up_pc(pointcloud)  # xyz: b x n x 3  clouds_global: b x n x c
        # xyz: b x n x 3
        # clouds_global: b x n x c

        xyz_trans = xyz.transpose(1, 2).contiguous()        # b x 3 x n

        # ------------------------------------ prepare --------------------------------------------
        # number of clusters for FPS
        num_clusters = int(xyz.size(1) * self.rate)
        
        # -------------------------------------------------------
        # find knn index of each point
        # knn_idx = pointops.knnquery(self.nsample, xyz, xyz) # knn index: b x n x k
        
        # --------------------------------------------------------
        # calculate idx of superpoints and points
        cluster_idx = pointops.furthestsampling(xyz, num_clusters) 
        # index: b x m  check ok
        cluster_xyz = pointops.gathering(xyz_trans, cluster_idx).transpose(1, 2).contiguous()
        # b x m x 3
        
        # c2p_idx: near clusters to each point
        # (b x m x 3, b x m, b x n x 3) -> b x n x nc2p, b x n x nc2p
        # nc2p == 6
        c2p_idx, c2p_idx_abs = pointops.knnquerycluster(self.nc2p, cluster_xyz, cluster_idx, xyz)
        # c2p_idx: b x n x 6
        # c2p_idx_abs: b x n x 6
        
        # association matrix
        asso_matrix, sp_nei_cnt, sp_lab = pointops.assomatrixpluslabel(self.nc2p, c2p_idx, label.int(), cluster_idx.unsqueeze(-1), self.classes)
        asso_matrix = asso_matrix.float()
        sp_nei_cnt = sp_nei_cnt.float()
        # asso_matrix: b x m x n
        # sp_nei_cnt: b x m x 1
        # sp_lab: b x m x class
        
        # ----------------------- embedding ----------------------------
        clouds_knn = clouds_knn.view(batch_size, num_points, clouds_knn.size(1), clouds_knn.size(2)).contiguous()
        clouds_knn = clouds_knn.transpose(1, 2).contiguous()
        
        embedding = self.backbone(clouds_knn, clouds_global.transpose(1, 2).contiguous())
        # embedding: b x 32 x n
        
        p_fea = embedding.transpose(1, 2).contiguous()

        out = self.mlp(p_fea.transpose(1, 2).contiguous())

        sp_fea = init_fea(p_fea, asso_matrix, sp_nei_cnt)
        # cluster_xyz: b x m x 3
        # sp_fea: b x m x 32         initial superpoints features

        # c2p_idx: b x n x 6
        sp_fea, cluster_xyz = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, xyz, c2p_idx_abs, c2p_idx, cluster_idx)
        # sp_fea: b x m x c

        sp_fea, cluster_xyz = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, xyz, c2p_idx_abs, c2p_idx, cluster_idx)
        # sp_fea: b x m x c

        sp_fea, cluster_xyz = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, xyz, c2p_idx_abs, c2p_idx, cluster_idx)
        # sp_fea: b x m x c

        fea_dist, sp_fea, cluster_xyz = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, xyz, c2p_idx_abs, c2p_idx, cluster_idx)
        # sp_fea: b x m x c
        
        final_asso = fea_dist
        
        if onehot_label is not None:
            # ------------------------------ reconstruct xyz ----------------------------
            sp_xyz = cluster_xyz
            # sp_xyz: b x m x 3

            p2sp_idx = torch.argmax(final_asso, dim=2, keepdim=False)
            # p2sp_idx: b x n

            # (b x 3 x m,  b x n, b x n x 6) -> (b x 3 x n)
            re_p_xyz = pointops.gathering_cluster(sp_xyz.transpose(1, 2).contiguous(), p2sp_idx.int(), c2p_idx_abs)
            # re_p_xyz: b x 3 x n
            # (b, c, n), idx : (b, m) tensor, idx_3d: (b, m, k)

            # ------------------------------ reconstruct label ----------------------------
            # onehot_label: b x classes x n
            sp_label = calc_sp_fea(c2p_idx, final_asso, onehot_label.transpose(1, 2).contiguous(), c2p_idx_abs, self.nc2p, c2p_idx, cluster_idx)
            # sp_label: b x m x classes
            
            sp_pseudo_lab = torch.argmax(sp_lab, dim=2, keepdim=False)  # b x m
            sp_pseudo_lab_onehot = self.label2one_hot(sp_pseudo_lab, self.classes)    # b x class x m
            # c2p_idx: b x n x 6
            # final_asso: b x n x 6
            # f: b x n x m
            # (b, n, m) X (b, m, classes) -> (b, n, classes)
            # re_p_label = torch.matmul(f, sp_label)
            # re_p_label: b x n x classes
            
            # (b, classes, m), (b, n, 6) -> b x classes x n x 6
            c2p_label = pointops.grouping(sp_label.transpose(1, 2).contiguous(), c2p_idx_abs)
            # (b, classes, m), (b, n, 6) -> b x classes x n x 6
            
            re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(1), dim=-1, keepdim=False)
            # re_p_label: b x classes x n
        else:
            re_p_xyz = None
            re_p_label = None

        return final_asso, cluster_idx, c2p_idx, c2p_idx_abs, out, re_p_xyz, re_p_label, fea_dist, p_fea, sp_label.transpose(1, 2).contiguous(), sp_pseudo_lab, sp_pseudo_lab_onehot
