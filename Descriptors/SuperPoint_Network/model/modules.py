#!/usr/bin/env python
# coding=utf-8

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import pt_util
from lib.pointops.functions import pointops

class learn_SLIC_calc_v1_new(nn.Module):
    """
    update features between superpoints and points
    """
    def __init__(self, ch_wc2p_fea: List[int], ch_wc2p_xyz: List[int], ch_mlp: List[int],
                 bn=True, use_xyz=True, use_softmax=True, use_norm=True, last=False):
        super().__init__()
        self.bn = bn
        self.use_xyz = use_xyz
        self.use_softmax = use_softmax
        self.use_norm = use_norm
        self.last = last

        self.w_c2p_fea = pt_util.SharedMLP(ch_wc2p_fea, bn=self.bn)
        self.w_c2p_xyz = pt_util.SharedMLP(ch_wc2p_xyz, bn=self.bn)
        self.mlp = pt_util.SharedMLP_1d(ch_mlp, bn=self.bn)

    
    def forward(self, sp_fea, sp_xyz, o_p_fea, p_xyz, c2p_idx_abs, c2p_idx, cluster_idx):
        # sp_fea: b x m x c
        # sp_xyz: b x m x 3
        # o_p_fea: b x n x c
        # p_xyz: b x n x 3
        # c2p_idx_abs: b x n x nc2p
        bs, n, nc2p = c2p_idx_abs.size()

        c2p_fea = pointops.grouping(sp_fea.transpose(1, 2).contiguous(), c2p_idx_abs) - o_p_fea.transpose(1, 2).contiguous().unsqueeze(-1).repeat(1, 1, 1, nc2p)
        # c2p_fea: b x c x n x nc2p
        
        c2p_xyz = pointops.grouping(sp_xyz.transpose(1, 2).contiguous(), c2p_idx_abs) - p_xyz.transpose(1, 2).contiguous().unsqueeze(-1).repeat(1, 1, 1, nc2p)
        # c2p_xyz: b x 3 x n x nc2p

        p_fea = self.mlp(o_p_fea.transpose(1, 2).contiguous())    # b x 16 x n
        c2p_fea = self.w_c2p_fea(c2p_fea)   # b x 16 x n x nc2p
        c2p_xyz = self.w_c2p_xyz(c2p_xyz)   # b x 16 x n x nc2p

        p_fea = p_fea.transpose(1, 2).contiguous().view(bs*n, p_fea.size(1))    # bn x 16
        if self.use_norm:
            p_fea = F.normalize(p_fea, p=2, dim=1)
        
        w_fea = c2p_fea.transpose(1, 2).contiguous().view(bs*n, c2p_fea.size(1), nc2p)  # bn x 16 x nc2p
        if self.use_norm:
            w_fea = F.normalize(w_fea, p=2, dim=1)
        
        w_xyz = c2p_xyz.transpose(1, 2).contiguous().view(bs*n, c2p_xyz.size(1), nc2p)  # bn x 16 x nc2p
        if self.use_norm:
            w_xyz = F.normalize(w_xyz, p=2, dim=1)

        new_w_fea = torch.matmul(p_fea.unsqueeze(1), w_fea)     # (bn, 1, 16) X (bn, 16, nc2p) -> (bn, 1, nc2p)
        new_w_xyz = torch.matmul(p_fea.unsqueeze(1), w_xyz)     # (bn, 1, 16) X (bn, 16, nc2p) -> (bn, 1, nc2p)

        bi_w = (new_w_fea * new_w_xyz).unsqueeze(1).view(bs, n, nc2p)   # b x n x nc2p
        if self.use_softmax:
            bi_w = F.softmax(bi_w, dim=-1)  # b x n x nc2p

        f, sp_nei_cnt = pointops.assomatrixfloat(nc2p, bi_w, c2p_idx, cluster_idx.unsqueeze(-1))
        # f: b x m x n
        # sp_nei_cnt: b x m x 1

        sp_sum = f.sum(dim=2, keepdim=True)                 # b x m x 1
        sp_fea = torch.matmul(f, o_p_fea) / (sp_sum+1e-8)   # (b, m, n) X (b, n, c) -> (b, m, c)
        
        sp_xyz = torch.matmul(f, p_xyz) / (sp_sum+1e-8)     # (b, m, n) X (b, n, 3) -> (b, m, 3)
        

        if self.last:
            return bi_w, sp_fea, sp_xyz
        return sp_fea, sp_xyz

def init_fea(p_fea, asso_matrix, sp_nei_cnt):
    # p_fea: b x n x 32
    # asso_matrix: b x m x n
    # sp_nei_cnt: b x m x 1

    sp_fea = torch.matmul(asso_matrix, p_fea) / sp_nei_cnt

    return sp_fea       # b x m x 32

def compute_final_assignments(pt_center_index, sp_fea, p_fea, c2p_idx_abs):
    # pt_center_index: b x n x 6
    # sp_fea: b x m x c
    # p_fea: b x n x c             
      
    # b x c x n x 6 - b x c x n x 1
    tp_fea = p_fea.transpose(1, 2).contiguous()     # b x c x n
    tp_fea = tp_fea.unsqueeze(-1).repeat(1, 1, 1, pt_center_index.size(2))  # b x c x n x 6
    nn_sp_fea = pointops.grouping(sp_fea.transpose(1, 2).contiguous(), c2p_idx_abs)     # b x c x n x 6

    fea_dist = -torch.norm(nn_sp_fea - tp_fea, p=2, dim=1, keepdim=False)   # b x n x 6
    asso = (fea_dist - fea_dist.max(dim=2, keepdim=True)[0]).exp()          # b x n x 6
    asso = asso / (asso.sum(dim=2, keepdim=True))                           # b x n x 6

    return fea_dist, asso   # asso: b x n x 6

def calc_sp_fea(pt_center_index, pt_asso, p_fea, c2p_idx_abs, nc2p, c2p_idx, cluster_idx):
    # pt_center_index: b x n x 6
    # pt_asso: b x n x 6 
    # p_fea: b x n x c
    # num: m
    
    # b x m x n
    f, sp_nei_cnt = pointops.assomatrixfloat(nc2p, pt_asso, c2p_idx, cluster_idx.unsqueeze(-1))

    sp_sum = f.sum(dim=2, keepdim=True)                 # b x m x 1
    sp_fea = torch.matmul(f, p_fea) / (sp_sum+1e-8)     # (b x m x n) X (b x n x c) -> (b x m x c) 

    return sp_fea   # b x m x c

class LPE_stn(nn.Module):
    def __init__(self, input_channels, cfg):
        super(LPE_stn, self).__init__()
        self.stn=LPE_STNKD(input_channels)
        add = 0
        if cfg['use_rgb']:
            add += 3
        if cfg['ver_value'] == 'geof':
            add += 4
        self.convs=nn.Sequential(
            nn.Conv2d(3+add, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fcs=nn.Sequential(
            nn.Conv1d(136+add, 64, 1),
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, clouds, clouds_global):
        # b x c x n x 20
        # b x 7 x n
        # out: bn x 32
        
        batch_size, _, num_point, _ = clouds.size()
        t_m = self.stn(clouds[:, :2, :, :])    # b x 2 x n x 20 -> bn x 2 x 2
        
        # (bn, 20, 2) x (bn, 2, 2) -> (bn, 20, 2) -> (bn, 2, 20)
        # b x 2 x n x 20 -> b x n x 2 x 20 -> bn x 2 x 20
        tmp = clouds[:,:2,:,:].transpose(1,2).contiguous().view(-1, 2, 20).contiguous()
        # bn x 20 x 2
        transf=torch.bmm(tmp.transpose(1, 2).contiguous(), t_m).transpose(1,2).contiguous()

        # bn x 2 x 20 -> b x n x 2 x 20 -> b x 2 x n x 20
        transf = transf.view(batch_size, num_point, 2, 20).contiguous().transpose(1, 2).contiguous()

        clouds = torch.cat([transf, clouds[:,2:,:,:]], 1)   # b x c x n x 20

        # clouds_global: b x 7 x n + bn x 4 -> b x 7 x n + b x n x 4 -> b x 7 x n + b x 4 x n -> b x 11 x n
        clouds_global = torch.cat([clouds_global, t_m.view(-1,4).contiguous().view(batch_size, num_point, 4).contiguous().transpose(1, 2).contiguous()], 1)
        
        clouds=self.convs(clouds)   # b x 128 x n x 20
        clouds=clouds.max(dim=-1, keepdim=False)[0]     # b x 128 x n
        clouds=torch.cat([clouds, clouds_global],1)     # b x 139 x n
        return self.fcs(clouds)

class LPE_STNKD(nn.Module):
    def __init__(self, input_channels=2):
        super(LPE_STNKD, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 32, 1), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, input_channels * input_channels, 1)
        )

    def forward(self, x):
        # x: b x 2 x n x 20
        
        x = self.mlp1(x)    # b x 128 x n x 20
        x = x.max(dim=-1, keepdim=False)[0]   # b x 128 x n

        x = self.mlp2(x)    # b x 4 x n
        x = x.transpose(1, 2).contiguous().view(-1, 4).contiguous()     # bn x 4
        I = torch.eye(self.input_channels).view(-1).contiguous().to(x.device)   # 4
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x
