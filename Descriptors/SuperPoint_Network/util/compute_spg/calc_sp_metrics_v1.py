#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:12:49 2018

@author: landrieuloic
"""

"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
import glob, os
import argparse
import numpy as np
import sys
import ast
import csv
import h5py
import torchnet as tnt
import torch
import metrics
from metrics import *


def perfect_prediction(components, labels):
    """assign each superpoint with the majority label"""
    #print(pred.shape)
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')
    
    for i_com in range(len(components)):
        #print(len(components[i_com]))
        #te=labels[components[i_com]]
        #print(te.shape)
        #label_com = np.argmax(np.bincount(labels[components[i_com]]))
        # label_com = labels[components[i_com],1:].sum(0).argmax()
        label_com = labels[components[i_com], :].sum(0).argmax()
        #print(label_com)
        full_pred[components[i_com]]=label_com

    
    return full_pred

def relax_edge_binary(edg_binary, edg_source, edg_target, n_ver, tolerance):
    if torch.is_tensor(edg_binary):
        relaxed_binary = edg_binary.cpu().numpy().copy()
    else:
        relaxed_binary = edg_binary.copy()
    transition_vertex = np.full((n_ver,), 0, dtype = 'uint8')
    for i_tolerance in range(tolerance):
        transition_vertex[edg_source[relaxed_binary.nonzero()]] = True
        transition_vertex[edg_target[relaxed_binary.nonzero()]] = True
        relaxed_binary[transition_vertex[edg_source]] = True
        relaxed_binary[transition_vertex[edg_target]>0] = True
    return relaxed_binary

def partition2ply(filename, xyz, components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
        , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

# data_root='/all/Dataset/S3DIS/superpoint_graphs/'
# data_root='/all/Dataset/S3DIS/spg_ssp_superpoint_graphs/'

file_cnt=0
p_cnt=0
sp_cnt=0
sp_cnt2=0

BR_tolerance=1

#reg_strenth=[0.02,0.03,0.04,0.05,0.06]
reg_strenth=[0.10]
reg_strenth=[0.12]
#data_root='/all/Dataset/vKITTI/new_vkitti/'
#spg_root='/all/hl_cvpr2021/Results/vKITTI/SPG/'
#spg_root='/all/hl_cvpr2021/Results/vKITTI/SPG_norgb/'
# for rs in reg_strenth:
# confusion_matrix_classes = metrics.ConfusionMatrix(13)
# confusion_matrix_BR = metrics.ConfusionMatrix(2)
# confusion_matrix_BP = metrics.ConfusionMatrix(2)

confusion_matrix = metrics.ConfusionMatrix(13)
BR_meter = tnt.meter.AverageValueMeter()
BP_meter = tnt.meter.AverageValueMeter()

data_root = '/test/dataset/3d_datasets/spnet_semanticposs/20201121_point_all_rm_unlabel_2geof'
data_folder = os.path.join(data_root, 'sp_voxel')
fea_folder = os.path.join(data_root, 'points_voxels')

for file in os.listdir(data_folder):
    if '02_' not in file: continue
    file_cnt += 1
    file_name = os.path.join(data_folder, file)
    data_file = h5py.File(file_name, 'r')
    print(file_name)
    # sp_length= np.array(data_file["sp_length"], dtype='float32')
    in_component = np.array(data_file["in_component_v"], dtype='uint32')

    # n_com = len(sp_length)
    grp = data_file['components_v']
    n_com = len(grp)

    components = np.empty((n_com,), dtype=object)

    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()

    #fea_file=fea_folder+file
    fea_file = h5py.File(os.path.join(fea_folder, file), 'r')
    xyz = np.array(fea_file['xyz_v'], dtype='float32')
    labels = np.array(fea_file['labels_v']).squeeze()
    edg_source = np.array(fea_file['source'], dtype='int').squeeze()
    edg_target = np.array(fea_file['target'], dtype='int').squeeze()
    is_transition = np.array(fea_file['is_transition'])

    pred_transition = in_component[edg_source]!=in_component[edg_target]

    print('labels: {}'.format(labels.shape))

    one_hot = np.eye(13)
    labels = one_hot[labels]
    print('labels: {}'.format(labels.shape))

    # exit()
    full_pred= perfect_prediction(components, labels)
    #full_pred= perfect_prediction(components,tlabels)

    # confusion_matrix.count_predicted_batch(labels[:, 1:], full_pred)
    confusion_matrix.count_predicted_batch(labels[:, :], full_pred)
    BR_meter.add((is_transition.sum())*compute_boundary_recall(is_transition, relax_edge_binary(pred_transition, edg_source, edg_target, xyz.shape[0], BR_tolerance)),n=is_transition.sum())
    BP_meter.add((pred_transition.sum())*compute_boundary_precision(relax_edge_binary(is_transition, edg_source, edg_target, xyz.shape[0], BR_tolerance), pred_transition),n=pred_transition.sum())
    #print(confusion_matrix.get_overall_accuracy(),BR_meter.value()[0], BP_meter.value()[0])


    # n_com = len(sp_length)
    sp_cnt+=n_com
    p_cnt+=len(in_component)
#print('======Area_{}====== Finished'.format(i))


print(p_cnt, sp_cnt)
print(p_cnt//sp_cnt)
print(sp_cnt, file_cnt)
print('avg: superpoint {}'.format(sp_cnt//file_cnt))
print(confusion_matrix.get_overall_accuracy(), BR_meter.value()[0], BP_meter.value()[0])