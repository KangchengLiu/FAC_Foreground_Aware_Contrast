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
from learning import metrics
from learning.metrics import *


def perfect_prediction(components,labels):
	"""assign each superpoint with the majority label"""
	#print(pred.shape)
	full_pred = np.zeros((labels.shape[0],),dtype='uint32')
	
	for i_com in range(len(components)):
		#print(len(components[i_com]))
		#te=labels[components[i_com]]
		#print(te.shape)
		#label_com = np.argmax(np.bincount(labels[components[i_com]]))
		label_com = labels[components[i_com],1:].sum(0).argmax()
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

#data_root='/all/Dataset/S3DIS/superpoint_graphs/'
#data_root='/all/Dataset/S3DIS/spg_ssp_superpoint_graphs/'

file_cnt=0
p_cnt=0
sp_cnt=0
sp_cnt2=0

BR_meter = tnt.meter.AverageValueMeter()
BP_meter = tnt.meter.AverageValueMeter()
confusion_matrix = metrics.ConfusionMatrix(13)
BR_tolerance=1

#reg_strenth=[0.02,0.03,0.04,0.05,0.06]
reg_strenth=[0.10]
reg_strenth=[0.12]
for rs in reg_strenth:
	confusion_matrix_classes = metrics.ConfusionMatrix(13)
	confusion_matrix_BR = metrics.ConfusionMatrix(2)
	confusion_matrix_BP = metrics.ConfusionMatrix(2)

	res='/all/hl_cvpr2021/Results/S3DIS/SPG/res_file/rs_{}_A5.h5'.format(rs)
	#res='/test/hl_cvpr2021/Results/S3DIS/SPG/6_fold/rs_{}/'.format(rs)
	#res='/all/hl_cvpr2021/Results/S3DIS/SPG_norgb/6_fold/rs_{}/'.format(rs)
	#res='/all/hl_cvpr2021/Results/S3DIS/SPG_norgb/A5/rs_{}/res.h5'.format(rs)
	#res='/all/hl_cvpr2021/Results/ScanNet/SPG/results/rs_{}.h5'.format(rs)
	#res='/all/hl_cvpr2021/Results/ScanNet/SPG_norgb/results/rs_{}.h5'.format(rs)

	#res='/all/hl_cvpr2021/Results/vKITTI/SPG/rs_0.03.h5'
	#res='/all/hl_cvpr2021/Results/vKITTI/SPG/rs_0.03.h5'
	#res='/all/hl_cvpr2021/Results/vKITTI/SPG/rs_0.5.h5'
	#res='/all/hl_cvpr2021/Results/vKITTI/SPG_norgb/rs_0.5.h5'
	print(res)
	C_classes = np.zeros((13,13))
	C_BR = np.zeros((2,2))
	C_BP = np.zeros((2,2))

	#res_file = h5py.File(res+'res.h5', 'r')
	res_file = h5py.File(res, 'r')
	c_classes = np.array(res_file["confusion_matrix_classes"])
	c_BP = np.array(res_file["confusion_matrix_BP"])
	c_BR = np.array(res_file["confusion_matrix_BR"])
	n_sp= np.array(res_file["average_superpoint"])
	#print(n_sp)
	print('='*20)
	print('rs={}'.format(rs))
	print("n_sp = %5.1f \t ASA = %3.2f %% \t BR = %3.2f %% \t BP = %3.2f %%" %  \
		 (n_sp,100 * c_classes.trace() / c_classes.sum(), 100 * c_BR[1,1] / (c_BR[1,1] + c_BR[1,0]),100 * c_BP[1,1] / (c_BP[1,1] + c_BP[0,1]) ))


