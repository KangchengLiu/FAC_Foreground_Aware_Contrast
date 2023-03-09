'''

@author and maintained by: Dr. Kangcheng Liu 

'''

import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from class_util import classes, class_to_id, class_to_color_rgb
import itertools
import random
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import math
import networkx as nx
from scipy.cluster.vq import vq, kmeans
import time
import matplotlib.pyplot as plt
import scipy.special
from learn_region_grow_util import *

numpy.random.seed(0)
NUM_POINT = 512
NUM_NEIGHBOR_POINT = 512
num_neighbors = 50
neighbor_radii = 0.3
embedding_size = 10
FEATURE_SIZE = 19
TEST_AREAS = [1,2,3,4,5,6,'scannet']
resolution = 0.1
classification_threshold = 0.5
completion_threshold = 0.5
cluster_threshold = 10
save_results = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []

for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		TEST_AREAS = sys.argv[i+1].split(',')
	elif sys.argv[i]=='--save':
		save_results = True

for AREA in TEST_AREAS:
	tf.reset_default_graph()
	if AREA=='scannet':
		MCP_PATH = 'models/mcpnet_model%s.ckpt'%'6'
		MODEL_PATH = 'models/lrgnet_model%s.ckpt'%'6'
	else:
		MCP_PATH = 'models/mcpnet_model%s.ckpt'%AREA
		MODEL_PATH = 'models/lrgnet_model%s.ckpt'%AREA
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)
	net = LrgNet(1, NUM_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_PATH)
	print('Restored from %s'%MODEL_PATH)
	mcpnet = MCPNet(1, 50, 6, 200, 10)
	saver = tf.train.Saver(mcpnet.kernels + mcpnet.biases)
	saver.restore(sess, MCP_PATH)
	print('Restored from %s'%MCP_PATH)

	if AREA=='scannet':
		all_points,all_obj_id,all_cls_id = loadFromH5('data/scannet.h5')
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)

	for room_id in range(len(all_points)):
#	for room_id in [0]:
		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]
		centroid = 0.5 * (unequalized_points[:,:2].min(axis=0) + unequalized_points[:,:2].max(axis=0))
		unequalized_points[:,:2] -= centroid
		unequalized_points[:,2] -= unequalized_points[:,2].min()

		#equalize resolution
		equalized_idx = []
		equalized_map = {}
		coarse_map = {}
		unequalized_idx = []
		normal_grid = {}
		for i in range(len(unequalized_points)):
			k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_map:
				equalized_map[k] = len(equalized_idx)
				equalized_idx.append(i)
				kk = tuple(numpy.round(unequalized_points[i,:3]/neighbor_radii).astype(int))
				if not kk in coarse_map:
					coarse_map[kk] = []
				coarse_map[kk].append(equalized_map[k])
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
			unequalized_idx.append(equalized_map[k])
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]

		#compute normals
		normals = []
		curvatures = []
		for i in range(len(points)):
			k = tuple(numpy.round(points[i,:3]/resolution).astype(int))
			neighbors = []
			for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
				kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
				if kk in normal_grid:
					neighbors.extend(normal_grid[kk])
			accA = numpy.zeros((3,3))
			accB = numpy.zeros(3)
			for n in neighbors:
				p = unequalized_points[n,:3]
				accA += numpy.outer(p,p)
				accB += p
			cov = accA / len(neighbors) - numpy.outer(accB, accB) / len(neighbors)**2
			U,S,V = numpy.linalg.svd(cov)
			normals.append(numpy.fabs(V[2]))
			curvature = S[2] / (S[0] + S[1] + S[2])
			curvatures.append(numpy.fabs(curvature)) # change to absolute values?
		curvatures = numpy.array(curvatures)
		normals = numpy.array(normals)

		#compute neighbors for each point
		neighbor_array = numpy.zeros((len(points), num_neighbors, 6), dtype=float)
		for i in range(len(points)):
			p = points[i,:6]
			k = tuple(numpy.round(points[i,:3]/neighbor_radii).astype(int))
			neighbors = []
			for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
				kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
				if kk in coarse_map:
					neighbors.extend(coarse_map[kk])
			neighbors = numpy.random.choice(neighbors, num_neighbors, replace=len(neighbors)<num_neighbors)
			neighbors = points[neighbors, :6].copy()
			neighbors -= p
			neighbor_array[i,:,:] = neighbors

		#compute embedding for each point
		embeddings = numpy.zeros((len(points), embedding_size), dtype=float)
		input_points = numpy.zeros((1, 4), dtype=float)
		input_neighbors = numpy.zeros((1, num_neighbors, 6), dtype=float)
		for i in range(len(points)):
			input_points[0,:] = points[i, 2:6]	
			input_neighbors[0,:,:] = neighbor_array[i, :, :6]
			emb_val = sess.run(mcpnet.embeddings, {mcpnet.input_pl:input_points, mcpnet.neighbor_pl:input_neighbors})
			embeddings[i] = emb_val


		points = numpy.hstack((points, normals, embeddings)).astype(numpy.float32)
		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		cluster_label = numpy.zeros(len(points), dtype=int)
		cluster_id = 1
		visited = numpy.zeros(len(point_voxels), dtype=bool)
		input_points = numpy.zeros((1, NUM_POINT, FEATURE_SIZE), dtype=numpy.float32)
		neighbor_points = numpy.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=numpy.float32)
		input_classes = numpy.zeros((1, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
		#iterate over each object in the room
#		for seed_id in range(len(point_voxels)):
		for seed_id in numpy.arange(len(points))[numpy.argsort(curvatures)]:
			if visited[seed_id]:
				continue
			seed_voxel = point_voxels[seed_id]
			target_id = obj_id[seed_id]
			gt_mask = obj_id==target_id
			obj_voxels = point_voxels[gt_mask]
			obj_voxel_set = set([tuple(p) for p in obj_voxels])
			original_minDims = obj_voxels.min(axis=0)
			original_maxDims = obj_voxels.max(axis=0)
#			print('original',numpy.sum(gt_mask), original_minDims, original_maxDims)
			currentMask = numpy.zeros(len(points), dtype=bool)
			currentMask[seed_id] = True
			minDims = seed_voxel.copy()
			maxDims = seed_voxel.copy()
			steps = 0

			#perform region growing
			while True:

				def stop_growing(reason):
					global cluster_id
					visited[currentMask] = True
					if numpy.sum(currentMask) > cluster_threshold:
						cluster_label[currentMask] = cluster_id
						cluster_id += 1
					iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
#					print('room %d target %3d: step %3d %4d/%4d points IOU %.2f cls %.3f cmpl %.2f %s'%(room_id, target_id, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, cls_acc, cmpl_conf, reason))

				#determine the current points and the neighboring points
				currentPoints = points[currentMask, :].copy()
				expandPoints = []
				expandClass = []
				for a in range(len(action_map)):
					if a==0:
						mask = numpy.logical_and(numpy.all(point_voxels>=minDims,axis=1), numpy.all(point_voxels<=maxDims, axis=1))
						mask = numpy.logical_and(mask, numpy.logical_not(currentMask))
					else:
						newMinDims = minDims.copy()	
						newMaxDims = maxDims.copy()	
						expand_dim = numpy.nonzero(action_map[a])[0][0] % 3
						if numpy.sum(action_map[a])>0:
							newMinDims[expand_dim] = newMaxDims[expand_dim] = maxDims[expand_dim]+1
						else:
							newMinDims[expand_dim] = newMaxDims[expand_dim] = minDims[expand_dim]-1
						mask = numpy.logical_and(numpy.all(point_voxels>=newMinDims,axis=1), numpy.all(point_voxels<=newMaxDims, axis=1))
					mask = numpy.logical_and(mask, numpy.logical_not(visited))
					expandPoints.extend(points[mask,:].copy())
					#determine which neighboring points should be added
					expandClass.extend(obj_id[mask] == target_id)

				if len(expandPoints)==0: #no neighbors (early termination)
					stop_growing('noneighbor')
					break 

				subset = numpy.random.choice(len(currentPoints), NUM_POINT, replace=len(currentPoints)<NUM_POINT)
				center = numpy.mean(currentPoints, axis=0)
				expandPoints = numpy.array(expandPoints)
				expandPoints[:,:2] -= center[:2]
				expandPoints[:,3:] -= center[3:]
				input_points[0,:,:] = currentPoints[subset, :]
				input_points[0,:,:2] -= center[:2]
				if len(expandPoints) >= NUM_NEIGHBOR_POINT:
					subset = numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = range(len(expandPoints)) + list(numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
				neighbor_points[0,:,:] = numpy.array(expandPoints)[subset, :]
				input_classes[0,:] = numpy.array(expandClass)[subset]
				input_complete = numpy.zeros(1,dtype=numpy.int32)
				ls, cls, cls_acc, cmpl, cmpl_acc = sess.run([net.loss, net.class_output, net.class_acc, net.completeness_output, net.completeness_acc],
					{net.input_pl:input_points, net.neighbor_pl:neighbor_points, net.completeness_pl:input_complete, net.class_pl:input_classes})

				cls_conf = scipy.special.softmax(cls[0], axis=-1)[:,1]
				cls_mask = cls_conf > classification_threshold
#				cls_mask = input_classes[0].astype(bool)
				cmpl_conf = scipy.special.softmax(cmpl[0], axis=-1)[1]
				validPoints = neighbor_points[0,:,:][cls_mask]
				validPoints[:,:2] += center[:2]
				validVoxels = numpy.round(validPoints[:,:3]/resolution).astype(int)
				expandSet = set([tuple(p) for p in validVoxels])
				updated = False
				for i in range(len(point_voxels)):
					if tuple(point_voxels[i]) in expandSet and not currentMask[i]:
						currentMask[i] = True
						updated = True

				if cmpl_conf > completion_threshold:
					stop_growing('')
					break 
				else:
					if updated: #continue growing
						minDims = point_voxels[currentMask, :].min(axis=0)
						maxDims = point_voxels[currentMask, :].max(axis=0)
					else: #no matching neighbors (early termination)
						stop_growing('noexpand')
						break 
				steps += 1

		#fill in points with no labels
		for i in numpy.nonzero(cluster_label==0)[0]:
			k = tuple(numpy.round(points[i,:3]/resolution).astype(int))
			d = 1
			while cluster_label[i]==0:
				neighbors = []
				for offset in itertools.product(range(-d,d+1),range(-d,d+1),range(-d,d+1)):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in equalized_map:
						neighbors.append(equalized_map[kk])
				for n in neighbors:
					if cluster_label[n] > 0:
						cluster_label[i] = cluster_label[n]
						break
				d += 1

#		#find connected edges on a voxel grid
#		voxel_map = {}
#		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
#		for i in range(len(point_voxels)):
#			voxel_map[tuple(point_voxels[i])] = i
#		edges = []
#		for i in range(len(point_voxels)):
#			k = tuple(point_voxels[i])
#			for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
#				if offset!=(0,0,0):
#					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
#					if kk in voxel_map and embeddings[voxel_map[kk]].dot(embeddings[i]) > 0.9:
#						edges.append([i, voxel_map[kk]])
#
#		#calculate connected components from edges
#		G = nx.Graph(edges)
#		clusters = nx.connected_components(G)
#		clusters = [list(c) for c in clusters]
#		cluster_label = numpy.zeros(len(point_voxels),dtype=int)
#		min_cluster_size = 10
#		cluster_id = 1
#		for i in range(len(clusters)):
#			if len(clusters[i]) > min_cluster_size:
#				cluster_label[clusters[i]] = cluster_id
#				cluster_id += 1
	
		#calculate statistics 
		gt_match = 0
		match_id = 0
		dt_match = numpy.zeros(cluster_label.max(), dtype=bool)
		cluster_label2 = numpy.zeros(len(cluster_label), dtype=int)
		room_iou = []
		for i in set(obj_id):
			best_iou = 0
			for j in range(1, cluster_label.max()+1):
				if not dt_match[j-1]:
					iou = 1.0 * numpy.sum(numpy.logical_and(obj_id==i, cluster_label==j)) / numpy.sum(numpy.logical_or(obj_id==i, cluster_label==j))
					best_iou = max(best_iou, iou)
					if iou > 0.5:
						dt_match[j-1] = True
						gt_match += 1
						cluster_label2[cluster_label==j] = i
						break
			room_iou.append(best_iou)
		for j in range(1,cluster_label.max()+1):
			if not dt_match[j-1]:
				cluster_label2[cluster_label==j] = j + obj_id.max()
		prc = numpy.mean(dt_match)
		rcl = 1.0 * gt_match / len(set(obj_id))
		room_iou = numpy.mean(room_iou)

		nmi = normalized_mutual_info_score(obj_id,cluster_label)
		ami = adjusted_mutual_info_score(obj_id,cluster_label)
		ars = adjusted_rand_score(obj_id,cluster_label)
		agg_nmi.append(nmi)
		agg_ami.append(ami)
		agg_ars.append(ars)
		agg_prc.append(prc)
		agg_rcl.append(rcl)
		agg_iou.append(room_iou)
		print("Area %s room %d NMI: %.2f AMI: %.2f ARS: %.2f PRC: %.2f RCL: %.2f IOU: %.2f"%(str(AREA), room_id, nmi,ami,ars, prc, rcl, room_iou))

		#save point cloud results to file
		if save_results:
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
			savePLY('data/results/%d.ply'%save_id, unequalized_points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))

