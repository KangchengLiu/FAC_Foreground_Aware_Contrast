import os
import sys
import glob
import numpy as np
import h5py
import random
from random import choice
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import argparse
from timeit import default_timer as timer
import torchnet as tnt
import functools
import argparse
import transforms3d
from sklearn.linear_model import RANSACRegressor
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from provider import *
from lib.ply_c import libply_c


def augment_cloud_whole(args, xyz, rgb):
    """" rotate the whole graph, add jitter """
    if args['pc_augm_rot']:
        ref_point = xyz[np.random.randint(xyz.shape[0]),:3]
        ref_point[2] = 0
        M = transforms3d.axangles.axangle2mat([0,0,1],np.random.uniform(0,2*math.pi)).astype('f4')
        xyz = np.matmul(xyz[:,:3]-ref_point, M)+ref_point
    if args['pc_augm_jitter']:
        sigma, clip= 0.002, 0.005 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        xyz = xyz + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32)
        if args['use_rgb']:
            rgb = np.clip(rgb + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32),-1,1)
    return xyz, rgb

def read_structure(file_name, read_geof):
    """
    read the input point cloud in a format ready for embedding    
    """
    data_file = h5py.File(file_name, 'r')

    xyz = np.array(data_file['xyz'], dtype='float32')
    rgb = np.array(data_file['rgb'], dtype='float32')
    elevation = np.array(data_file['elevation'], dtype='float32')
    xyn = np.array(data_file['xyn'], dtype='float32')
    edg_source = np.array(data_file['source'], dtype='int').squeeze()
    edg_target = np.array(data_file['target'], dtype='int').squeeze()
    is_transition = np.array(data_file['is_transition'])
    objects = np.array(data_file['objects'][()])
    labels = np.array(data_file['labels']).squeeze()

    if len(labels.shape) == 0:
        labels = np.array([0])
    if len(is_transition.shape) == 0:
        is_transition = np.array([0])
    if read_geof:
        local_geometry = np.array(data_file['geof'], dtype='float32')
    else:
        local_geometry = np.array(data_file['target_local_geometry'], dtype='uint32')
	
    geof = np.array(data_file['geof'], dtype='float32')

    data_file.close()
    return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, geof, labels, objects, elevation, xyn

def create_s3dis_datasets(args, logger, test_seed_offset=0 ):
    """ Gets training and test datasets. """
    # Load formatted clouds
    testlist, trainlist = [], []
    for n in range(1,7):
        if n != args['test_area']: 
            path = os.path.join(args['data_root'], 'Area_{:d}/'.format(n))
            for fname in sorted(os.listdir(path), key=lambda x:os.stat(path + "/" + x).st_size):
                if fname.endswith(".h5"):
                    trainlist.append(path+fname)
    
    print('train list: {}'.format(len(trainlist)))
    path = os.path.join(args['data_root'], 'Area_{:d}_test/'.format(args['test_area']))
    
    # for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size):
    for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size, reverse=True):
        if fname.endswith(".h5"):
            testlist.append(path+fname)
    print('test list: {}'.format(testlist))
    print('test list: {}'.format(len(testlist)))
    return tnt.dataset.ListDataset(trainlist,
                                   functools.partial(graph_loader, train=True, args=args, logger=logger, db_path=args['data_root'])),\
            tnt.dataset.ListDataset(testlist,
                                   functools.partial(graph_loader, train=False, args=args, logger=logger, db_path=args['data_root']))


def graph_loader(entry, train, args, logger, db_path, test_seed_offset=0, full_cpu = False):
    """ Load the point cloud and the graph structure """
    xyz, rgb, edg_source, edg_target, is_transition, local_geometry, geof, \
            labels, objects, elevation, xyn = read_structure(entry, False)
    
    short_name= entry.split(os.sep)[-2]+'_'+entry.split(os.sep)[-1]
    raw_xyz=np.array(xyz)
    raw_rgb=np.array(rgb)
    raw_labels=np.array(labels)
    rgb = rgb/255

    n_ver = np.shape(xyz)[0]
    n_edg = np.shape(edg_source)[0]
    selected_ver = np.full((n_ver,), True, dtype='?')
    selected_edg = np.full((n_edg,), True, dtype='?')

    if train:
        xyz, rgb = augment_cloud_whole(args, xyz, rgb)

    subsample = False
    new_ver_index = []

    if train and (0 < args['num_point'] < n_ver):
        subsample = True
        selected_edg, selected_ver = libply_c.random_subgraph(n_ver, 
                                                              edg_source.astype('uint32'),
                                                              edg_target.astype('uint32'),
                                                              int(args['num_point']))
        selected_edg = selected_edg.astype('?')
        selected_ver = selected_ver.astype('?')

        new_ver_index = -np.ones((n_ver,), dtype = int)
        new_ver_index[selected_ver.nonzero()] = range(selected_ver.sum())

        edg_source = new_ver_index[edg_source[selected_edg.astype('?')]]
        edg_target = new_ver_index[edg_target[selected_edg.astype('?')]]

        is_transition = is_transition[selected_edg]
        labels = raw_labels[selected_ver,]
        objects = objects[selected_ver,]
        elevation = elevation[selected_ver]
        xyn = xyn[selected_ver,]

    if args['learned_embeddings']:
        # we use point nets to embed the point clouds
        # local_geometry: N x 20  index
        nei = local_geometry[selected_ver, :args['k_nn_local']].astype('int64')
        
        clouds, clouds_global = [], []
        #clouds_global is cloud global features. here, just the diameter + elevation

        clouds = xyz[nei,]
        #diameters = np.max(np.max(clouds,axis=1) - np.min(clouds,axis=1), axis = 1)
        diameters = np.sqrt(clouds.var(1).sum(1))
        clouds = (clouds - xyz[selected_ver,np.newaxis,:]) / (diameters[:,np.newaxis,np.newaxis] + 1e-10)

        if args['use_rgb']:
            clouds = np.concatenate([clouds, rgb[nei,]],axis=2)

        if args['ver_value'] == 'geof':                             # N x 4
            clouds = np.concatenate([clouds, geof[nei,]],axis=2)    # n x 20 x (xyz+rgb+geof) = n x 20 x 10

        clouds = clouds.transpose([0,2,1])

        clouds_global = diameters[:,None]
        if 'e' in args['global_feat']:
            clouds_global = np.hstack((clouds_global, elevation[:,None]))
        if 'rgb' in args['global_feat']:
            clouds_global = np.hstack((clouds_global, rgb[selected_ver,]))
        if 'XY' in args['global_feat']:
            clouds_global = np.hstack((clouds_global, xyn))
        if 'xy' in args['global_feat']:
            clouds_global = np.hstack((clouds_global, xyz[selected_ver,:2]))
        if 'o' in args['global_feat']:
            clouds_global = np.hstack((clouds_global, geof[selected_ver,]))

    xyz = xyz[selected_ver,]

    is_transition = torch.from_numpy(is_transition)
    labels = torch.from_numpy(labels)
    objects = torch.from_numpy(objects.astype('int64'))
    clouds = torch.from_numpy(clouds)
    clouds_global = torch.from_numpy(clouds_global)

    xyz = torch.from_numpy(xyz)
    del raw_labels
    del raw_rgb
    del raw_xyz
    del nei

    return short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz

def my_collate(batch):
    short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz = list(zip(*batch))

    xyz = [t.unsqueeze(0) for t in xyz]
    clouds_global = [t.unsqueeze(0) for t in clouds_global]
    labels = [t.unsqueeze(0) for t in labels]
    
    clouds = torch.cat(clouds, 0)
    clouds_global = torch.cat(clouds_global, 0)
    xyz = torch.cat(xyz, 0)
        
    labels = torch.cat(labels, 0)
    objects = torch.cat(objects, 0)

    return short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz
