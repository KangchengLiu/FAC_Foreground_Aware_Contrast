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
    #if args.pc_augm_rot:
    if args['pc_augm_rot']:
        ref_point = xyz[np.random.randint(xyz.shape[0]),:3]
        ref_point[2] = 0
        M = transforms3d.axangles.axangle2mat([0,0,1],np.random.uniform(0,2*math.pi)).astype('f4')
        xyz = np.matmul(xyz[:,:3]-ref_point, M)+ref_point
    #if args.pc_augm_jitter: #add jitter
    if args['pc_augm_jitter']: #add jitter
        sigma, clip= 0.002, 0.005 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        xyz = xyz + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32)
        #if args.use_rgb:
        if args['use_rgb']:
            rgb = np.clip(rgb + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32),-1,1)
    return xyz, rgb

def read_structure(file_name, read_geof):
    """
    read the input point cloud in a format ready for embedding    
    """
    #print(file_name)
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

    if len(labels.shape) == 0:#dirty fix
        labels = np.array([0])
    if len(is_transition.shape) == 0:#dirty fix
        is_transition = np.array([0])
    if read_geof: #geometry = geometric features
        local_geometry = np.array(data_file['geof'], dtype='float32')
    else: #geometry = neighborhood structure
        local_geometry = np.array(data_file['target_local_geometry'], dtype='uint32')
    
    #voxel_nei = np.array(data_file['voxel_nei'], dtype='uint32')
    #dispersed_voxel = np.array(data_file['dispersed_voxel'], dtype='uint32')
    #dispersed_num = np.array(data_file['dispersed_num'], dtype='uint32')
    data_file.close()
    #return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn,voxel_nei,dispersed_voxel,dispersed_num
    return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn
    #return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn,selected_ver,selected_edg,center

def create_scannet_datasets(args, logger, test_seed_offset=0):
    """ Gets training and test datasets. """
    # Load formatted clouds
    testlist, trainlist = [], []
    #for n in range(1,7):
    #    if n != args['test_area']: 
    #        # path = '{}/Area_{:d}/'.format(args['data_root'], n)
    #        path = os.path.join(args['data_root'], 'Area_{:d}/'.format(n))
    #        for fname in sorted(os.listdir(path), key=lambda x:os.stat(path + "/" + x).st_size):
    #            if fname.endswith(".h5"):
    #                trainlist.append(path+fname)
    if args['num_point'] == 14999:
        path = os.path.join(args['data_root'], 'Train_14999/')
        # logger.info("149999")
        # exit()
    else:
        path = os.path.join(args['data_root'], 'Train/')
    for fname in sorted(os.listdir(path), key=lambda x:os.stat(path + "/" + x).st_size):
        if fname.endswith(".h5"):
            trainlist.append(path+fname)
    
    print('train list: {}'.format(len(trainlist)))
    
    # path = os.path.join(args['data_root'], 'Test_2/')
    path = os.path.join(args['data_root'], 'Test/')
    for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size):
        if fname.endswith(".h5"):
            testlist.append(path+fname)
    print('test list: {}'.format(len(testlist)))
    return tnt.dataset.ListDataset(trainlist,
                                   functools.partial(graph_loader, train=True, args=args, logger=logger, db_path=args['data_root'])),\
            tnt.dataset.ListDataset(testlist,
                                   functools.partial(graph_loader, train=False, args=args, logger=logger, db_path=args['data_root']))


def graph_loader(entry, train, args, logger, db_path, test_seed_offset=0, full_cpu = False):
    """ Load the point cloud and the graph structure """
    #xyz, rgb, edg_source, edg_target, is_transition, local_geometry, \
    #        labels, objects, elevation, xyn,voxel_nei,dispersed_voxel, \
    #        dispersed_num = read_structure(entry, 'geof' in args['ver_value'])
            #dispersed_num = read_structure(entry, 'geof' in args.ver_value)
    xyz, rgb, edg_source, edg_target, is_transition, local_geometry,\
            labels, objects, elevation, xyn = read_structure(entry, 'geof' in args['ver_value'])
    
    short_name= entry.split(os.sep)[-2]+'_'+entry.split(os.sep)[-1]
    #logger.info('{}'.format(short_name))
    raw_xyz = np.array(xyz)
    raw_rgb = np.array(rgb)
    raw_labels = np.array(labels)
    rgb = rgb/255

    n_ver = np.shape(xyz)[0]
    n_edg = np.shape(edg_source)[0]
    #print(n_ver)
    selected_ver = np.full((n_ver,), True, dtype='?')
    selected_edg = np.full((n_edg,), True, dtype='?')
    # print(raw_xyz.shape)
    # print(raw_rgb.shape)
    # print(raw_labels.shape)
    if train:
        xyz, rgb = augment_cloud_whole(args, xyz, rgb)

    subsample = False
    new_ver_index = []

    #logger.info('name {} train: {} num_point: {} n_ver: {}'.format(short_name, train, args['num_point'], n_ver))
    #if train and (0<args.max_ver_train<n_ver):
    if train and (0 < args['num_point'] < n_ver):
        subsample = True
        selected_edg, selected_ver = libply_c.random_subgraph(n_ver, 
                                                              edg_source.astype('uint32'),
                                                              edg_target.astype('uint32'),
                                                              int(args['num_point']))
                                                              #int(args.max_ver_train))
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

    # if train and (0<args.max_ver_train<n_ver):
    # if train==False and (0 < args['num_point'] < n_ver):
    #     subsample = True
    #     selected_edg, selected_ver = libply_c.random_subgraph(n_ver, 
    #                                                           edg_source.astype('uint32'),
    #                                                           edg_target.astype('uint32'),
    #                                                           int(args['num_point']*4))
    #                                                           #int(args.max_ver_train))
    #     selected_edg = selected_edg.astype('?')
    #     selected_ver = selected_ver.astype('?')

    #     new_ver_index = -np.ones((n_ver,), dtype = int)
    #     new_ver_index[selected_ver.nonzero()] = range(selected_ver.sum())

    #     edg_source = new_ver_index[edg_source[selected_edg.astype('?')]]
    #     edg_target = new_ver_index[edg_target[selected_edg.astype('?')]]

    #     is_transition = is_transition[selected_edg]
    #     labels = raw_labels[selected_ver,]
    #     objects = objects[selected_ver,]
    #     elevation = elevation[selected_ver]
    #     xyn = xyn[selected_ver,]

    #if args.learned_embeddings:
    if args['learned_embeddings']:
        #we use point nets to embed the point clouds
        #nei = local_geometry[selected_ver,:args.k_nn_local].astype('int64')
        nei = local_geometry[selected_ver, :args['k_nn_local']].astype('int64')
        
        clouds, clouds_global = [], []
        #clouds_global is cloud global features. here, just the diameter + elevation

        clouds = xyz[nei,]
        #diameters = np.max(np.max(clouds,axis=1) - np.min(clouds,axis=1), axis = 1)
        diameters = np.sqrt(clouds.var(1).sum(1))
        clouds = (clouds - xyz[selected_ver,np.newaxis,:]) / (diameters[:,np.newaxis,np.newaxis] + 1e-10)

        #if args.use_rgb:
        if args['use_rgb']:
            clouds = np.concatenate([clouds, rgb[nei,]],axis=2)

        clouds = clouds.transpose([0,2,1])

        clouds_global = diameters[:,None]
        #if 'e' in args.global_feat:
        if 'e' in args['global_feat']:
            #print('a')
            clouds_global = np.hstack((clouds_global, elevation[:,None]))
        if 'rgb' in args['global_feat']:
            #print('b')
            clouds_global = np.hstack((clouds_global, rgb[selected_ver,]))
        if 'XY' in args['global_feat']:
            #print('c')
            clouds_global = np.hstack((clouds_global, xyn))
        if 'xy' in args['global_feat']:
            #print('d')
            clouds_global = np.hstack((clouds_global, xyz[selected_ver,:2]))
        #clouds_global = np.hstack((diameters[:,None], ((xyz[selected_ver,2] - min_z) / (max_z- min_z)-0.5)[:,None],np.zeros_like(rgb[selected_ver,])))


    is_transition = torch.from_numpy(is_transition)
    labels = torch.from_numpy(labels)
    objects = torch.from_numpy(objects.astype('int64'))
    clouds = torch.from_numpy(clouds)
    clouds_global = torch.from_numpy(clouds_global)
    # logger.info('labels: {}'.format(labels.size()))
    # logger.info('clouds: {}'.format(clouds.size()))
    xyz = xyz[selected_ver,]
    xyz = torch.from_numpy(xyz)
    # logger.info('name: {} raw_xyz: {} xyz: {} clouds_global: {}'.format(short_name, raw_xyz.shape, xyz.size(), clouds_global.size()))
    # logger.info('xyz: {}'.format(xyz.size()))
    # logger.info('{}'.format('\n'))

    del raw_labels
    del raw_rgb
    del raw_xyz
    del nei
    
    return short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz

    #return short_name, edg_source, edg_target, is_transition, labels,objects, clouds, clouds_global, xyz,voxel_nei,dispersed_voxel,dispersed_num
    # 
    # return short_name, edg_source, edg_target, is_transition, labels, objects

def my_collate_scannet(batch):

    short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz = list(zip(*batch))

    xyz = [t.unsqueeze(0) for t in xyz]
    clouds_global = [t.unsqueeze(0) for t in clouds_global]
    # clouds_global = []
    # for t in clouds_global:
    #     t = t.unsqueeze(0)
    #     if logger is not None:
    #         logger.info('t:', t.size())
    #     clouds_global.append(t)

    labels = [t.unsqueeze(0) for t in labels]
    #is_transition = [t.unsqueeze(0) for t in is_transition]
    
    clouds = torch.cat(clouds, 0)
    clouds_global = torch.cat(clouds_global, 0)
    xyz = torch.cat(xyz, 0)
    
    #is_transition = torch.cat(is_transition, 0)
    
    labels = torch.cat(labels, 0)
    objects = torch.cat(objects, 0)
    #print('-------->', len(xyz))
    return short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, xyz
    #return short_name, edg_source, edg_target, is_transition, labels, objects, (clouds, clouds_global), xyz,voxel_nei,dispersed_voxel,dispersed_num
