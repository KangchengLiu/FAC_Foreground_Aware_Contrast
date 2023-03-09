import os
import sys
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as initer
import torch.nn.functional as F

from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

def knn(x, k=20):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def pairwise_distance(x):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pdist = -xx - inner - xx.transpose(2, 1)
    pdist_softmax = torch.nn.functional.softmax(pdist, dim=2)
    return pdist_softmax

def pairwise_distance_mask(x, k=20):
    # print(x.size())
    bs, ch, nump = x.size()
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pdist = -xx - inner - xx.transpose(2, 1)
    
    topk, indices = pdist.topk(k=k, dim=-1)

    res = torch.autograd.Variable(torch.zeros(bs, nump, nump)).cuda()
    res = res.scatter(2, indices, topk)

    pdist_softmax = torch.nn.functional.softmax(pdist, dim=2)
    return pdist_softmax

def pairwise_distance_mask1(x, k=20):
    # print(x.size())
    bs, ch, nump = x.size()
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pdist = -xx - inner - xx.transpose(2, 1)
    
    topk, indices = pdist.topk(k=k, dim=-2)

    res = torch.autograd.Variable(torch.ones(bs, nump, nump)).cuda()
    res = res.scatter(1, indices, topk)

    res = res < 0.00001
    res = res.float()
    # pdist_softmax = torch.nn.functional.softmax(pdist, dim=2)
    return res

def pairwise_distance_mask1_dilate(x, k=20):
    # print(x.size())
    bs, ch, nump = x.size()
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pdist = -xx - inner - xx.transpose(2, 1)
    
    ek = k + k
    topk, indices = pdist.topk(k=ek, dim=-2)  # indices: BxekXN

    idx_ek = np.array([i for i in range(ek)])
    np.random.shuffle(idx_ek)
    idx_k = idx_ek[:k] 
    indices = indices[:, idx_k, :]

    res = torch.autograd.Variable(torch.ones(bs, nump, nump)).cuda()
    res = res.scatter(1, indices, topk)

    res = res < 0.00001
    res = res.float()
    # pdist_softmax = torch.nn.functional.softmax(pdist, dim=2)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def convert_to_syncbn(model):
    def recursive_set(cur_module, name, module):
        if len(name.split('.')) > 1:
            recursive_set(getattr(cur_module, name[:name.find('.')]), name[name.find('.')+1:], module)
        else:
            setattr(cur_module, name, module)
    from lib.sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            recursive_set(model, name, SynchronizedBatchNorm1d(m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm2d):
            recursive_set(model, name, SynchronizedBatchNorm2d(m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm3d):
            recursive_set(model, name, SynchronizedBatchNorm3d(m.num_features, m.eps, m.momentum, m.affine))


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

# ------------------------------ superpoint metrics ------------------------------
def get_components(init_center,pt_center_index,asso, getmax=False, trick=False, logger=None):
    #print(asso[0])
    # init_center=init_center.cpu().numpy()
    # pt_center_index=pt_center_index.cpu().numpy()
    # asso=asso.argmax(axis=1).cpu().numpy().reshape(-1,1)
    # init_center=init_center.cpu().numpy()
    # pt_center_index=pt_center_index.cpu().numpy()
    #asso=asso.argmax(axis=1).reshape(-1,1)
    if getmax:
        if trick == False:
            asso=asso.argmax(axis=1).reshape(-1,1)  # n x 1
        else:
            # print(asso.shape)
            mea = np.mean(asso, axis=1)     # n
            # print(mea.shape)
            tmp = np.zeros((mea.shape[0], 1), dtype=np.int32)
            for i in range(asso.shape[0]):
                for j in range(asso.shape[1]):
                    if asso[i, j]*3.0 > mea[i]:
                        tmp[i, 0] = j
                        break
            asso = tmp

    else:
        asso=asso.argmin(axis=1).reshape(-1,1)
    components=[]

    
    in_component=np.take_along_axis(pt_center_index,asso,1).reshape(-1)
    #sp_index=init_center[in_components].reshape(-1)
    in_component=init_center[in_component]
    coms=np.unique(in_component)
    #   #print(len(init_center))
    # print(len(coms))
    # idx=np.isin(init_center, coms, invert=False)
    # coms=init_center[idx]
    #print(init_center)
    real_center=[]
    for i in range(len(coms)):
        #te=[]
        te=np.where(in_component==coms[i])
        in_component[te] = i
        components.append(te)
        
    # real_center=np.array(real_center,dtype=np.int64)
    # real_center=torch.from_numpy(real_center).cuda()
    # print(len(components))
    #logger.info('len components: {}'.format(len(components)))
    #logger.info('len in_component: {}'.format(len(in_component)))
    return components,in_component

def get_components_fixasso(init_center,pt_center_index,asso):
    #print(asso[0])
    # init_center=init_center.cpu().numpy()
    # pt_center_index=pt_center_index.cpu().numpy()
    # asso=asso.argmax(axis=1).cpu().numpy().reshape(-1,1)
    # init_center=init_center.cpu().numpy()
    # pt_center_index=pt_center_index.cpu().numpy()
    #asso=asso.argmax(axis=1).reshape(-1,1)
    # asso=asso.argmin(axis=1).reshape(-1,1)
    components=[]

    
    in_component=np.take_along_axis(pt_center_index,asso,1).reshape(-1)
    #sp_index=init_center[in_components].reshape(-1)
    in_component=init_center[in_component]
    coms=np.unique(in_component)
#   #print(len(init_center))
#   #print(len(coms))
    # idx=np.isin(init_center, coms, invert=False)
    # coms=init_center[idx]
    #print(init_center)
    real_center=[]
    for i in range(len(coms)):
        #te=[]
        te=np.where(in_component==coms[i])

        components.append(te)
        
    # real_center=np.array(real_center,dtype=np.int64)
    # real_center=torch.from_numpy(real_center).cuda()
   # print(len(components))
    return components,in_component

def perfect_prediction(components,in_component,labels):
    """assign each superpoint with the majority label"""
    #print(pred.shape)
    #print(labels.shape)
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')

    for i_com in range(len(components)):
        #print(len(components[i_com]))
        #te=labels[components[i_com]]
        #print(te.shape)
        #label_com = np.argmax(np.bincount(labels[components[i_com]]))
        label_com = labels[components[i_com],1:].sum(0).argmax()
        full_pred[components[i_com]]=label_com


    return full_pred

def perfect_prediction_singlelabel(components,in_component,labels,logger=None):
    """assign each superpoint with the majority label"""
    #print(pred.shape)
    #print(labels.shape)
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')

    for i_com in range(len(components)):
        #print(len(components[i_com]))
        #te=labels[components[i_com]]
        #print(te.shape)
        #label_com = np.argmax(np.bincount(labels[components[i_com]]))
        #label_com = labels[components[i_com], 1:].sum(0).argmax()
        logger.info('components[i_com]: {}'.format(components[i_com].shape))
        exit()
        label_com = labels[components[i_com], 1:].sum(0).argmax()
        full_pred[components[i_com]]=label_com


    return full_pred

def perfect_prediction_base0(components,in_component,labels):
    """assign each superpoint with the majority label"""
    #print(pred.shape)
    #print(labels.shape)
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')

    for i_com in range(len(components)):
        #print(len(components[i_com]))
        #te=labels[components[i_com]]
        #print(te.shape)
        #label_com = np.argmax(np.bincount(labels[components[i_com]]))
        label_com = labels[components[i_com],:].sum(0).argmax()
        full_pred[components[i_com]]=label_com


    return full_pred

def perfect_prediction_num(components,in_component,labels):
    """assign each superpoint with the majority label"""
    #print(pred.shape)
    #print(labels.shape)
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')

    for i_com in range(len(components)):
        #print(len(components[i_com]))
        #te=labels[components[i_com]]
        #print(te.shape)
        label_com = np.argmax(np.bincount(labels[components[i_com]]))
        #label_com = labels[components[i_com],1:].sum(0).argmax()
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

def compute_boundary_recall(is_transition, pred_transitions):
    return 100.*((is_transition==pred_transitions)*is_transition).sum()/is_transition.sum()

def compute_boundary_precision(is_transition, pred_transitions):
    return 100.*((is_transition==pred_transitions)*pred_transitions).sum()/pred_transitions.sum()


# ------------------------------- save spg -------------------------------------
import h5py
def write_spg(file_name, graph_sp, components, in_component):
    """save the partition and spg information"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    grp = data_file.create_group('components')
    n_com = len(components)
    for i_com in range(0, n_com):
        grp.create_dataset(str(i_com), data=components[i_com], dtype='uint32')
    data_file.create_dataset('in_component'
                             , data=in_component, dtype='uint32')
    data_file.create_dataset('sp_labels'
                             , data=graph_sp["sp_labels"], dtype='uint32')
    data_file.create_dataset('sp_centroids'
                             , data=graph_sp["sp_centroids"], dtype='float32')
    data_file.create_dataset('sp_length'
                             , data=graph_sp["sp_length"], dtype='float32')
    data_file.create_dataset('sp_surface'
                             , data=graph_sp["sp_surface"], dtype='float32')
    data_file.create_dataset('sp_volume'
                             , data=graph_sp["sp_volume"], dtype='float32')
    data_file.create_dataset('sp_point_count'
                             , data=graph_sp["sp_point_count"], dtype='uint64')
    data_file.create_dataset('source'
                             , data=graph_sp["source"], dtype='uint32')
    data_file.create_dataset('target'
                             , data=graph_sp["target"], dtype='uint32')
    data_file.create_dataset('se_delta_mean'
                             , data=graph_sp["se_delta_mean"], dtype='float32')
    data_file.create_dataset('se_delta_std'
                             , data=graph_sp["se_delta_std"], dtype='float32')
    data_file.create_dataset('se_delta_norm'
                             , data=graph_sp["se_delta_norm"], dtype='float32')
    data_file.create_dataset('se_delta_centroid'
                             , data=graph_sp["se_delta_centroid"], dtype='float32')
    data_file.create_dataset('se_length_ratio'
                             , data=graph_sp["se_length_ratio"], dtype='float32')
    data_file.create_dataset('se_surface_ratio'
                             , data=graph_sp["se_surface_ratio"], dtype='float32')
    data_file.create_dataset('se_volume_ratio'
                             , data=graph_sp["se_volume_ratio"], dtype='float32')
    data_file.create_dataset('se_point_count_ratio'
                             , data=graph_sp["se_point_count_ratio"], dtype='float32')

# ------------------------------- read spg -------------------------------------
def read_spg(file_name):
    """read the partition and spg information"""
    data_file = h5py.File(file_name, 'r')
    graph = dict([("is_nn", False)])
    graph["source"] = np.array(data_file["source"], dtype='uint32')
    graph["target"] = np.array(data_file["target"], dtype='uint32')
    graph["sp_centroids"] = np.array(data_file["sp_centroids"], dtype='float32')
    graph["sp_length"] = np.array(data_file["sp_length"], dtype='float32')
    graph["sp_surface"] = np.array(data_file["sp_surface"], dtype='float32')
    graph["sp_volume"] = np.array(data_file["sp_volume"], dtype='float32')
    graph["sp_point_count"] = np.array(data_file["sp_point_count"], dtype='uint64')
    graph["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype='float32')
    graph["se_delta_std"] = np.array(data_file["se_delta_std"], dtype='float32')
    graph["se_delta_norm"] = np.array(data_file["se_delta_norm"], dtype='float32')
    graph["se_delta_centroid"] = np.array(data_file["se_delta_centroid"], dtype='float32')
    graph["se_length_ratio"] = np.array(data_file["se_length_ratio"], dtype='float32')
    graph["se_surface_ratio"] = np.array(data_file["se_surface_ratio"], dtype='float32')
    graph["se_volume_ratio"] = np.array(data_file["se_volume_ratio"], dtype='float32')
    graph["se_point_count_ratio"] = np.array(data_file["se_point_count_ratio"], dtype='float32')
    in_component = np.array(data_file["in_component"], dtype='uint32')
    n_com = len(graph["sp_length"])
    graph["sp_labels"] = np.array(data_file["sp_labels"], dtype='uint32')
    grp = data_file['components']
    components = np.empty((n_com,), dtype=object)
    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()
    return graph, components, in_component



# ----------------------------- vis segmentation -----------------
def get_color_from_label(object_label, dataset):
    """associate the color corresponding to the class"""
    if dataset == 's3dis': #S3DIS
        object_label = {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [0,255,0], #'ceiling' .-> .yellow
            2: [0,0,255], #'floor' .-> . blue
            3: [0,255,255], #'wall'  ->  brown
            4: [255,0,255], #'column'  ->  bluegreen
            5: [255,255,0], #'beam'  ->  salmon
            6: [100,100,255], #'window'  ->  bright green
            7: [200,200,100], #'door'   ->  dark green
            8: [170,120,200], #'table'  ->  dark grey
            9: [255,0,0], #'chair'  ->  darkblue
            10:[10,200,100], #'bookcase'  ->  red
            11:[200,100,100], #'sofa'  ->  purple
            12: [200,200,200], #'board'   ->  grey
            13: [50,50,50], #'clutter'  ->  light grey
        }.get(object_label, -1)
    elif dataset == 'vkitti':
        object_label = {
            0: [200, 90, 0],    # Terrian .->.brown
            1: [0, 128, 50],    # Tree .-> .dark green
            2: [0, 220, 0],     # Vegetation .-> . bright green
            3: [255, 0, 0],     # Building  ->  red
            4: [100, 100, 100], # road  ->  dark gray
            5: [200, 200, 200], # GuardRail  ->  bright gray
            6: [255, 0, 255],   # TrafficSign  ->  bright gray
            7: [255, 255, 0],   # TrafficLight  -> yellow
            8: [128, 0, 255],   # Pole  ->  violet
            9: [255, 200, 150], # Misc  ->  skin
            10:[0, 128, 255],   # Truck  ->  dark blue
            11:[0, 200, 255],   # Car  ->  bright blue
            12: [255, 128, 0],  # Van   ->  orange
            13: [0, 0, 0],      # don't care -> black
        }.get(object_label, -1)
    elif dataset == 'scannet':
        object_label = {
            0: [0, 0, 0],           # ('unlabeled', (0, 0, 0)),
            1: [174, 199, 232],     # ('wall', (174, 199, 232)),
            2: [152, 223, 138],     # ('floor', (152, 223, 138)),
            3: [31, 119, 180],      # ('cabinet', (31, 119, 180)),
            4: [255, 187, 120],     # ('bed', (255, 187, 120)),
            5: [188, 189, 34],      # ('chair', (188, 189, 34)),
            6: [140, 86, 75],       # ('sofa', (140, 86, 75)),
            7: [255, 152, 150],     # ('table', (255, 152, 150)),
            8: [214, 39, 40],       # ('door', (214, 39, 40)),
            9: [197, 176, 213],     # ('window', (197, 176, 213)),
            10: [148, 103, 189],    # ('bookshelf', (148, 103, 189)),
            11: [196, 156, 148],    # ('picture', (196, 156, 148)),
            12: [23, 190, 207],     # ('counter', (23, 190, 207)),
            13: [247, 182, 210],    # ('desk', (247, 182, 210)),
            14: [219, 219, 141],    # ('curtain', (219, 219, 141)),
            15: [255, 127, 14],     # ('refrigerator', (255, 127, 14)),
            16: [158, 218, 229],    # ('showercurtain', (158, 218, 229)),
            17: [44, 160, 44],      # ('toilet', (44, 160, 44)),
            18: [112, 128, 144],    # ('sink', (112, 128, 144)),
            19: [227, 119, 194],    # ('bathtub', (227, 119, 194)),
            20: [82, 84, 163],      # ('otherfurniture', (82, 84, 163)),
        }.get(object_label, -1)
    # g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']
    elif (dataset == 'sema3d'): #Semantic3D
        object_label =  {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 200, 200, 200], #'man-made terrain'  ->  grey
            2: [   0,  70,   0], #'natural terrain'  ->  dark green
            3: [   0, 255,   0], #'high vegetation'  ->  bright green
            4: [ 255, 255,   0], #'low vegetation'  ->  yellow
            5: [ 255,   0,   0], #'building'  ->  red
            6: [ 148,   0, 211], #'hard scape'  ->  violet
            7: [   0, 255, 255], #'artifact'   ->  cyan
            8: [ 255,   8, 127], #'cars'  ->  pink
        }.get(object_label, -1)
    elif (dataset == 'custom_dataset'): #Custom set
        object_label =  {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 255, 0, 0], #'classe A' -> red
            2: [ 0, 255, 0], #'classeB' -> green
        }.get(object_label, -1)
    else: 
        raise ValueError('Unknown dataset: %s' % (dataset))
    if object_label == -1:
        raise ValueError('Type not recognized: %s' % (object_label))
    return object_label

#prediction2ply(ply_file_name_pred, xyz, pred, 13, 's3dis')
def prediction2ply_seg(filename, xyz, prediction, n_label, dataset):
    """write a ply with colors for each class"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis = 1)
    color = np.zeros(xyz.shape)
    
    if dataset == 's3dis':
        for i_label in range(1, n_label+1):
            color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    elif dataset == 'scannet':
        for i_label in range(1, n_label+1):
            color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    elif dataset == 'vkitti':
        for i_label in range(0, n_label):
            color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def simple_write_ply(filename, xyz, rgb):
    """write a ply with colors for each class"""
    
    print(xyz.shape, rgb.shape)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = rgb[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

