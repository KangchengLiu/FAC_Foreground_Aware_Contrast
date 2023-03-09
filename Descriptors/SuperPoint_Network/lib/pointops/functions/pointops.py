from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops_cuda

import numpy as np


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, m):
        """
        input: xyz: (b, n, 3) and n > m, m: int32
        output: idx: (b, m)
        """
        assert xyz.is_contiguous()
        b, n, _ = xyz.size()
        idx = torch.cuda.IntTensor(b, m)
        temp = torch.cuda.FloatTensor(b, n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n, m, xyz, temp, idx)
        return idx

    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthestsampling = FurthestSampling.apply


class Gathering(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        input: features: (b, c, n), idx : (b, m) tensor
        output: (b, c, m)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        m = idx.size(1)
        output = torch.cuda.FloatTensor(b, c, m)
        pointops_cuda.gathering_forward_cuda(b, c, n, m, features, idx, output)
        ctx.for_backwards = (idx, c, n)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, c, n = ctx.for_backwards
        b, m = idx.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.gathering_backward_cuda(b, c, n, m, grad_out_data, idx, grad_features.data)
        return grad_features, None

gathering = Gathering.apply

class GatheringInt(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        input: features: (b, c, n), idx : (b, m) tensor
        output: (b, c, m)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        m = idx.size(1)
        output = torch.cuda.IntTensor(b, c, m)
        pointops_cuda.gathering_int_forward_cuda(b, c, n, m, features, idx, output)
        ctx.for_backwards = (idx, c, n)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, c, n = ctx.for_backwards
        b, m = idx.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.gathering_int_backward_cuda(b, c, n, m, grad_out_data, idx, grad_features.data)
        return grad_features, None

gatheringint = GatheringInt.apply

class Gathering_Cluster(Function):
    @staticmethod
    def forward(ctx, features, idx, idx_3d):
        """
        input: features: (b, c, n), idx : (b, m) tensor, idx_3d: (b, m, k)
        output: (b, c, m)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert idx_3d.is_contiguous()
        b, c, n = features.size()
        m = idx.size(1)
        k = idx_3d.size(2)
        output = torch.cuda.FloatTensor(b, c, m)
        pointops_cuda.gathering_cluster_forward_cuda(b, c, n, m, k, features, idx, idx_3d, output)
        ctx.for_backwards = (idx, idx_3d, c, n)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, idx_3d, c, n = ctx.for_backwards
        b, m = idx.size()
        k = idx_3d.size(2)
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.gathering_cluster_backward_cuda(b, c, n, m, k, grad_out_data, idx, idx_3d, grad_features.data)
        return grad_features, None, None

gathering_cluster = Gathering_Cluster.apply


class NearestNeighbor(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        input: unknown: (b, n, 3), known: (b, m, 3)
        output: dist2: (b, n, 3) l2 distance to the three nearest neighbors
                idx: (b, n, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()
        b, n, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(b, n, 3)
        idx = torch.cuda.IntTensor(b, n, 3)
        pointops_cuda.nearestneighbor_cuda(b, n, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

nearestneighbor = NearestNeighbor.apply


class Interpolation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        input: features: (b, c, m) features descriptors to be interpolated from
               idx: (b, n, 3) three nearest neighbors of the target features in features
               weight: (b, n, 3) weights
        output: (b, c, n) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()
        b, c, m = features.size()
        n = idx.size(1)
        ctx.interpolation_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(b, c, n)
        pointops_cuda.interpolation_forward_cuda(b, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input: grad_out: (b, c, n)
        output: grad_features: (b, c, m), None, None
        """
        idx, weight, m = ctx.interpolation_for_backward
        b, c, n = grad_out.size()
        grad_features = torch.cuda.FloatTensor(b, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.interpolation_backward_cuda(b, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None

interpolation = Interpolation.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        input: features: (b, c, n), idx : (b, m, nsample) containing the indicies of features to group with
        output: (b, c, m, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        _, m, nsample = idx.size()
        output = torch.cuda.FloatTensor(b, c, m, nsample)
        pointops_cuda.grouping_forward_cuda(b, c, n, m, nsample, features, idx, output)
        ctx.for_backwards = (idx, n)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: grad_out: (b, c, m, nsample)
        output: (b, c, n), None
        """
        idx, n = ctx.for_backwards
        b, c, m, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.grouping_backward_cuda(b, c, n, m, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None

grouping = Grouping.apply


class GroupingInt(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        input: features: (b, c, n), idx : (b, m, nsample) containing the indicies of features to group with
        output: (b, c, m, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        _, m, nsample = idx.size()
        output = torch.cuda.LongTensor(b, c, m, nsample)
        pointops_cuda.grouping_int_forward_cuda(b, c, n, m, nsample, features, idx, output)
        return output

    @staticmethod
    def backward(ctx, a=None):
        return None, None

grouping_int = GroupingInt.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        input: radius: float, radius of the balls
               nsample: int, maximum number of features in the balls
               xyz: torch.Tensor, (b, n, 3) xyz coordinates of the features
               new_xyz: torch.Tensor, (b, m, 3) centers of the ball query
        output: (b, m, nsample) tensor with the indicies of the features that form the query balls
        """
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, n, _ = xyz.size()
        m = new_xyz.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        pointops_cuda.ballquery_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

ballquery = BallQuery.apply


class FeatureDistribute(Function):
    @staticmethod
    def forward(ctx, max_xyz: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param max_xyz: (b, n, 3)
        :param xyz: (b, m, 3)
        :return: distribute_idx: (b, m)
        """
        assert max_xyz.is_contiguous()
        assert xyz.is_contiguous()
        b, n, _ = max_xyz.size()
        m = xyz.size(1)
        distribute_idx = torch.cuda.IntTensor(b, m).zero_()
        pointops_cuda.featuredistribute_cuda(b, n, m, max_xyz, xyz, distribute_idx)
        return distribute_idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None

featuredistribute = FeatureDistribute.apply


class FeatureGather(Function):
    @staticmethod
    def forward(ctx, max_feature: torch.Tensor, distribute_idx: torch.Tensor) -> torch.Tensor:
        '''
        :param ctx:
        :param max_feature: (b, c, n)
        :param distribute_idx: (b, m)
        :return: distribute_feature: (b, c, m)
        '''
        assert max_feature.is_contiguous()
        assert distribute_idx.is_contiguous()
        b, c, n = max_feature.size()
        m = distribute_idx.size(1)
        distribute_feature = torch.cuda.FloatTensor(b, c, m).zero_()
        pointops_cuda.featuregather_forward_cuda(b, n, m, c, max_feature, distribute_idx, distribute_feature)
        ctx.for_backwards = (distribute_idx, n)
        return distribute_feature

    @staticmethod
    def backward(ctx, grad_distribute_feature: torch.Tensor):
        '''
        :param ctx:
        :param grad_distribute_feature: (b, c, m)
        :return: grad_max_feature: (b, c, n),    None
        '''
        distribute_idx, n = ctx.for_backwards
        b, c, m = grad_distribute_feature.size()
        grad_max_feature = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_distribute_feature_data = grad_distribute_feature.data.contiguous()
        pointops_cuda.featuregather_backward_cuda(b, n, m, c, grad_distribute_feature_data, distribute_idx, grad_max_feature.data)
        return grad_max_feature, None

featuregather = FeatureGather.apply


class LabelStatBallRange(Function):
    @staticmethod
    def forward(ctx, radius: float, xyz: torch.Tensor, new_xyz: torch.Tensor, label_stat: torch.Tensor) -> torch.Tensor:
        '''
        :param ctx:
        :param radius:
        :param xyz: (b, n, 3)
        :param new_xyz: (b, m, 3)
        :param label_stat: (b, n, nclass)
        :return: new_label_stat: (b, m, nclass)
        '''
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        assert label_stat.is_contiguous()

        b, n, nclass = label_stat.size()
        m = new_xyz.size(1)
        new_label_stat = torch.cuda.IntTensor(b, m, nclass).zero_()
        pointops_cuda.labelstat_ballrange_cuda(b, n, m, radius, nclass, new_xyz, xyz, label_stat, new_label_stat)

        return new_label_stat

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

labelstat_ballrange = LabelStatBallRange.apply


class LabelStatIdx(Function):
    @staticmethod
    def forward(ctx, nsample: int, label_stat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''
        :param ctx:
        :param nsample:
        :param label_stat: (b, n, nclass)
        :param idx: (b, m, nsample)
        :return: new_label_stat: (b, m, nclass)
        '''
        assert label_stat.is_contiguous()
        assert idx.is_contiguous()

        b, n, nclass = label_stat.size()
        m = idx.size(1)
        new_label_stat = torch.cuda.IntTensor(b, m, nclass).zero_()
        pointops_cuda.labelstat_idx_cuda(b, n, m, nsample, nclass, label_stat, idx, new_label_stat)

        return new_label_stat

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

labelstat_idx = LabelStatIdx.apply


class LabelStatAndBallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, label_stat: torch.Tensor):
        '''
        :param ctx:
        :param radius:
        :param nsample:
        :param xyz: (b, n, 3)
        :param new_xyz: (b, m, 3)
        :param label_stat: (b, n, nclass)
        :return: new_label_stat: (b, m, nclass)  idx: (b, m, nsample)
        '''
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        assert label_stat.is_contiguous()

        b, n, nclass = label_stat.size()
        m = new_xyz.size(1)
        new_label_stat = torch.cuda.IntTensor(b, m, nclass).zero_()
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()

        pointops_cuda.labelstat_and_ballquery_cuda(b, n, m, radius, nsample, nclass, new_xyz, xyz, label_stat, idx, new_label_stat)

        return new_label_stat, idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None, None

labelstat_and_ballquery = LabelStatAndBallQuery.apply


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    import numpy as np
    return torch.clamp(dist, 0.0, np.inf)


class KNNQueryNaive(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        b, m, _ = new_xyz.size()
        n = xyz.size(1)

        '''
        idx = torch.zeros(b, m, nsample).int().cuda()
        for i in range(b):
            dist = pairwise_distances(new_xyz[i, :, :], xyz[i, :, :])
            [_, idxs] = torch.sort(dist, dim=1)
            idx[i, :, :] = idxs[:, 0:nsample]
        '''

        # '''
        # new_xyz_repeat = new_xyz.repeat(1, 1, n).view(b, m * n, 3)
        # xyz_repeat = xyz.repeat(1, m, 1).view(b, m * n, 3)
        # dist = (new_xyz_repeat - xyz_repeat).pow(2).sum(dim=2).view(b, m, n)
        dist = (new_xyz.repeat(1, 1, n).view(b, m * n, 3) - xyz.repeat(1, m, 1).view(b, m * n, 3)).pow(2).sum(dim=2).view(b, m, n)
        [_, idxs] = torch.sort(dist, dim=2)
        idx = idxs[:, :, 0:nsample].int()
        # '''
        return idx

    @staticmethod
    def backward(ctx):
        return None, None, None

knnquery_naive = KNNQueryNaive.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
                   ( dist2: (b, m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(b, m, nsample).zero_()
        pointops_cuda.knnquery_cuda(b, n, m, nsample, xyz, new_xyz, idx, dist2)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

knnquery = KNNQuery.apply


################################################################
# ---------   knn query clusters for each point ----------------
################################################################
class KNNQueryCluster(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, xyz_idx: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               xyz_idx: (b, n)  index of the coordinates
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
                   ( dist2: (b, m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous()
        assert xyz_idx.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        idx_abs = torch.cuda.IntTensor(b, m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(b, m, nsample).zero_()
        pointops_cuda.knnquerycluster_cuda(b, n, m, nsample, xyz, xyz_idx, new_xyz, idx, idx_abs, dist2)
        return idx, idx_abs

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None

knnquerycluster = KNNQueryCluster.apply


################################################################
# ---------   knn query clusters for each point ----------------
################################################################
class KNNQueryClusterGT(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, xyz_idx: torch.Tensor, xyz_gt: torch.Tensor, new_xyz: torch.Tensor = None, new_xyz_gt: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               xyz_idx: (b, n)  index of the coordinates
               xyz_gt: (b, n)  index of the coordinates
               new_xyz: (b, m, 3) centriods
               new_xyz_gt: (b, m, 1) centriods
            output: idx: (b, m, nsample)
                    idx_gt: (b, m, 1)
                   ( dist2: (b, m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        if new_xyz_gt is None:
            new_xyz_gt = xyz_gt.unsqueeze(-1)
        assert xyz.is_contiguous()
        assert xyz_idx.is_contiguous()
        assert xyz_gt.is_contiguous()
        assert new_xyz.is_contiguous()
        assert new_xyz_gt.is_contiguous()

        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        idx_abs = torch.cuda.IntTensor(b, m, nsample).zero_()
        idx_gt = torch.cuda.IntTensor(b, m, 1).zero_()
        idx_gt_abs = torch.cuda.IntTensor(b, m, 1).zero_()
        dist2 = torch.cuda.FloatTensor(b, m, nsample).zero_()
        pointops_cuda.knnqueryclustergt_cuda(b, n, m, nsample, xyz, xyz_idx, xyz_gt, new_xyz, new_xyz_gt, idx, idx_abs, idx_gt, idx_gt_abs, dist2)
        return idx, idx_abs, idx_gt, idx_gt_abs

    @staticmethod
    def backward(ctx, a=None, b=None, c=None, d=None):
        return None, None, None, None, None, None

knnqueryclustergt = KNNQueryClusterGT.apply

################################################################
# ---------   knn query points for each cluster ----------------
################################################################
class KNNQueryPoint(Function):
    @staticmethod
    def forward(ctx, ks: int, nsample: int, idx_c: torch.Tensor, cid: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               nsample: int32, Number of neighbor
               idx_c: (b, n, ks) indexs of cluster
               cid: (b, m, 1) centriods
        output: idx: (b, m, nsample)
        """
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        b, m, _ = cid.size()
        n = idx_c.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        pointops_cuda.knnquerypoint_cuda(b, n, m, ks, nsample, idx_c, cid, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

knnquerypoint = KNNQueryPoint.apply

################################################################
# ---------   association matrix for each cluster ----------------
################################################################
class AssoMatrix(Function):
    @staticmethod
    def forward(ctx, ks: int, idx_c: torch.Tensor, cid: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               idx_c: (b, n, ks) indexs of cluster
               cid: (b, m, 1) centriods
        output: idx: (b, m, n)
                cnt: (b, m, 1)
        """
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        b, m, _ = cid.size()
        n = idx_c.size(1)
        idx = torch.cuda.IntTensor(b, m, n).zero_()
        cnt = torch.cuda.IntTensor(b, m, 1).zero_()
        pointops_cuda.assomatrix_cuda(b, n, m, ks, idx_c, cid, idx, cnt)
        return idx, cnt

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None

assomatrix = AssoMatrix.apply


################################################################
# ---------   association matrix plus predict label for each cluster ----------------
################################################################
class AssoMatrixPlusLabel(Function):
    @staticmethod
    def forward(ctx, ks: int, idx_c: torch.Tensor, lab: torch.Tensor, cid: torch.Tensor = None, category=13) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               idx_c: (b, n, ks) indexs of cluster
               lab: (b, n, 1) label of point
               cid: (b, m, 1) centriods
        output: idx: (b, m, n)
                cnt: (b, m, 1)
                clab: (b, m, classes)
        """
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        b, m, _ = cid.size()
        n = idx_c.size(1)
        # print('category: {}'.format(category))
        idx = torch.cuda.IntTensor(b, m, n).zero_()
        cnt = torch.cuda.IntTensor(b, m, 1).zero_()
        clab = torch.cuda.IntTensor(b, m, category).zero_()
        pointops_cuda.assomatrix_label_cuda(b, n, m, ks, category, idx_c, lab, cid, idx, cnt, clab)
        return idx, cnt, clab

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None, None

assomatrixpluslabel = AssoMatrixPlusLabel.apply

################################################################
# ---------   association matrix for each cluster ----------------
################################################################
class AssoMatrixFloat(Function):
    @staticmethod
    def forward(ctx, ks: int, val_c: torch.Tensor, idx_c: torch.Tensor, cid: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               idx_c: (b, n, ks) indexs of cluster
               cid: (b, m, 1) centriods
        output: idx: (b, m, n)
                cnt: (b, m, 1)
        """
        assert val_c.is_contiguous()
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        b, m, _ = cid.size()
        n = idx_c.size(1)
        idx = torch.cuda.FloatTensor(b, m, n).zero_()
        cnt = torch.cuda.IntTensor(b, m, 1).zero_()
        pointops_cuda.assomatrix_float_cuda(b, n, m, ks, val_c, idx_c, cid, idx, cnt)
        return idx, cnt

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None

assomatrixfloat = AssoMatrixFloat.apply


################################################################
# ---------   association fix point for each cluster ----------------
################################################################
class AssoFixP2C(Function):
    @staticmethod
    def forward(ctx, ks: int, nsample: int, idx_c: torch.Tensor, cid: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               nsample: int32, Number of neighbor
               idx_c: (b, n, ks) indexs of cluster
               cid: (b, m, 1) centriods
        output: idx: (b, m, nsample)
        """
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        b, m, _ = cid.size()
        n = idx_c.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        pointops_cuda.assofixp2c_cuda(b, n, m, ks, nsample, idx_c, cid, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

assofixp2c = AssoFixP2C.apply

################################################################
# ---------   association fix point weight for each cluster ----------------
################################################################
class AssoFixP2CWeight(Function):
    @staticmethod
    def forward(ctx, ks: int, nsample: int, val_c: torch.Tensor, idx_c: torch.Tensor, cid: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               nsample: int32, Number of neighbor
               val_c: (b, n, ks) value of cluster
               idx_c: (b, n, ks) indexs of cluster
               cid: (b, m, 1) centriods
        output: idx: (b, m, nsample)
        """
        assert val_c.is_contiguous()
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        b, m, _ = cid.size()
        n = idx_c.size(1)
        idx = torch.cuda.FloatTensor(b, m, nsample).zero_()
        pointops_cuda.assofixp2c_weight_cuda(b, n, m, ks, nsample, val_c, idx_c, cid, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

assofixp2c_weight = AssoFixP2CWeight.apply