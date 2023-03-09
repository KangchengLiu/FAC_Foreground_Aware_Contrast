#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ballquery/ballquery_cuda_kernel.h"
#include "grouping/grouping_cuda_kernel.h"
#include "grouping_int/grouping_int_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"
#include "interpolation/interpolation_cuda_kernel.h"
#include "knnquery/knnquery_cuda_kernel.h"

#include "knnquerycluster/knnquerycluster_cuda_kernel.h"

#include "knnqueryclustergt/knnqueryclustergt_cuda_kernel.h"

#include "knnquerypoint/knnquerypoint_cuda_kernel.h"

#include "assofixp2c/assofixp2c_cuda_kernel.h"

#include "assofixp2c_weight/assofixp2c_weight_cuda_kernel.h"

#include "assomatrix/assomatrix_cuda_kernel.h"

#include "assomatrix_label/assomatrix_label_cuda_kernel.h"

#include "assomatrix_float/assomatrix_float_cuda_kernel.h"

//#include "knnquerydilate/knnquerydilate_cuda_kernel.h"

#include "labelstat/labelstat_cuda_kernel.h"
#include "featuredistribute/featuredistribute_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ballquery_cuda", &ballquery_cuda_fast, "ballquery_cuda_fast");   // name in python, cpp function address, docs

    m.def("knnquery_cuda", &knnquery_cuda, "knnquery_cuda");
    
    m.def("knnquerycluster_cuda", &knnquerycluster_cuda, "knnquerycluster_cuda");
    m.def("knnqueryclustergt_cuda", &knnqueryclustergt_cuda, "knnqueryclustergt_cuda");
    
    m.def("knnquerypoint_cuda", &knnquerypoint_cuda, "knnquerypoint_cuda");

    m.def("assofixp2c_cuda", &assofixp2c_cuda, "assofixp2c_cuda");

    m.def("assofixp2c_weight_cuda", &assofixp2c_weight_cuda, "assofixp2c_weight_cuda");

    m.def("assomatrix_cuda", &assomatrix_cuda, "assomatrix_cuda");

    m.def("assomatrix_label_cuda", &assomatrix_label_cuda, "assomatrix_label_cuda");

    m.def("assomatrix_float_cuda", &assomatrix_float_cuda, "assomatrix_float_cuda");
    
    //m.def("knnquerydilate_cuda", &knnquerydilate_cuda, "knnquerydilate_cuda");

    m.def("grouping_forward_cuda", &grouping_forward_cuda_fast, "grouping_forward_cuda_fast");
    m.def("grouping_backward_cuda", &grouping_backward_cuda, "grouping_backward_cuda");

    m.def("grouping_int_forward_cuda", &grouping_int_forward_cuda_fast, "grouping_int_forward_cuda_fast");

    m.def("gathering_forward_cuda", &gathering_forward_cuda, "gathering_forward_cuda");
    m.def("gathering_backward_cuda", &gathering_backward_cuda, "gathering_backward_cuda");

    // add gathering_intxxxx
    m.def("gathering_int_forward_cuda", &gathering_int_forward_cuda, "gathering_int_forward_cuda");
    m.def("gathering_int_backward_cuda", &gathering_int_backward_cuda, "gathering_int_backward_cuda");

    // add gathering_clusterxxxx
    m.def("gathering_cluster_forward_cuda", &gathering_cluster_forward_cuda, "gathering_cluster_forward_cuda");
    m.def("gathering_cluster_backward_cuda", &gathering_cluster_backward_cuda, "gathering_cluster_backward_cuda");
    
    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");

    m.def("nearestneighbor_cuda", &nearestneighbor_cuda_fast, "nearestneighbor_cuda_fast");
    m.def("interpolation_forward_cuda", &interpolation_forward_cuda_fast, "interpolation_forward_cuda_fast");
    m.def("interpolation_backward_cuda", &interpolation_backward_cuda, "interpolation_backward_cuda");

    m.def("labelstat_idx_cuda", &labelstat_idx_cuda_fast, "labelstat_idx_cuda_fast");
    m.def("labelstat_ballrange_cuda", &labelstat_ballrange_cuda_fast, "labelstat_ballrange_cuda_fast");
    m.def("labelstat_and_ballquery_cuda", &labelstat_and_ballquery_cuda_fast, "labelstat_and_ballquery_cuda_fast");

    m.def("featuredistribute_cuda", &featuredistribute_cuda, "featuredistribute_cuda");
    m.def("featuregather_forward_cuda", &featuregather_forward_cuda, "featuregather_forward_cuda");
    m.def("featuregather_backward_cuda", &featuregather_backward_cuda, "featuregather_backward_cuda");
}
