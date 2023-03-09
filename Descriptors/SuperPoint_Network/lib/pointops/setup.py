#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointops',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',

            'src/ballquery/ballquery_cuda.cpp',
            'src/ballquery/ballquery_cuda_kernel.cu',
            
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            
            'src/knnquerycluster/knnquerycluster_cuda.cpp',
            'src/knnquerycluster/knnquerycluster_cuda_kernel.cu',
            
            'src/knnqueryclustergt/knnqueryclustergt_cuda.cpp',
            'src/knnqueryclustergt/knnqueryclustergt_cuda_kernel.cu',

            'src/knnquerypoint/knnquerypoint_cuda.cpp',
            'src/knnquerypoint/knnquerypoint_cuda_kernel.cu',

            'src/assofixp2c/assofixp2c_cuda.cpp',
            'src/assofixp2c/assofixp2c_cuda_kernel.cu',

            'src/assofixp2c_weight/assofixp2c_weight_cuda.cpp',
            'src/assofixp2c_weight/assofixp2c_weight_cuda_kernel.cu',

            'src/assomatrix/assomatrix_cuda.cpp',
            'src/assomatrix/assomatrix_cuda_kernel.cu',

            'src/assomatrix_label/assomatrix_label_cuda.cpp',
            'src/assomatrix_label/assomatrix_label_cuda_kernel.cu',

            'src/assomatrix_float/assomatrix_float_cuda.cpp',
            'src/assomatrix_float/assomatrix_float_cuda_kernel.cu',

            #'src/knnquerydilate/knnquerydilate_cuda.cpp',
            #'src/knnquerydilate/knnquerydilate_cuda_kernel.cu',

            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/grouping_int/grouping_int_cuda.cpp',
            'src/grouping_int/grouping_int_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',

            'src/labelstat/labelstat_cuda.cpp',
            'src/labelstat/labelstat_cuda_kernel.cu',

            'src/featuredistribute/featuredistribute_cuda.cpp',
            'src/featuredistribute/featuredistribute_cuda_kernel.cu'
        ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
