'''

@author: Dr. Kangcheng Liu


'''

# Cython compile instructions
import numpy
from setuptools import setup, Extension
from Cython.Build import build_ext

# To compile, use
# python setup.py build --inplace
#

extensions = [
    Extension("pyshot",
              sources=["pyshot.pyx", './src/shot_descriptor.cpp'],
              include_dirs=[
                  numpy.get_include(),
                  'include/',
                  '/usr/include/eigen3/'],
              libraries=["lz4"],
              extra_compile_args=["-O3"],
              # we need this extra linker arguments to link against eigen3
              extra_link_args=['-L/usr/include/'],
              language="c++",
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              )
]

setup(
    name="pyshot",
    author="Julien Jerphanion",
    author_email="git@jjerphan.xyz",
    version='1.2',
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'numpy'
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: " "Implementation :: CPython",
    ],
    python_requires=">=3.6",
)
