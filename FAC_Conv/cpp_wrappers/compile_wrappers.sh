#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
rm grid_subsampling.cpython-37m-x86_64-linux-gnu.so
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
rm radius_neighbors.cpython-37m-x86_64-linux-gnu.so
python3 setup.py build_ext --inplace
cd ..