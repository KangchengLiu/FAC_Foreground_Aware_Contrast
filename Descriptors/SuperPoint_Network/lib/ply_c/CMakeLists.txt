# Graph for Cut Pursuit
# author: Loic Landrieu
# date: 2017

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(LIBGEOF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")


##############################
### Find required packages ###
##############################

find_package(Boost 1.65.1 COMPONENTS python numpy REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message("Boost includes ARE " ${Boost_INCLUDE_DIRS})
message("Boost LIBRARIES ARE " ${Boost_LIBRARY_DIRS})

#find_package(Eigen3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
#LINK_DIRECTORIES(${EIGEN3_LIBRARY_DIRS})


#SET(PYTHON_LIBRARIES /root/anaconda3/envs/Pytorch_1.0.0/lib/libpython3.6m.so)
#SET(PYTHON_INCLUDE_DIRS /root/anaconda3/envs/Pytorch_1.0.0/lib/python3.6/site-packages/)
find_package(PythonLibs)
message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})
##############################
###  Build target library  ###
##############################

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -fopenmp -O3 -Wall")

add_library(ply_c SHARED ply_c.cpp)
target_link_libraries(ply_c
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )
