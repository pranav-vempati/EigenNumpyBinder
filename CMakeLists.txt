cmake_minimum_required(VERSION 3.12.2)
project(EigenNumpyBinder)

set(CMAKE_CXX_STANDARD 11)

find_package( PkgConfig)

pkg_check_modules(EIGEN3 REQUIRED eigen3)
include_directories( ${EIGEN3_INCLUDE_DIRS})

add_subdirectory(pybind11/pybind11)

pybind11_add_module(EigenNumpyBinder EigenNumpyBinder.cpp)
