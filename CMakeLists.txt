cmake_minimum_required(VERSION 3.8)
project(HelloWorldCUDA LANGUAGES CXX CUDA)

# 设置CUDA的最小版本要求
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA工具集和库
find_package(CUDA REQUIRED)

# 添加CUDA编译标志
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_50,code=sm_50)

set(cuda_sources
    src/CudaDemo.cu  
    src/CudaTest.cpp
)

# 创建可执行文件
add_executable(CudaTest ${cuda_sources})

target_include_directories(CudaTest PRIVATE ${CMAKE_SOURCE_DIR})

# 链接CUDA库
target_link_libraries(CudaTest PRIVATE cuda)

# 设置CUDA属性
set_target_properties(CudaTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
