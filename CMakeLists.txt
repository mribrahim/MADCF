cmake_minimum_required(VERSION 3.10)
project(tracker)



set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -pthread ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)

# set libtorch path
set(CMAKE_PREFIX_PATH /home/ibrahim/Desktop/Programlar/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED )
find_package(Threads REQUIRED)

include_directories (${OpenCV_INCLUDE_DIRS} {TORCH_INCLUDE_DIRS})


file(GLOB csrt
        src/csrt_ma/*.hpp
        src/csrt_ma/*.cpp)


add_executable(${PROJECT_NAME} src/main.cpp src/mytracker.cpp src/utils.cpp src/readData.cpp ${csrt})

add_compile_options(-pthread)

## at the 0mq library to our link directive
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES} Threads::Threads -lstdc++fs)
