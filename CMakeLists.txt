cmake_minimum_required(VERSION 3.10)
project(RadarProcessing)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(radar_processing src/test_main.cc)

target_link_libraries(radar_processing PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
