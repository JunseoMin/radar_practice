cmake_minimum_required(VERSION 3.10)
project(RadarSLAM)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib/gcc/x86_64-linux-gnu")

find_package(OpenMP)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

configure_file(
  pmc/pmc.CMakeLists.txt.in
  pmc-download/CMakeLists.txt
  @ONLY
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pmc-download"
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pmc-download"
)
add_subdirectory(
  "${CMAKE_BINARY_DIR}/pmc-download/pmc-src"
  "${CMAKE_BINARY_DIR}/pmc-download/pmc-build"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

set(radar_src
  src/FeatureBasedPoseTracker.cc
  src/LocalMapping.cc
  src/radar_utills.cc
)

add_executable(radar_processing src/test_main.cc ${radar_src})
add_executable(pose_estimate src/pose_test.cc ${radar_src})


target_link_libraries(radar_processing 
  PUBLIC 
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    Eigen3::Eigen
    pmc
    stdc++fs
    )

target_link_libraries(pose_estimate 
  PUBLIC 
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    Eigen3::Eigen
    pmc
    stdc++fs
    )
