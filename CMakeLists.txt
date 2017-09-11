#------------------------------------------------------------------------------
# Test face alignment algorithms
#------------------------------------------------------------------------------
project(cnn)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)

#-- Setup required libraries
find_package(Caffe REQUIRED)
message(STATUS Caffe_LIBRARIES=${Caffe_LIBRARIES})
message(STATUS Caffe_INCLUDE_DIRS=${Caffe_INCLUDE_DIRS})
message(STATUS Caffe_DEFINITIONS=${Caffe_DEFINITIONS})

#find_package(HDF5 REQUIRED)
#message(STATUS HDF5_LIBRARIES=${HDF5_LIBRARIES})
#message(STATUS HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS})
#message(STATUS HDF5_DEFINITIONS=${HDF5_DEFINITIONS})

#-- Find include files
set(face_headpose_cnn_include
  ${faces_framework_include}
  ${Caffe_INCLUDE_DIRS}
#  ${HDF5_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_headpose_cnn_src
  ${faces_framework_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceHeadPoseCnn.cpp
)

set(face_headpose_cnn_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_headpose_cnn_test.cpp
)

set(face_headpose_cnn_libs
  ${faces_framework_libs}
  ${Caffe_LIBRARIES}
#  ${HDF5_LIBRARIES}
)

#-- Setup CMake to run tests
enable_testing()

include_directories(${face_headpose_cnn_include})

foreach(test ${face_headpose_cnn_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_headpose_cnn_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_headpose_cnn_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name})
endforeach()
