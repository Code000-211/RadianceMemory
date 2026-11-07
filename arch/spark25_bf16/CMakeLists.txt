# Built in tree or against an installed radiance.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(spark25_bf16 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(spark25_bf16 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spark25_bf16.cpp)
