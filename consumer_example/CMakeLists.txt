cmake_minimum_required(VERSION 3.20)
project(libhmm_consumer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed libhmm package
find_package(libhmm REQUIRED)

add_executable(consumer_demo main.cpp)
target_link_libraries(consumer_demo PRIVATE libhmm::hmm)
