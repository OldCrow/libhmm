cmake_minimum_required(VERSION 3.20)
project(libhmm_fetchcontent_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch libhmm from GitHub (or use a local path for development)
FetchContent_Declare(
    libhmm
    GIT_REPOSITORY https://github.com/OldCrow/libhmm.git
    GIT_TAG main)

# To use a local checkout instead of fetching from GitHub, replace the FetchContent_Declare above
# with: FetchContent_Declare(libhmm SOURCE_DIR /path/to/libhmm)

# libhmm's LIBHMM_BUILD_* options default off when embedded
# (PROJECT_IS_TOP_LEVEL), so no forcing is needed for current main.
# Kept as explicit OFF for clarity and to protect against fetching an
# older ref where the defaults were unconditionally ON.
set(LIBHMM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LIBHMM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIBHMM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(libhmm)

add_executable(consumer_demo main.cpp)
target_link_libraries(consumer_demo PRIVATE libhmm::hmm)
