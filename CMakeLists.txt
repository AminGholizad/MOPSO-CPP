cmake_minimum_required(VERSION 3.10)
project(MOPSO-CPP CXX)

# ---- Compiler setup ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- clang-tidy ----


if(false)
    find_program(CLANG_TIDY_EXE clang-tidy)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY
    "${CLANG_TIDY_EXE}"
    "--extra-arg=--driver-mode=g++"
    "--extra-arg=-std=c++23"
    "--header-filter=${CMAKE_SOURCE_DIR}/.*"
  )
    else()
        message(WARNING "clang-tidy not found, CXX clang-tidy checks will not be run.")
    endif()
endif()


# ---- Default build type ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as default.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ---- Compiler warnings ----
add_compile_options(
    -Wall -Wextra -Werror -Wshadow -Wnon-virtual-dtor -Wold-style-cast
    -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion
    -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2
    -Wimplicit-fallthrough
)

# ---- Optimization flags ----
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
endif()

# ---- Output directories ----
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)

# ---- Add Random-Helper library ----
add_subdirectory(lib/Random-Helper)

# ---- Application ----
add_executable(app src/main.cpp)

target_include_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link header-only library (safe, no linking required)
target_link_libraries(app PRIVATE RandomHelper::RandomHelper)
