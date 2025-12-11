cmake_minimum_required(VERSION 3.15)
project(MOPSO-CPP LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(lib/Random-Helper)

add_executable(MOPSO-CPP
    src/main.cpp
    # Add all other .cpp files here
)

target_link_libraries(MOPSO-CPP PRIVATE Random-Helper::Random-Helper)

set_target_properties(MOPSO-CPP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(MOPSO-CPP PRIVATE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_options(MOPSO-CPP PRIVATE
    -Wall -Wextra -Werror
    -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
    -Wunused -Wpedantic -Wconversion -Wsign-conversion
    -Wnull-dereference -Wdouble-promotion -Wformat=2
    -Wimplicit-fallthrough
)
target_compile_options(MOPSO-CPP PRIVATE
    \$<$<CONFIG:Debug>:-O0>
    \$<$<CONFIG:Release>:-O3 -DNDEBUG>
    \$<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    \$<$<CONFIG:RelWithDebInfo>:-O2 -DNDEBUG>
)
target_link_options(MOPSO-CPP PRIVATE
    \$<$<CONFIG:Debug>:-g>
    \$<$<CONFIG:RelWithDebInfo>:-g>
)

option(MOPSO-CPP_BUILD_TESTS "Build tests" ON)
if (MOPSO-CPP_BUILD_TESTS)
    enable_testing()
    add_executable(MOPSO-CPPTests
        tests/test_main.cpp
        # Add all other .cpp files here
    )
    target_include_directories(MOPSO-CPPTests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    add_test(NAME MOPSO-CPPTests COMMAND MOPSO-CPPTests)
endif()
