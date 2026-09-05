cmake_minimum_required(VERSION 3.21)

project(CppVisionInferenceEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(CppVisionCore STATIC
    include/vision/Logging.h
    include/vision/Status.h
    include/vision/Stopwatch.h
    include/vision/TaskMetadata.h
    src/Logging.cpp
    src/Status.cpp
    src/Stopwatch.cpp
    src/TaskMetadata.cpp
)

target_include_directories(CppVisionCore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(CppVisionCore PUBLIC cxx_std_17)

add_executable(CppVisionInferenceEngine
    app/main.cpp
)

target_link_libraries(CppVisionInferenceEngine
    PRIVATE
        CppVisionCore
)

if(BUILD_TESTING)
    add_executable(VisionCoreTests
        tests/VisionCoreTests.cpp
    )

    target_link_libraries(VisionCoreTests
        PRIVATE
            CppVisionCore
    )

    add_test(NAME VisionCoreTests COMMAND VisionCoreTests)
endif()
