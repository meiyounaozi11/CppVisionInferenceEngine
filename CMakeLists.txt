cmake_minimum_required(VERSION 3.21)

project(CppVisionInferenceEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(VISION_ENABLE_OPENCV "Build the OpenCV preprocessing pipeline" OFF)

set(CPPVISION_CORE_SOURCES
    include/vision/Logging.h
    include/vision/Status.h
    include/vision/Stopwatch.h
    include/vision/TaskMetadata.h
    src/Logging.cpp
    src/Status.cpp
    src/Stopwatch.cpp
    src/TaskMetadata.cpp
)

if(VISION_ENABLE_OPENCV)
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
    list(APPEND CPPVISION_CORE_SOURCES
        include/vision/ImagePreprocessor.h
        src/ImagePreprocessor.cpp
    )
endif()

add_library(CppVisionCore STATIC
    ${CPPVISION_CORE_SOURCES}
)

target_include_directories(CppVisionCore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(CppVisionCore PUBLIC cxx_std_17)

if(VISION_ENABLE_OPENCV)
    target_compile_definitions(CppVisionCore PUBLIC VISION_ENABLE_OPENCV=1)
    target_link_libraries(CppVisionCore
        PUBLIC
            opencv_core
            opencv_imgproc
            opencv_imgcodecs
    )
endif()

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

    if(VISION_ENABLE_OPENCV)
        add_executable(ImagePreprocessorTests
            tests/ImagePreprocessorTests.cpp
        )

        target_link_libraries(ImagePreprocessorTests
            PRIVATE
                CppVisionCore
        )

        target_compile_definitions(ImagePreprocessorTests
            PRIVATE
                VISION_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        )

        add_test(NAME ImagePreprocessorTests COMMAND ImagePreprocessorTests)
    endif()
endif()
