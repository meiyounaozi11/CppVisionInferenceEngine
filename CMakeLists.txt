cmake_minimum_required(VERSION 3.21)

project(CppVisionInferenceEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(VISION_ENABLE_OPENCV "Build the OpenCV preprocessing pipeline" OFF)
option(VISION_ENABLE_ONNXRUNTIME "Build the ONNX Runtime CPU inference engine" OFF)
set(VISION_STAGE5_MODEL_PATH "" CACHE FILEPATH
    "Optional representative ONNX model used by Stage 5 correctness tests")

set(CPPVISION_CORE_SOURCES
    include/vision/Logging.h
    include/vision/ImageTensor.h
    include/vision/Status.h
    include/vision/Stopwatch.h
    include/vision/TaskMetadata.h
    include/vision/PerformanceMetrics.h
    src/Logging.cpp
    src/Status.cpp
    src/Stopwatch.cpp
    src/TaskMetadata.cpp
    src/PerformanceMetrics.cpp
)

if(VISION_ENABLE_OPENCV)
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
    list(APPEND CPPVISION_CORE_SOURCES
        include/vision/ImagePreprocessor.h
        src/ImagePreprocessor.cpp
    )
endif()

if(VISION_ENABLE_ONNXRUNTIME)
    if(NOT WIN32)
        message(FATAL_ERROR "Stage 2 currently supports the official Windows ONNX Runtime package only")
    endif()
    if(NOT DEFINED ENV{ONNXRUNTIME_ROOT} OR "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
        message(FATAL_ERROR "VISION_ENABLE_ONNXRUNTIME requires ONNXRUNTIME_ROOT")
    endif()
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain ONNX Runtime headers: ${ONNXRUNTIME_ROOT}")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib" OR
       NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain the Windows CPU runtime library")
    endif()
    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
    list(APPEND CPPVISION_CORE_SOURCES
        include/vision/InferenceEngine.h
        src/InferenceEngine.cpp
        include/vision/BoundedBlockingQueue.h
        include/vision/InferencePipeline.h
        src/InferencePipeline.cpp
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

if(VISION_ENABLE_ONNXRUNTIME)
    target_compile_definitions(CppVisionCore PUBLIC VISION_ENABLE_ONNXRUNTIME=1)
    target_link_libraries(CppVisionCore PUBLIC onnxruntime::onnxruntime)
endif()

add_executable(CppVisionInferenceEngine
    app/main.cpp
)

target_link_libraries(CppVisionInferenceEngine
    PRIVATE
        CppVisionCore
)

if(VISION_ENABLE_ONNXRUNTIME)
    add_custom_command(TARGET CppVisionInferenceEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:CppVisionInferenceEngine>/onnxruntime.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:CppVisionInferenceEngine>/onnxruntime_providers_shared.dll"
    )
endif()

if(VISION_ENABLE_ONNXRUNTIME AND VISION_ENABLE_OPENCV)
    add_executable(VisionPipelineBenchmark
        app/pipeline_benchmark.cpp
    )

    add_executable(RepresentativeVisionBenchmark
        app/representative_benchmark.cpp
    )

    target_link_libraries(RepresentativeVisionBenchmark
        PRIVATE
            CppVisionCore
    )

    target_link_libraries(VisionPipelineBenchmark
        PRIVATE
            CppVisionCore
    )

    add_custom_command(TARGET VisionPipelineBenchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:VisionPipelineBenchmark>/onnxruntime.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:VisionPipelineBenchmark>/onnxruntime_providers_shared.dll"
    )

    add_custom_command(TARGET RepresentativeVisionBenchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:RepresentativeVisionBenchmark>/onnxruntime.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:RepresentativeVisionBenchmark>/onnxruntime_providers_shared.dll"
    )
endif()

if(BUILD_TESTING)
    add_executable(VisionCoreTests
        tests/VisionCoreTests.cpp
    )

    target_link_libraries(VisionCoreTests
        PRIVATE
            CppVisionCore
    )

    add_test(NAME VisionCoreTests COMMAND VisionCoreTests)

    add_executable(PerformanceMetricsTests
        tests/PerformanceMetricsTests.cpp
    )

    target_link_libraries(PerformanceMetricsTests
        PRIVATE
            CppVisionCore
    )

    add_test(NAME PerformanceMetricsTests COMMAND PerformanceMetricsTests)

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

    if(VISION_ENABLE_ONNXRUNTIME AND VISION_ENABLE_OPENCV)
        add_executable(InferenceEngineTests
            tests/InferenceEngineTests.cpp
        )

        target_link_libraries(InferenceEngineTests
            PRIVATE
                CppVisionCore
        )

        target_compile_definitions(InferenceEngineTests
            PRIVATE
                VISION_TEST_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/models"
                VISION_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        )

        add_custom_command(TARGET InferenceEngineTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:InferenceEngineTests>/onnxruntime.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
                "$<TARGET_FILE_DIR:InferenceEngineTests>/onnxruntime_providers_shared.dll"
        )

        add_test(NAME InferenceEngineTests COMMAND InferenceEngineTests)

        add_executable(ConcurrencyPipelineTests
            tests/ConcurrencyPipelineTests.cpp
        )

        target_link_libraries(ConcurrencyPipelineTests
            PRIVATE
                CppVisionCore
        )

        target_compile_definitions(ConcurrencyPipelineTests
            PRIVATE
                VISION_TEST_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/models"
        )

        add_test(NAME ConcurrencyPipelineTests COMMAND ConcurrencyPipelineTests)

        if(VISION_STAGE5_MODEL_PATH AND EXISTS "${VISION_STAGE5_MODEL_PATH}")
            add_executable(RepresentativeModelInferenceTest
                tests/RepresentativeModelInferenceTest.cpp
            )

            target_link_libraries(RepresentativeModelInferenceTest
                PRIVATE
                    CppVisionCore
            )

            target_compile_definitions(RepresentativeModelInferenceTest
                PRIVATE
                    VISION_STAGE5_MODEL_PATH="${VISION_STAGE5_MODEL_PATH}"
                    VISION_STAGE5_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/representative/cat_image.jpg"
            )

            add_custom_command(TARGET RepresentativeModelInferenceTest POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                    "$<TARGET_FILE_DIR:RepresentativeModelInferenceTest>/onnxruntime.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
                    "$<TARGET_FILE_DIR:RepresentativeModelInferenceTest>/onnxruntime_providers_shared.dll"
            )

            add_test(NAME RepresentativeModelInferenceTest
                COMMAND RepresentativeModelInferenceTest)
        else()
            message(STATUS "Stage 5 representative model test disabled; set VISION_STAGE5_MODEL_PATH to an existing ONNX file")
        endif()
    endif()
endif()
