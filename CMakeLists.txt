cmake_minimum_required(VERSION 3.31.9)

project(TFG-FA
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Angel Note: Clangd fails to recognize imports if the includes are in response files.
# Disable them.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)


# ============================================================
# Python + PyTorch
# ============================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND
        ${Python3_EXECUTABLE}
        -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# pytorch some kind of error not including the .so 
# FIX:
execute_process(
    COMMAND
        ${Python3_EXECUTABLE}
        -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib' / 'libtorch_python.so')"
    OUTPUT_VARIABLE TORCH_PYTHON_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT EXISTS "${TORCH_PYTHON_LIBRARY}")
    message(FATAL_ERROR
        "libtorch_python.so not found: ${TORCH_PYTHON_LIBRARY}"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# ============================================================
# Standalone CUDA executable
# ============================================================

add_executable(TFG-FA-Launcher
    src/cuda/standalone_launcher.cu
)

target_include_directories(TFG-FA-Launcher
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(TFG-FA-Launcher PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)


# ============================================================
# Python CUDA extension
# ============================================================

add_library(tfg_fa_cuda MODULE
    src/cuda/torch_binding_ampere.cu
)

set(FA_BUFFER_UNROLL $ENV{FA_BUFFER_UNROLL})
set(FA_BUFFER_NUM $ENV{FA_BUFFER_NUM})
set(FA_WARPS_PER_BLOCK $ENV{FA_WARPS_PER_BLOCK})
set(FA_MIN_CTAS_PER_SM $ENV{FA_MIN_CTAS_PER_SM})

target_compile_definitions(tfg_fa_cuda PRIVATE
    FA_BUFFER_UNROLL=${FA_BUFFER_UNROLL}
    FA_BUFFER_NUM=${FA_BUFFER_NUM}
    FA_WARPS_PER_BLOCK=${FA_WARPS_PER_BLOCK}
    FA_MIN_CTAS_PER_SM=${FA_MIN_CTAS_PER_SM}
)

target_include_directories(tfg_fa_cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tfg_fa_cuda
    PRIVATE
        "${TORCH_LIBRARIES}"
        "${TORCH_PYTHON_LIBRARY}"
        Python3::Python
)

target_compile_options(tfg_fa_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

set_target_properties(tfg_fa_cuda PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/src/py/tfg_fa"
)

# ============================================================
# Tests
# ============================================================

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(test)
