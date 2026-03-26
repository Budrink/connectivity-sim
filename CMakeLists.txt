cmake_minimum_required(VERSION 3.20)
project(connectivity_sim LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(ANISO_HAS_CUDA TRUE)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} — GPU targets enabled")
else()
    message(FATAL_ERROR "CUDA is required for this project")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- yaml-cpp ---
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# --- GLFW ---
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- Dear ImGui ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

# --- implot ---
FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG        v0.16
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(implot)

add_library(implot_lib STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_lib)

# ===========================================================================
#  GPU targets
# ===========================================================================
find_package(CUDAToolkit REQUIRED)

# --- GPU kernel library ---
add_library(aniso_gpu_core STATIC
    src/gpu/kernels.cu
    src/gpu/gpu_grid.cu
)
target_include_directories(aniso_gpu_core PUBLIC include)
target_link_libraries(aniso_gpu_core PUBLIC
    CUDA::cudart
    yaml-cpp::yaml-cpp
)
set_target_properties(aniso_gpu_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)
target_compile_options(aniso_gpu_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# --- GPU GUI executable ---
find_package(OpenGL REQUIRED)
add_executable(aniso_gpu src/gpu_main.cu)
target_include_directories(aniso_gpu PRIVATE include)
target_link_libraries(aniso_gpu PRIVATE
    aniso_gpu_core
    implot_lib
    CUDA::cudart
    OpenGL::GL
)
set_target_properties(aniso_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)
target_compile_options(aniso_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# --- GPU headless sweep (GL variant, CSV output) ---
add_executable(aniso_sweep_gl src/gpu_sweep_gl.cu)
target_include_directories(aniso_sweep_gl PRIVATE include)
target_link_libraries(aniso_sweep_gl PRIVATE aniso_gpu_core CUDA::cudart)
set_target_properties(aniso_sweep_gl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)
target_compile_options(aniso_sweep_gl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

message(STATUS "GPU targets: aniso_gpu_core, aniso_gpu, aniso_sweep_gl")
