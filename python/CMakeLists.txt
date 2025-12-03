cmake_minimum_required(VERSION 3.10)
project(ppf_python)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 查找OpenMP
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 第三方库路径配置
set(EIGEN3_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/eigen)
set(XSIMD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xsimd/include)
set(KDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../kdtree)
set(RPLY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/rply/rply)
set(RPLY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/rply/rply/rply.c)
set(GTL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/gtl/include")

# 获取pybind11包含目录
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11. Please install it with: pip install pybind11")
endif()

message(STATUS "Using pybind11 from: ${PYBIND11_INCLUDE_DIR}")

# 构建ppf库
add_library(ppf SHARED
    ../src/ppf.cpp 
    ../src/util.cpp 
    ../src/type.cpp 
    ../src/icp.cpp 
    ../src/filePLY.cpp 
    ${RPLY_SOURCE} 
    ../src/serialize.cpp 
)

target_include_directories(ppf PRIVATE
    ${KDTREE_INCLUDE_DIRS}
    ${XSIMD_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${RPLY_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${GTL_INCLUDE_DIR}
)

target_link_libraries(ppf PUBLIC 
    OpenMP::OpenMP_CXX 
)

target_compile_options(ppf PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj /arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU>: -mavx -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL>
)

target_compile_definitions(ppf PUBLIC API_EXPORTS)

add_library(ppf::ppf ALIAS ppf)

# Python绑定模块
add_library(ppf_python MODULE
    ppf_python.cpp
)

target_include_directories(ppf_python PRIVATE
    ${KDTREE_INCLUDE_DIRS}
    ${XSIMD_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${RPLY_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${GTL_INCLUDE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(ppf_python PRIVATE 
    ppf::ppf
    OpenMP::OpenMP_CXX
)

# 设置编译选项
target_compile_options(ppf_python PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj /arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU>: -mavx -fPIC>
)

# 设置输出名称
set_target_properties(ppf_python PROPERTIES
    OUTPUT_NAME ppf
    PREFIX ""  # 移除lib前缀
)