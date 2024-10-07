
# use FetchContent_Declare to download Eigen instead of CPMAddPackage("gl:libeigen/eigen") as the CMakeLists.txt in the Eigen repo
# creates a painfully large number of targets that are not required and can also not be suppressed via CMake options.
FetchContent_Declare(Eigen3 SYSTEM
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c # 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake) # no CMakeLists.txt in cmake, so this turns off configure
FetchContent_MakeAvailable(Eigen3)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 INTERFACE ${eigen3_SOURCE_DIR})

add_executable(inverse_matrix src/main.cpp)
target_link_libraries(inverse_matrix sycl_target util eigen3)
target_include_directories(inverse_matrix PRIVATE src)
target_compile_definitions(inverse_matrix PRIVATE EIGEN_RUNTIME_NO_MALLOC EIGEN_NO_MALLOC EIGEN_DONT_VECTORIZE)
