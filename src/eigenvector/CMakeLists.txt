
# use FetchContent_Declare to download Eigen instead of CPMAddPackage("gl:libeigen/eigen") as the CMakeLists.txt in the Eigen repo
# creates a painfully large number of targets that are not required and can also not be suppressed via CMake options.
FetchContent_Declare(Eigen3 SYSTEM
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG a097f728fe8a6b87dddd33f205839eb617796238 # latest
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake) # no CMakeLists.txt in cmake, so this turns off configure
FetchContent_MakeAvailable(Eigen3)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 INTERFACE ${eigen3_SOURCE_DIR})

add_executable(eigenvector src/main.cpp)
target_link_libraries(eigenvector sycl_target util eigen3)
target_include_directories(eigenvector PRIVATE src)
target_compile_definitions(eigenvector PRIVATE EIGEN_RUNTIME_NO_MALLOC EIGEN_NO_MALLOC EIGEN_DONT_VECTORIZE)
