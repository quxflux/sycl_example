add_library(util src/benchmark.cpp)
target_link_libraries(util PRIVATE sycl_target)
target_include_directories(util PUBLIC inc)
