CPMAddPackage("gh:quxflux/sorting_network_cpp#0a45cfa")

add_executable(sorting_net src/main.cpp)
target_link_libraries(sorting_net sycl_target util sorting_network_cpp)
target_include_directories(sorting_net PRIVATE src)
