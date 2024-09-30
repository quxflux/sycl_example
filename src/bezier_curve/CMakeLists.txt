add_executable(bezier_curve src/main.cpp)
target_link_libraries(bezier_curve sycl_target util)
target_include_directories(bezier_curve PRIVATE src)
