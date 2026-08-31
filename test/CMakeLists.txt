add_executable(
    cuda_tests
  test_tile_load.cu
  test_mma.cu  
)

target_include_directories(
    cuda_tests
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_tests
    PRIVATE
    GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(cuda_tests)
