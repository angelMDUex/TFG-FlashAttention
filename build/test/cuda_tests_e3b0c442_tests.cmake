add_test([=[QLoad.CopyTest]=]  /root/TFG-FlashAttention/build/test/cuda_tests [==[--gtest_filter=QLoad.CopyTest]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[QLoad.CopyTest]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/root/TFG-FlashAttention/test/test_tile_load.cu:72]==]
    WORKING_DIRECTORY [==[/root/TFG-FlashAttention/build/test]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[MMA.BF16Ones]=]  /root/TFG-FlashAttention/build/test/cuda_tests [==[--gtest_filter=MMA.BF16Ones]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[MMA.BF16Ones]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/root/TFG-FlashAttention/test/test_mma.cu:64]==]
    WORKING_DIRECTORY [==[/root/TFG-FlashAttention/build/test]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(cuda_tests_TESTS [==[QLoad.CopyTest]==] [==[MMA.BF16Ones]==])
