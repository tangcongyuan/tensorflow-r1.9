#compile_flags="--config=mkl --config=opt --verbose_failures"
#compile_flags="--compilation_mode=dbg --verbose_failures"
#compile_flags="--config cuda --compilation_mode=dbg --verbose_failures"
compile_flags="--config=opt --cxxopt=-DTENSORFLOW_USE_EPU"

#testcases="//tensorflow/python:mock_ip_test //tensorflow/core:fpga_util_test //tensorflow/python/kernel_tests:fpga_corex_ip_ops_test" #//tensorflow/core/kernels:fpga_pooling_op_test
#testcases="//tensorflow/core:fpga_util_test" # //tensorflow/python/kernel_tests:fpga_corex_ip_ops_test" #//tensorflow/core/kernels:fpga_pooling_op_test

use_python3=1
