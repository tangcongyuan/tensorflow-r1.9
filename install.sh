#! /bin/bash -e
source flags.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/poweic/Local/ATG/acore/acore/python/
bazel build $compile_flags //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt

if [ $use_python3 -eq 1 ]
then 
    echo "Use Python3"
    pip3 uninstall -y tensorflow
    pip3 install /mnt/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl
else
    echo "Use Python2"
    pip2.7 uninstall -y tensorflow
    pip2.7 install /mnt/tensorflow-1.9.0rc0-cp27-cp27mu-linux_x86_64.whl
fi
