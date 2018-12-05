#! /usr/bin/bash -e

docker run -t -w /tensorflow \
  -v /home/cytang/workspace/tensorflow:/tensorflow \
  -v /home/cytang/workspace:/root \
  --name tensorflow \
  tensorflow/tensorflow:nightly-devel-py3 bash
