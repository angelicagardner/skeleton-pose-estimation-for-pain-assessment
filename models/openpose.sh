#!/bin/bash
# script to install OpenPose for macOS 

# run this script while being in the /models/ directory

brew update
brew install cmake --cask

git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose/
git submodule update --init --recursive --remote

mkdir build/
cd build/

# open file /build/caffe/src/openpose_lib-build/CMakeCache.txt
# change line 474 into:
# vecLib_INCLUDE_DIR:PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/

# open file /models/openpose/3rdparty/caffe/src/caffe/util/io.cpp
# change line 61 into only one argument:
# coded_input->SetTotalBytesLimit(kProtoReadBytesLimit);

make -j`sysctl -n hw.logicalcpu`