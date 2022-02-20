#!/bin/bash
# script to install OpenFace for macOS 

# run this script while being in the /models/ directory

brew update
brew install gcc 
brew install boost
brew install tbb
brew install openblas
brew install --build-from-source dlib
brew install wget
brew install opencv

git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace

mkdir build
cd build
cmake -D WITH_OPENMP=ON CMAKE_BUILD_TYPE=RELEASE ..  
make

cd ..
bash download_models.sh 
cp lib/local/LandmarkDetector/model/patch_experts/*.dat build/bin/model/patch_experts/