#!/bin/bash
# script to install PoseNet for macOS 

# run this script while being in the /models/ directory

brew upgrade
brew install pkg-config cairo pango libpng jpeg giflib
mkdir PoseNet
cd PoseNet
yarn add @tensorflow-models/pose-detection
yarn add @tensorflow/tfjs-core
yarn add @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
touch posenet.js