# Master's thesis project
==============================

This project contains the implementation described in '...', which I carried out in the spring semester of 2022 as part of my Master's degree at Linnaeus University.

## Abstract

...

## Dataset 

I used a private dataset consisting of 6-second video recordings with one person performing a overhead deep squat. The whole body and face are visible, the person looks into the camera from the front.

From the videos, [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) was used to detect the pose and extract a confidence score of the pose and an array of 17 keypoints. Each keypoint contains the x-position, the y-position and the confidence score. [OpenFace](https://github.com/cmusatyalab/openface) was used for face feature recognition and action unit (AU) extraction. OpenFace is able to recognise a subset of AUs: 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28 and 45. Out of those, OpenFace can detect all but one of the declared PSPI measure of pain intensity: **AU4 + (AU6 k AU7) + (AU9 k AU10) + AU43**. The missing action unit is AU43 (eye closure).

The file ```make_dataset.py``` was run once to load the CSV file output by PoseNet and OpenFace to create the final dataset used in this project.

## Usage

...

## Project Organization

------------
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models or script files to download models from GitHub repositories
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── architectures  <- Generated graph plots of the deep learning models
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   └── load_dataset.py    
    │   │
    │   ├── lib             <- External libraries used in this project
    │   │   └── DeepStack
    │   │   └── time_series_augmentation
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Note 

This repository is a hands-on part of a thesis project, and is therefore short-lived without long-term maintenance.