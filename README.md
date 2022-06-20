# Master Thesis Project
==============================

This project contains the implementation described in [Spot the Pain: Exploring the Application of Skeleton Pose Estimation for Automated Pain Assessment](), which I carried out in the spring semester of 2022 as part of my Master's degree project at Linnaeus University.

## Abstract

**Automated pain assessment is emerging as an essential part of pain management in areas such as healthcare, rehabilitation, sports and fitness. These automated systems are based on machine learning applications and can provide reliable, objective and cost-effective benefits. To enable an automated approach, at least one channel of sensory input, known as modality, must be available to the system. So far, most studies of automated pain assessment have focused on facial expressions or physiological signals, and although body gestures are considered to be indicators of pain, not much attention has been paid to this modality. Using skeleton pose estimation, we can model body gestures and investigate how body movement information affects pain assessment performance in different approaches. In this study, we explored approaches to pain assessment using skeleton pose estimation for three objectives: pain recognition, pain intensity estimation, and pain area classification. Because pain is a complex experience and is often expressed across multiple modalities, we analysed both unimodal approaches using only body data and bimodal approaches using skeleton pose estimation with facial expressions and head pose. In our experiments, we trained models based on two deep learning architectures: a hybrid CNN-BiLSTM and a recurrent CNN (RCNN), on a real-world dataset consisting of video recordings of people performing an overhead deep squat exercise. We also investigated bimodal fusion of body and face modalities in three different strategies: early fusion, late fusion and ensemble learning. Although our results are still preliminary, they show promising indications and possible future improvements. The best performance was obtained with ensemble for pain recognition (AUC 0.71), unimodal body CNN-BiLSTM for pain intensity estimation (AUC 0.75) and late fusion of body and face modalities using RCNN for pain area classification (AUC 0.75). Our experimental results demonstrate the feasibility of using skeleton pose estimation to represent body modality, the importance of incorporating body movements into automated pain assessment, and the exploration of the previously understudied assessment objective of localising pain areas in the body.**

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
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   └── load_dataset.py    
    │   │
    │   ├── lib             <- External libraries used in this project
    │   │   └── DeepStack
    │   │   └── time_series_augmentation
    │   │
    │   ├── models         <- Python classes to use for training models and save to make predictions
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Note 

This repository is a hands-on part of a thesis project, and is therefore short-lived without long-term maintenance.