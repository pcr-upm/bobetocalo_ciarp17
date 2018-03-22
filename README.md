# Head-Pose Estimation with Convolutional Neural Networks

We provide C++ code in order to replicate the head-pose experiments in our paper https://link.springer.com/chapter/10.1007/978-3-319-75193-1_6

If you use this code for your own research, you must reference our CIARP paper:

```
Benchmarking Head Pose Estimation in-the-Wild
Elvira Amador, Roberto Valle, José M. Buenaposada, Luis Baumela.
Conference on Progress in Pattern Recognition, Image Analysis, Computer Vision and Applications, 22nd Iberoamerican Congress, CIARP 2017, pp. 45-52, Valparaíso, Chile, November 7-10, 2017.
```

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework

#### Installation
This repository must be located inside the following directory:

    faces_framework
        └── headpose 
             └── bobetocalo_ciarp17
#### Usage
Example of how to use this repository:

```
> ./cmake-build-debug/face_headpose_bobetocalo_ciarp17_test --cnn GoogLeNet
```

