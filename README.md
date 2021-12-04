
## Overview
This repository provides an official implementation for our paper CVPR 2021 "Multi-view 3D Reconstruction of a Texture-less Smooth Surface of Unknown Generic Reflectance" - a multiview photometric pipeline for reconstructing glossy objects from a few calibrated multi-view images.

We assume the objects to be reconstructed are made from uniform material, and are illuminated by a single point light source that is closely attached to a moving camera (e.g. smartphones with a built-in flash light). The input to our system is a set of images captured by the camera, and the output is an oriented point cloud (3D + normal directions) that corresponds to the surface of the object, as well as its BRDF estimation.

There is no machine learning component in this pipeline. Instead, we base our solution solely on an energy model and design a robust optimisation technique for solving it! For more details please refer to our paper.

![pipeline overview](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/illustration.png?raw=true)

Please consider citing our paper if you find this implementation useful.
```
@inproceedings{cheng2021multi,
  title={Multi-view 3D Reconstruction of a Texture-less Smooth Surface of Unknown Generic Reflectance},
  author={Cheng, Ziang and Li, Hongdong and Asano, Yuta and Zheng, Yinqiang and Sato, Imari},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16226--16235},
  year={2021}
}
```
## Dependencies and requirement
 This implementation requqires an NVIDIA GPU for parallel computing. You will need about 2.3GB free VRAM for running the demo (memory requirement linearly increases with number of images and resolution). A conda virtual environment is recommended for managing packages. Starting with a fresh conda venv with Python3, install following packages:
- opencv-python
    '''shell
    pip install opencv-python --user
    '''
- pytorch with CUDA support (tested w/ v1.7.0)
    '''shell
    conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
    '''

- trimesh with pyembree support
   '''shell
   conda install -c conda-forge pyembree
   pip install trimesh --user
   '''

- matplotlib, scipy, tqdm, h5py, sklearn
    '''shell 
    pip install numpy scipy matplotlib tqdm h5py scikit-learn --user
    '''

## Clone this repository

```shell
git clone https://github.com/za-cheng/PM-PMVS.git
cd PM-PMVS
conda activate VENV_NAME # insert virtual environment name
```

## Prepare input files
Before running the pipeline you need to create an input file that contains multiview images and camera parameters. An exmaple script `steel-bunny.py` is included showing how to do this. In this example a steel bunny model is rendered at 10 viewpoints:
```shell
python steel-bunny.py
```
Running above command should create an input file at `data/steel-bunny.npy`. It should also save rendered images to `results/input-XX.png`, though these images won't be needed since they are already packed in the input file. 

(They will look something like this:)
![input0](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/input-00.png?raw=true)
![input1](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/input-01.png?raw=true)
![input2](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/input-02.png?raw=true)

Parameter configurations are supplied via `params.py` with instructions given inside. You will probably want to modify this script as well before running the pipeline on your own images.

## Run the pipeline
Simply run the pipeline with following commands:
```shell
python main.py
```

Intermediate results will be saved to `results/` folder as each iteration completes.


![shape](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/steel-bunny-29-shape.png?raw=true)
![normal](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/steel-bunny-29-normal.png?raw=true)
![reflectance](https://github.com/za-cheng/PM-PMVS/blob/main/git-imgs/steel-bunny-29-f.png?raw=true)

## License
The software is free for academic purposes. For commercial use please [Email us](mailto:ziang.cheng@anu.edu.au,hongdong.li@anu.edu.au).