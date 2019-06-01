[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# FUNIT: Few-Shot Unsupervised Image-to-Image Translation
![animal swap gif](docs/images/animal.gif)

### [Project page](https://nvlabs.github.io/FUNIT/) |   [Paper](https://arxiv.org/abs/1905.01723) | [FUNIT Explained](https://youtu.be/kgPAqsC8PLM) | [PetSwap Demo Video](https://youtu.be/JTu-U0C4xEU) | [Have fun with PetSwap](https://nvlabs.github.io/FUNIT/petswap.html)


Few-shot Unsupervised Image-to-Image Translation<br>
[Ming-Yu Liu](http://mingyuliu.net/), [Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Arun Mallya](http://arunmallya.com/), [Tero Karras](https://research.nvidia.com/person/tero-karras), [Timo Aila](https://users.aalto.fi/~ailat1/), [Jaakko Lehtinen](https://users.aalto.fi/~lehtinj7/), and [Jan Kautz](http://jankautz.com/).<br>
In arXiv 2019.


### [License](https://raw.githubusercontent.com/nvlabs/FUNIT/master/LICENSE.md)

Copyright (C) 2019 NVIDIA Corporation.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [researchinquiries@nvidia.com](researchinquiries@nvidia.com).

## Installation

- Clone this repo `git clone https://github.com/NVlabs/FUNIT.git`
- Install [CUDA10.1+](https://developer.nvidia.com/cuda-downloads)
- Install [cuDNN7.5](https://developer.nvidia.com/cudnn)
- Install [Anaconda3](https://www.anaconda.com/distribution/)
- Install required python pakcages
    - `conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch`

To reproduce the results reported in the paper, you would need an **NVIDIA DGX1 machine with 8 V100 GPUs**.

## Dataset Preparation

### Animal Face Dataset

We are releasing the Animal Face dataset. **If you use this dataset in your publication, please cite the FUNIT paper.**

- The dataset consists of image crops of the [ImageNet ILSVRC2012 training set](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads). Download the dataset and untar the files
```
cd dataset
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
tar xvf ILSVRC2012_img_train.tar
```
- The training images should be in `datasets/ILSVRC/Data/CLS-LOC/train`. Now, extract the animal face images by running
```
python tools/extract_animal_faces.py datasets/ILSVRC/Data/CLS-LOC/train --output_folder datasets/animals --coor_file datasets/animal_face_coordinates.txt
```
- The animal face images should be in `datasets/animals`. Note there are 149 folders. Each folder contains images of one animal kind. The number of images of the dataset is 117,484.
- We use 119 animal kinds for training and the ramining 30 animal kinds for evaluation. 


## Training New Models

Once the animal face dataset is prepared, you can train an animal face translation model by running
```bash
python train.py --config configs/funit_animals.yaml
```

For training a model for a different task, please create a new config file based on the [example config](configs/funit_animals.yaml).

### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{liu2019few,
  title={Few-shot Unsueprvised Image-to-Image Translation},
  author={Ming-Yu Liu and Xun Huang and Arun Mallya and Tero Karras and Timo Aila and Jaakko Lehtinen and Jan Kautz.},
  booktitle={arxiv},
  year={2019}
}
```
