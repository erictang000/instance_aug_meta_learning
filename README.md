# Instance Specific Data Augmentation for Meta-Learning

## Dependencies
* Python 3.6+
* [PyTorch 0.4.0+](http://pytorch.org)
* [qpth 0.0.11+](https://github.com/locuslab/qpth)
* [tqdm](https://github.com/tqdm/tqdm)
* [kornia](https://github.com/kornia/kornia)

### Installation

1. Clone this repository:
    ```
    git clone https://github.com/erictang000/meta_instance_aug.git
    ```
2. Download and decompress dataset files: [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [**Spyros Gidaris**](https://github.com/gidariss/FewShotWithoutForgetting)), [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

### Reproducing Experiments
1. All experiment commands for training instance specific augmentations can be found in bash.sh. 
2. Test time augmentation was added in test_aug.ipynb, and results can be reproduced in that notebook.

## Acknowledgments
This code is based off the of the [**MetaAug**](https://github.com/RenkunNi/MetaAug) and [**InstaAug**](https://github.com/NingMiao/InstaAug) repos.

