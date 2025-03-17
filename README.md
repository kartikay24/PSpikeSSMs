# P-SpikeSSM


Spiking neural networks (SNNs) are posited as a computationally efficient and biologically plausible alternative to conventional neural architectures, with their core computational framework primarily using the leaky integrate-and-fire (LIF) neuron model. However, the limited hidden state representation of LIF neurons, characterized by a scalar membrane potential, and sequential spike generation process, poses challenges for effectively developing scalable spiking models to address long-range dependencies in sequence learning tasks. In this study, we  develop a scalable probabilistic spiking learning framework for long-range dependency tasks leveraging the fundamentals of state space models. Unlike LIF neurons that rely on the deterministic Heaviside function for a sequential process of spike generation, we introduce a SpikeSampler layer that samples spikes stochastically based on an SSM-based neuronal model while allowing parallel computations. To address non-differentiability of the spiking operation and enable effective training, we also propose a surrogate function tailored for the stochastic nature of the SpikeSampler layer. To enhance inter-neuron communication, we introduce the SpikeMixer block, which integrates spikes from neuron populations in each layer. This is followed by a ClampFuse layer, incorporating a residual connection to capture complex dependencies, enabling scalability of the model. Our models attain state-of-the-art performance among SNN models across diverse long-range dependency tasks, encompassing the Long Range Arena benchmark, permuted sequential MNIST, and the Speech Command dataset and demonstrate sparse spiking pattern highlighting its computational efficiency. 

This code implements the methodology described in the paper titled: "P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks". The paper is accepted at ICLR-25.

## Setup

### Requirements
This repository requires Python 3.9+ and PyTorch 1.10+.
It has been tested up to PyTorch 1.13.1.
Other packages are listed in [requirements.txt](./requirements.txt). The code setup is similar to the S4 paper codebase.
Example installation:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```


#### Custom CUDA Kernel

Run `python setup.py install` from the directory `extensions/kernels/`.

#### Pykeops

This version is provided by the [pykeops library](https://www.kernel-operations.io/keops/python/installation.html).
Installation usually works out of the box with `pip install pykeops cmake` which are also listed in the requirements file.


## Getting Started


### Training with this Repository (Internal Usage)

This repository aims to provide a very flexible framework for training P-SpikeSSM based SNNs on the datasets specified in the paper.

The basic entrypoint is `python -m train`, for example,
```
python -m train experiment=lra/pSpikeSSM-mnist
```
This trains the pSpikeSSM model on the psMNIST dataset. Hyperparameters can also be modified in configs/model/base.yaml or added in configs/model/pSpikeSSM.yaml

## Training

The core training infrastructure of this repository is based on [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) with a configuration scheme based on [Hydra](https://hydra.cc/docs/intro/).

The main entrypoint is `train.py` and configs are found in `configs/`.

### Data

Basic datasets are auto-downloaded, including MNIST, and Speech Commands.
Directions for creating and loading datasets are in (./src/dataloaders/) directory.
The README inside this subdirectory documents how to download and organize other datasets like LRA benchmark.

### Models

Models are defined in (src/models). See the README in this subdirectory for an overview.


### Configs and Hyperparameters
Each experiment config is present within the experiments folder inside configs folder. For example, to run experiment on ListOps check  /config/experiment/lra/pSpikeSSM-listops.yaml
Configs can also be easily modified through the command line.
An example experiment is
```
python -m train experiment=lra/pSpikeSSM-imdb
```
Additional hyperparameter details are added in the Appendix of the paper. All associated hyperparameters can be found in configs/model/base.yaml.





<!--
#### Registries

This codebase uses a modification of the hydra `instantiate` utility that provides shorthand names of different classes, for convenience in configuration and logging.
The mapping from shorthand to full path can be found in `src/utils/registry.py`.
-->



## Overall Repository Structure
```
configs/         Config files for model, data pipeline, training loop, etc.
data/            Default location of raw data
extensions/      CUDA extensions (Cauchy and Vandermonde kernels)
src/             Main source code for model, datasets, etc.
  callbacks/     Training loop utilities (e.g. checkpointing)
  dataloaders/   Dataset and dataloader definitions
  models/        Model definitions
  tasks/         Encoder/decoder modules to interface between data and model backbone
  utils/
models/          Model-specific information (code, experiments, additional resources)
train.py         Training entrypoint for this repo
```






## Reference
Please cite this code with the following bibliography (to appear in ICLR-25 proceedings): 

```
@article{bal2024p,
  title={P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks},
  author={Bal, Malyaban and Sengupta, Abhronil},
  journal={arXiv preprint arXiv:2406.02923},
  year={2024}
}

