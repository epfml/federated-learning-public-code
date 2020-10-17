This repository maintains a codebase for Federated Learning research. It supports:
* PyTorch with MPI backend for a Master-Worker computation/communication topology.
* Local training can be efficiently executed in a parallel-fashion over GPUs for randomly sampled clients.
* Different FL algorithms, e.g., FedAvg, FedProx, FedAvg with Server Momentum, and FedDF, are implemented as the baselines.

# Code Usage
## Requirements
We rely on `Docker` for our experimental environments. Please refer to the folder `environments` for more details.

## Usage
The current repository includes
* the methods evaluated in the paper `FedDF: Ensemble Distillation for Robust Model Fusion in Federated Learning`. For the detailed instructions and more examples, please refer to the file `codes/FedDF-code/README.md`.

# Reference
If you use the code in this repository, please consider to cite the following papers:
```
@inproceedings{lin2020ensemble,
  title={Ensemble Distillation for Robust Model Fusion in Federated Learning},
  author={Lin, Tao and Kong, Lingjing and Stich, Sebastian U and Jaggi, Martin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}
```
