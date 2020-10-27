# Ensemble Distillation for Robust Model Fusion in Federated Learning
This repository is the official implementation of the preprint: *Ensemble Distillation for Robust Model Fusion in Federated Learning*. 

**Abstract:**
Federated Learning (FL) is a machine learning setting where many devices collaboratively train a machine learning model while keeping the training data decentralized. In most of the current training schemes the central model is refined by averaging the parameters of the server model and the updated parameters from the client side. However, directly averaging model parameters is only possible if all models have the same structure and size, which could be a restrictive constraint in many scenarios. In this work we investigate more powerful and more flexible aggregation schemes for FL. Specifically, we propose ensemble distillation for model fusion, i.e. training the central classifier through unlabeled data on the outputs of the models from the clients.
This knowledge distillation technique mitigates privacy risk and cost to the same extent as the baseline FL algorithms, but allows flexible aggregation over heterogeneous client models that can differ e.g. in size, numerical precision or structure. We show in extensive empirical experiments on various CV/NLP datasets (CIFAR-10/100, ImageNet, AG News, SST2) and settings (heterogeneous models/data) that the server model can be trained much faster, requiring fewer communication rounds than any existing FL technique so far.


## Requirements
Our implementations heavily rely on `Docker` and the detailed environment setup refers to `Dockerfile` under the `../environments` folder.

By running command `docker-compose build` under the folder `environments`, you can build our main docker image `pytorch-mpi`.


## Training and Evaluation
To train and evaluate the model(s) in the paper, run the following commands.


### CIFAR-10 with ResNet-8
We first consider a general FL system and then provide the configuration details for different methods.
* The non-iid local data distribution is controlled by the Dirichlet distribution with `alpha=1`.
* The FL system has `20` clients in total and the activation fraction of the clients per communication round is `0.4`.
* The FL system has `100` communication rounds in total and each round has `40` local training epochs.
* The local training schedule: lr=0.1, w/o lr decay, w/o nesterov momentum and heavy-ball momentum, w/o weight decay.

#### CIFAR-10 with ResNet-8 (homogeneous)
The setup of the FedAvg/FedProx for resnet-8 with cifar10:

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 40 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the FedAvg/FedProx for resnet-8 with cifar10 (with target performance and early stopping):
* The FL system has `100` communication rounds in total. However, the FL system will terminate if it reaches the target performance `target_perf`, or its performance plateaus for `early_stopping_rounds` rounds.

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 40 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False \
    --target_perf 100 --early_stopping_rounds 10
```

The setup of the FedAvg/FedProx for resnet-8 with cifar10 (drop random predictors):

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 40 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average,server_teaching_scheme=drop_worst \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the FedDF for resnet-8 with cifar10:
* The distillation dataset of the FedDF: downsampled ImageNet with image resolution 32.

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 40 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```


### CIFAR-100 with ResNet-8 (homogeneous)
Similar to the script examples illustrated above, we showcase the example of FedDF for resnet-8 with cifar100:
* The distillation dataset of the FedDF: downsampled ImageNet with image resolution 32.

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 40 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_percentage=1.0,data_name=imagenet32,data_dir=./dataset/ILSVRC,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
    --img_resolution 32 \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize False --same_seed_process False
```

We show the example of FedDF for resnet-8 with cifar100 (with controlled distillation dataset in terms of e.g. # of classes):
* The distillation dataset of the FedDF: downsampled ImageNet with image resolution 32.
* We select `num_total_class` from the original ImageNet while the `num_overlap_class` controls the number of overlapped classes w.r.t. CIFAR-100.

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 40 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=class_selection,data_percentage=1.0,num_total_class=100,num_overlap_class=0,data_name=imagenet32,data_dir=./dataset/ILSVRC,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
    --img_resolution 32 \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize False --same_seed_process False
```

### ImageNet with ResNet-8 (heterogeneous)
The script below shows how to train a heterogeneous FL system (i.e., `ResNet-32`, `shufflenetv2-1`, and `resnet20`) on ImageNet.

```
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --complex_arch master=resnet20,worker=resnet32:shufflenetv2-1:resnet20,num_clients_per_model=50 --experiment heterogeneous \
    --data imagenet32 --data_dir ./dataset/ILSVRC --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 \
    --train_data_ratio 1 --val_data_ratio 0.01 \
    --n_clients 150 --participation_ratio 0.1 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
    --n_comm_rounds 30 --local_n_epochs 40 \
    --optimizer sgd --lr 0.1 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize False --same_seed_process False
```
