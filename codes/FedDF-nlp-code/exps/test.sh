declare -a list_of_manual_seed=(1)
# fusion
for ((sd=0;sd<${#list_of_manual_seed[@]};++sd)); do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python main.py --max_seq_len 128 \
        --experiment fusion \
        --model_info ptl=distilbert,model=distilbert-base-uncased \
        --task 'sst2' \
        --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
        --world 0 --num_workers 6 --batch_size 64 \
        --manual_seed ${list_of_manual_seed[sd]} \
        --partition_data non_iid_dirichlet --non_iid_alpha 1 \
        --target_perf 100 --early_stopping_rounds 10 \
        --n_clients 10 --participation_ratio 1  \
        --train_data_ratio 0.5 \
        --n_comm_rounds 10 --local_n_epochs 1 --early_stopping_epochs 100 \
        --fl_aggregate scheme=noise_knowledge_transfer,eval_ensemble=True,update_student_scheme=avg_logits,data_source=same,data_name=sst2,agg_data_ratio=0.97,total_n_server_pseudo_batches=5000,eval_batches_freq=20,early_stopping_server_batches=200 \
        --track_time False 
done

# fedavg
for ((sd=0;sd<${#list_of_manual_seed[@]};++sd)); do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python main.py --max_seq_len 128 \
        --experiment fedavg \
        --model_info ptl=distilbert,model=distilbert-base-uncased \
        --task 'sst2' \
        --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
        --world 0 --num_workers 6 --batch_size 64 \
        --manual_seed ${list_of_manual_seed[sd]} \
        --partition_data non_iid_dirichlet --non_iid_alpha 1 \
        --target_perf 100 --early_stopping_rounds 10 \
        --n_clients 10 --participation_ratio 1  \
        --train_data_ratio 0.5 \
        --n_comm_rounds 10 --local_n_epochs 1 --early_stopping_epochs 100 \
        --fl_aggregate scheme=federated_average,eval_ensemble=True,update_student_scheme=avg_logits,data_source=same,data_name=sst2,agg_data_ratio=0.97,total_n_server_pseudo_batches=5000,eval_batches_freq=20,early_stopping_server_batches=200 \
        --track_time False 
done