#!/bin/bash

echo "=== Début des entraînements pour le dataset $1 ==="

DATASET_NAME=$1

if [ -z "$DATASET_NAME" ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

run() {
    echo "---------------------------------------------"
    echo "Lancement : $1"
    eval "$1"
    echo "Process terminé."
    echo "---------------------------------------------"
}

# Séquentiel 
run "python trainer.py --model_name=ssrn --dataset_name=$DATASET_NAME --n_epochs=200 --eval_step=1 --lr=3e-4 --patch_size=7 --gpu_id=0 --split_strategy=random  > training_random_${DATASET_NAME}_log_ssrn.log 2>&1"
run "python trainer.py --model_name=hamidaetal --dataset_name=$DATASET_NAME --n_epochs=200 --eval_step=1 --patch_size=5 --lr=3e-4 --gpu_id=0 --split_strategy=random  > training_random_${DATASET_NAME}_log_hamidaetal.log 2>&1"
run "python trainer.py --model_name=dsformer --dataset_name=$DATASET_NAME --use_pca=True --n_comps=30 --n_epochs=200 --eval_step=1 --patch_size=11 --lr=3e-4 --gpu_id=0 --split_strategy=random  > training_random_${DATASET_NAME}_log_dsformer.log 2>&1"
run "python trainer.py --model_name=spectralformer --dataset_name=$DATASET_NAME --n_epochs=200 --eval_step=1 --patch_size=7 --lr=3e-4 --gpu_id=0 --split_strategy=random  > training_random_${DATASET_NAME}_log_spectralformer.log 2>&1"
run "python trainer.py --model_name=vit --dataset_name=$DATASET_NAME --n_epochs=200 --eval_step=1 --patch_size=7 --lr=3e-4 --gpu_id=0 --split_strategy=random  > training_random_${DATASET_NAME}_log_vit.log 2>&1"

echo "=== Tous les entraînements pour $DATASET_NAME sont terminés ==="
