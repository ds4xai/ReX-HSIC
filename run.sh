#!/bin/bash

echo "=== Début des entraînements ==="

DATASET_NAME=$1
SPLIT=$2

# Vérification des arguments
if [ -z "$DATASET_NAME" ] || [ -z "$SPLIT" ]; then
    echo "Usage: $0 <dataset_name> <split_strategy>"
    echo "Exemple: $0 IndianPines random"
    exit 1
fi

echo "Dataset : $DATASET_NAME"
echo "Split   : $SPLIT"

# Fonction pour lancer une commande
run () {
    echo "---------------------------------------------"
    echo "Lancement : $1"
    eval "$1"

    if [ $? -ne 0 ]; then
        echo "Erreur pendant l'exécution."
        exit 1
    fi

    echo "Process terminé."
    echo "---------------------------------------------"
}

mkdir -p logs

# Entraînements séquentiels

run "python trainer.py --model_name=ssrn --dataset_name=${DATASET_NAME} --n_epochs=200 --eval_step=1 --lr=3e-4 --patch_size=7 --gpu_id=0 --split_strategy=${SPLIT} > logs/training_${SPLIT}_${DATASET_NAME}_ssrn.log 2>&1"

run "python trainer.py --model_name=hamidaetal --dataset_name=${DATASET_NAME} --n_epochs=200 --eval_step=1 --patch_size=5 --lr=3e-4 --gpu_id=0 --split_strategy=${SPLIT} > logs/training_${SPLIT}_${DATASET_NAME}_hamidaetal.log 2>&1"

run "python trainer.py --model_name=dsformer --dataset_name=${DATASET_NAME} --use_pca=True --n_comps=30 --n_epochs=200 --eval_step=1 --patch_size=11 --lr=3e-4 --gpu_id=0 --split_strategy=${SPLIT} > logs/training_${SPLIT}_${DATASET_NAME}_dsformer.log 2>&1"

run "python trainer.py --model_name=spectralformer --dataset_name=${DATASET_NAME} --n_epochs=200 --eval_step=1 --patch_size=7 --lr=3e-4 --gpu_id=0 --split_strategy=${SPLIT} > logs/training_${SPLIT}_${DATASET_NAME}_spectralformer.log 2>&1"

run "python trainer.py --model_name=vit --dataset_name=${DATASET_NAME} --n_epochs=200 --eval_step=1 --patch_size=7 --lr=3e-4 --gpu_id=0 --split_strategy=${SPLIT} > logs/training_${SPLIT}_${DATASET_NAME}_vit.log 2>&1"

echo "=== Tous les entraînements pour ${DATASET_NAME} sont terminés ==="