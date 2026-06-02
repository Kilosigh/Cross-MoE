#!/bin/bash
# UNSW-NB15 Anomaly Detection — Reconstruction (Time-Series-Library standard)
# Multi (TS + PCA text) vs Uni (TS only)
set -e; cd "$(dirname "$0")/.."; export CUDA_VISIBLE_DEVICES=${GPU:-0}

MODELS=("PatchTST" "TimesNet" "iTransformer")
ROOT="./data/UNSW-NB15"

for MODEL in "${MODELS[@]}"; do
    echo "=== $MODEL | Multimodal (TS + PCA text) ==="
    python -u run.py \
        --task_name anomaly_detection --is_training 1 \
        --model_id "UNSW_NB15_${MODEL}_multi" \
        --model $MODEL --data UNSWNB15 --root_path $ROOT \
        --features M --seq_len 64 --pred_len 0 --label_len 0 \
        --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 256 \
        --enc_in 47 --dec_in 47 --c_out 47 \
        --batch_size 64 --train_epochs 10 --patience 3 \
        --learning_rate 1e-3 --lradj cosine \
        --anomaly_ratio 0.1 --use_text 1 \
        --embed fixed --itr 1 --seed 2024 --des "multi_${MODEL}"

    echo "=== $MODEL | Unimodal (TS only) ==="
    python -u run.py \
        --task_name anomaly_detection --is_training 1 \
        --model_id "UNSW_NB15_${MODEL}_uni" \
        --model $MODEL --data UNSWNB15 --root_path $ROOT \
        --features M --seq_len 64 --pred_len 0 --label_len 0 \
        --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 256 \
        --enc_in 39 --dec_in 39 --c_out 39 \
        --batch_size 64 --train_epochs 10 --patience 3 \
        --learning_rate 1e-3 --lradj cosine \
        --anomaly_ratio 0.1 --use_text 0 \
        --embed fixed --itr 1 --seed 2024 --des "uni_${MODEL}"
done
echo "Done. Results in result_anomaly_detection_MoE.txt"
