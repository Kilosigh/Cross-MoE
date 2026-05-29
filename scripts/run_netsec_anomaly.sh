#!/bin/bash
# ============================================================================
# Network Security Anomaly Detection — Multimodal (TS + BERT text)
#
# Prerequisites:
#   1. Run:  python scripts/prepare_unsw_nb15.py
#   2. This downloads UNSW-NB15 and pre-computes BERT text embeddings.
#
# Modes:
#   classify:    TS encoder + BERT text → MLP → binary class   [SOTA, default]
#   reconstruct: TS encoder-decoder → reconstruction error      [baseline]
#
# Comparisons to run:
#   (a) Unimodal classify  → use_text=0
#   (b) Multimodal classify → use_text=1
#   (c) Unimodal reconstruct → use_text=0 detect_mode=reconstruct
#   (d) Multimodal reconstruct → use_text=1 detect_mode=reconstruct
# ============================================================================
set -e
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# ---- config ---------------------------------------------------------------
MODELS=("PatchTST" "TimesNet" "iTransformer")
ROOT_PATH="./data/UNSW-NB15"
DATA="UNSWNB15"

for MODEL in "${MODELS[@]}"; do
    echo "=============================================="
    echo "  Model: $MODEL  |  Multimodal Classify"
    echo "=============================================="
    python -u run.py \
        --task_name anomaly_detection \
        --is_training 1 \
        --model_id "UNSW_NB15_${MODEL}_multimodal_classify" \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --features M \
        --seq_len 64 \
        --pred_len 0 \
        --label_len 0 \
        --d_model 128 \
        --n_heads 8 \
        --e_layers 2 \
        --d_layers 1 \
        --d_ff 256 \
        --enc_in 49 \
        --dec_in 49 \
        --c_out 49 \
        --batch_size 64 \
        --train_epochs 30 \
        --patience 5 \
        --learning_rate 1e-3 \
        --loss MSE \
        --lradj cosine \
        --anomaly_ratio 0.1 \
        --detect_mode classify \
        --use_text 1 \
        --llm_model BERT \
        --llm_dim 768 \
        --embed fixed \
        --des "MultimodalClassify_${MODEL}"

    echo "=============================================="
    echo "  Model: $MODEL  |  Unimodal Classify (no text)"
    echo "=============================================="
    python -u run.py \
        --task_name anomaly_detection \
        --is_training 1 \
        --model_id "UNSW_NB15_${MODEL}_unimodal_classify" \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --features M \
        --seq_len 64 \
        --pred_len 0 \
        --label_len 0 \
        --d_model 128 \
        --n_heads 8 \
        --e_layers 2 \
        --d_layers 1 \
        --d_ff 256 \
        --enc_in 49 \
        --dec_in 49 \
        --c_out 49 \
        --batch_size 64 \
        --train_epochs 30 \
        --patience 5 \
        --learning_rate 1e-3 \
        --loss MSE \
        --lradj cosine \
        --anomaly_ratio 0.1 \
        --detect_mode classify \
        --use_text 0 \
        --llm_model BERT \
        --llm_dim 768 \
        --embed fixed \
        --des "UnimodalClassify_${MODEL}"

    echo "=============================================="
    echo "  Model: $MODEL  |  Unimodal Reconstruct"
    echo "=============================================="
    python -u run.py \
        --task_name anomaly_detection \
        --is_training 1 \
        --model_id "UNSW_NB15_${MODEL}_unimodal_reconstruct" \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --features M \
        --seq_len 64 \
        --pred_len 0 \
        --label_len 0 \
        --d_model 128 \
        --n_heads 8 \
        --e_layers 2 \
        --d_layers 1 \
        --d_ff 256 \
        --enc_in 49 \
        --dec_in 49 \
        --c_out 49 \
        --batch_size 64 \
        --train_epochs 30 \
        --patience 5 \
        --learning_rate 1e-3 \
        --loss MSE \
        --lradj cosine \
        --anomaly_ratio 0.1 \
        --detect_mode reconstruct \
        --use_text 0 \
        --embed fixed \
        --des "UnimodalReconstruct_${MODEL}"
done

echo ""
echo "All done! Results in result_anomaly_detection_MoE.txt"
