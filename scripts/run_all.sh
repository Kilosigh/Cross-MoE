export CUDA_VISIBLE_DEVICES=$3

# cd ../

all_models=("Informer" "Reformer" "PatchTST" "iTransformer" "FEDformer" "Nonstationary_Transformer" "TimeXer" "TimesNet" "TimeLLM")
start_index=$1
end_index=$2
use_uni=0
models=("${all_models[@]:$start_index:$end_index-$start_index+1}")
root_paths=("./data/Algriculture" "./data/Climate" "./data/Economy" "./data/Energy" "./data/Environment" "./data/Public_Health" "./data/Security" "./data/SocialGood" "./data/Traffic")
data_paths=("US_RetailBroilerComposite_Month.csv" "US_precipitation_month.csv" "US_TradeBalance_Month.csv" "US_GasolinePrice_Week.csv" "NewYork_AQI_Day.csv" "US_FLURATIO_Week.csv" "US_FEMAGrant_Month.csv" "Unadj_UnemploymentRate_ALL_processed.csv" "US_VMT_Month.csv")

# root_paths=("./data/Algriculture_non" "./data/Algriculture_LLM" "./data/Algriculture_gpt4")
# data_paths=("Algriculture_non.csv" "Algriculture_LLM.csv" "Algriculture_LLM_gpt4.csv" )


time_granularity_type=(0 0 0 1 2 1 0 0 0)
freq_array=("m" "m" "m" "w" "d" "w" "m" "m" "m" )
day_pred_lengths=(48 96 192 336)
week_pred_lengths=(12 24 36 48)
month_pred_lengths=(6 8 10 12)
pred_lengths=(6 8 10 12 12 24 36 48 48 96 192 336)

seeds=(2021)
use_fullmodel=0
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for model_name in "${models[@]}"
  do
    for ((i=0; i<$length; i++))
    do
      granularity_type=${time_granularity_type[$i]}
      start_offset=$((4 * granularity_type))
      # for pred_len in "${pred_lengths[@]}"
      for ((k=0; k<4; k++))
      do
        pl_addr=$((k + start_offset))
        pred_len=${pred_lengths[$pl_addr]}
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        model_id=$(basename ${root_path})

        echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path \
          --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
          --model $model_name \
          --data custom \
          --features S \
          --seq_len 24 \
          --label_len 12 \
          --pred_len $pred_len \
          --des 'Exp' \
          --seed $seed \
          --type_tag "#F#" \
          --text_len 4 \
          --prompt_weight 0.1 \
          --pool_type "avg" \
          --save_name "result_health_bert1" \
          --llm_model BERT \
          --huggingface_token 'NA'\
          --use_fullmodel $use_fullmodel \
          --freq ${freq_array[$i]} \
          --use_tx_moe 1 \
          --use_ts_moe 0 \
          --num_tx_experts 4 \
          --num_ts_experts 0 \
          --num_tx_moe_layers 1 \
          --num_ts_moe_layers 0 \
          --use_eva 0 \
          --use_text 1 \
          --use_Unified_model ${use_uni} \
          --use_Cross_MoE 1 \
          --mix_type 1 \
          --use_trainable_center 1 \
          --use_Cross_ranker 0 \
          --calculate_overhead 0 \
          --plot_tsne 0 \
          --use_k_means_init 1 \
          --plot_attn 1
      done
    done
  done
done
