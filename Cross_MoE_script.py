
import subprocess
import sys

# data_list = ["Algriculture", "Climate", "Economy", "Energy", "Entertainment", "Environment", "Public_Health",
#              "Security", "SocialGood", "Traffic"]

data_list = ["Algriculture", "Climate", "Economy", "Energy", "Environment", "Public_Health"]

data_dict = {
    "Algriculture": ["US_RetailBroilerComposite_Month_new_utf_8.csv", 3],
    "Climate": ["US_precipitation_month_new_utf_8.csv", 2],
    "Economy": ["US_TradeBalance_Month_new_utf_8.csv", 3],
    "Energy": ["US_GasolinePrice_Week_new_utf_8.csv", 9],
    "Entertainment": ["US_MovixGross_Day_new_utf_8.csv", 5],
    "Environment": ["NewYork_AQI_Day_new_utf_8.csv", 2],
    "Public_Health": ["US_FLURATIO_Week_multi_new_utf_8.csv", 8],
    "Security": ["US_FEMAGrant_Month_new_utf_8.csv", 1],
    "SocialGood": ["Unadj_UnemploymentRate_ALL_processed_new_utf_8.csv", 1],
    "Traffic": ["US_VMT_Month_new_utf_8.csv", 1]
}

horizon_dict = {
    "Algriculture": [6, 8, 10, 12],
    "Climate": [6, 8, 10, 12],
    "Economy": [6, 8, 10, 12],
    "Energy": [12, 24, 36, 48],
    "Entertainment": [48, 96, 192, 336],
    "Environment": [48, 96, 192, 336],
    "Public_Health": [12, 24, 36, 48],
    "Security": [6, 8, 10, 12],
    "SocialGood": [6, 8, 10, 12],
    "Traffic": [6, 8, 10, 12]
}

freq_dict = {
    "Algriculture": "m",
    "Climate": "m",
    "Economy": "m",
    "Energy": "w",
    "Entertainment": "d",
    "Environment": "d",
    "Public_Health": "w",
    "Security": "m",
    "SocialGood": "m",
    "Traffic": "m"
}

def pipeline(model, data, pred_len, batch_size=64):
    cmd = ["python", "run.py",
           "--task_name", "long_term_forecast",
           "--is_training", "1",
           "--root_path", "./data/" + data,
           "--data_path", data_dict[data][0],
           "--model_id", str(model) + data,
           "--model", model,
           "--data", "custom",
           "--features", "M",
           "--enc_in", str(data_dict[data][1]),
           "--dec_in", str(data_dict[data][1]),
           "--c_out", str(data_dict[data][1]),
           "--batch_size", str(batch_size),
           "--seq_len", "24",
           "--label_len", "12",
           "--pred_len", str(pred_len),
           "--seed", "2025",
           "--text_len", "4",
           "--prompt_weight", "0.1",
           "--pool_type", "avg",
           "--save_name", "result_health_bert",
           "--llm_model", "BERT",
           "--huggingface_token", "NA",
           "--use_fullmodel", "0", 
           "--freq", freq_dict[data], 
           "--use_tx_moe", str(use_tx_MoE),
           "--use_ts_moe", str(use_ts_MoE),
           "--num_tx_experts", str(MoE_num),
           "--num_ts_experts", str(MoE_num),
           "--use_eva", "0",
           "--use_text", str(use_text),
           "--use_Unified_model", str(use_uni), 
           "--use_Cross_MoE", str(use_Cross_MoE),
           "--mix_type", str(mix_type)
           
           ]
    subprocess.run(cmd)

all_models=["Informer", "Reformer", "PatchTST", "iTransformer", "FEDformer", "Nonstationary_Transformer", "TimeXer", "TimesNet"]



if __name__ == '__main__':
    # 第一个命令行参数（脚本名称）
    # print(f"Script name: {sys.argv[0]}")
    num_args = 1
    # 第二个命令行参数
    if len(sys.argv) > num_args:
        # print(f"First argument: {sys.argv[1]}")
        start_index=sys.argv[num_args]
        num_args += 1
    # 第三个命令行参数
    # if len(sys.argv) > num_args:
    #     # print(f"Second argument: {sys.argv[num_args]}")
    #     end_index=sys.argv[num_args]
    #     num_args += 1
    if len(sys.argv) > num_args:
        use_uni=sys.argv[num_args]
        num_args += 1

    model_name = all_models[int(start_index)]
    use_text = 1
    use_tx_MoE = 1
    use_ts_MoE = 0
    MoE_num = 4
    mix_type = 1
    use_Cross_MoE = 0

    for data in data_list:
        horizon_list = horizon_dict[data]
        for horizon in horizon_list:
            pipeline(model=model_name, data=data, pred_len=horizon)