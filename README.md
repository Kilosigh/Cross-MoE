 
## Usage

1. Install environment, execute the following command.

```
conda install --yes --file requirements.txt
```

2. Prepare for ClosedSource LLM. Our framework is already capable of integrating closed-source LLMs. To save costs, you should first use closed-source LLMs, such as GPT-3.5, to generate text-based predictions. We have provided specific preprocessing methods in the [[document/file](https://github.com/AdityaLab/MM-TSFlib/tree/main/data/DataPre_ClosedSourceLLM)]. We have also provided preprocessed data that can be directly used in `./data/` You can use any other closedsource llm to replace it.

3. Prepare for open-source LLMs. Our framework currently supports models such as LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, and GPT2XL, all available on Hugging Face. Please ensure you have your own Hugging Face token ready.

4. Train and evaluate model. We provide the example experiment script under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
#Conduct experiments on the all datasets using GPU 0 , and utilize the i-th to j-th models.
bash ./scripts/run_all.sh i j 0
```
- You can set a list of model names, prediction lengths, and random seeds in the script for batch experiments. We recommend specifying `--save_name` to better organize and save the results.
- `--llm_model` can set as LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, GPT2XL, Doc2Vec, ClosedLLM. When using ClosedLLM, you need to do Step 3 at first.
- `--pool_type` can set as avg min max attention for different pooling ways of token. When `--pool_type` is set to attention, we use the output of the time series model to calculate attention scores for each token in the LLM output and perform weighted aggregation.
- `--use_text` can set whether to use textual modality information. 1 denotes True, and 0 denotes False.
- `--use_Cross_MoE` can set whether to use Cross-MoE. 1 denotes True, and 0 denotes False.
- `--mix_type`  denotes fusion types. 0 for addition, 1 for cross-attn, 2 for cross-ranker
- `--use_tx_moe` can set whether to use MoE to project text embeddings. 1 denotes True, and 0 denotes False.
- `--use_ts_moe` can set whether to use MoE to project TS embeddings. 1 denotes True, and 0 denotes False.
- `--calculate_overhead` can set whether to execute the overhead calculation. 1 denotes True, and 0 denotes False.

# Cross-MoE
