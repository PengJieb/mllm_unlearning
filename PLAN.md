For unlearning of VLM-Safety-Unlearn/scripts/v1_5/finetune_unlearn_qwen3vl_lora.sh or VLM-Safety-Unlearn/scripts/v1_5/finetune_unlearn_qwen3vl_text_lora.sh, the unlearned model need: 
1. merge the unlearning model as they are LoRA weight by VLM-Safety-Unlearn/scripts/merge_lora_weights.py
2. Then evaluate on the sorry-bench in the following steps:
    1. generate sorry-bench answer by text_safety_bench/sorry-bench/gen_qwen3vl_answer.py
    2. generate llm judgement by text_safety_bench/sorry-bench/gen_judgment_safety_vllm.py
    3. obtain the evaluation metric, the code can refer notebook: notebook/vl_guard_split.ipynb
    4. Then save the result metrics into a json file
3. Evaluate on the beavertails benchmark by beavertails_qwen3vl_eval.py, also save the result metrics into a json file
4. Evaluate on the mmlu_redux by the mmlu_redux_qwen3vl_eval.py, also save the metrics into a json file

Now I have the unlearned model on vision+language VLGuard of algorithm: npo [VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-merged], rmu [VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-rmu-merged].
Then, I need the unlearned model on purely language VLGuaed of algorithm npo, due to rmu already have the checkpoint (LoRA).
So write bash files to get all results I need (original model and unleaned model results on these abovementioned becnhmarks).
In the finall, I only need to launch one bash scripts to launch all experiments.
Note, I can use the GPU 1, 2, 3, 4, 6, 7. The unlearning training use 4 GPUs, and evaluation per dataset uses 1 GPU.
My target model for unlearning (V+L and L) is Qwen/Qwen3-VL-2B-Instruct.
VLM-Safety-Unlearn/scripts/v1_5/finetune_unlearn_qwen3vl_lora.sh: Unlearning scripts of Qwen/Qwen3-VL-2B-Instruct on V+L VLGurad dataset.
VLM-Safety-Unlearn/scripts/v1_5/finetune_unlearn_qwen3vl_text_lora.sh, Unlearning scripts of Qwen/Qwen3-VL-2B-Instruct on L VLGurad dataset.
V+L is vision + language modality
L is the language modality.


In running_scripts/run_all_experiments.sh
No model qwen3-vl-2b-l-dpo, qwen3-vl-2b-l-npo, and VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-merged is the checkpoint from npo unlearning method not dpo.
We do not need to evaluate the dpo algorithm but need to evaluate npo.