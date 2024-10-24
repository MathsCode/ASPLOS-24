# CUDA_VISIBLE_DEVICES=1 python -m eagle.evaluation.gen_ea_answer_llama2chat --ea-model-path ~/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B --base-model-path /share/datasets/public_models/Llama-2-7b-chat-hf --question-begin 0 --question-end 1 --top-k 10 --total-token 60 > out_true.txt

# CUDA_VISIBLE_DEVICES=1 python -m eagle.evaluation.gen_ea_answer_llama2chat --ea-model-path ~/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B --base-model-path /share/datasets/public_models/Llama-2-7b-chat-hf --bench-name alpaca > out_true_alpaca.txt

CUDA_VISIBLE_DEVICES=7 python -m eagle.evaluation.gen_ea_answer_llama2chat --ea-model-path ~/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B --base-model-path /share/datasets/public_models/Llama-2-7b-chat-hf > out_7B_mt_EE.txt