set -x
set -e

# CUDA_VISIBLE_DEVICES=0,1 python inference/mc_inference.py \
#     --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
#     --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
#     --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
#     --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
#     --output_path 'probabilistic_multiple_choice_results/open_ended_single_evidence_baseline'\
#     --input_key 'open_ended_single_evidence_prompt'\

# CUDA_VISIBLE_DEVICES=0,1 python inference/mc_inference.py \
#     --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
#     --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
#     --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
#     --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
#     --output_path 'probabilistic_multiple_choice_results/TACS_S_open_ended_single_evidence'\
#     --input_key 'open_ended_single_evidence_prompt'\
#     --TACS_mode 'PMC_single_sentence'

# CUDA_VISIBLE_DEVICES=0,1 python inference/mc_inference.py \
#     --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
#     --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
#     --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
#     --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
#     --output_path 'probabilistic_multiple_choice_results/TACS_T_open_ended_single_evidence'\
#     --input_key 'open_ended_single_evidence_prompt'\
#     --TACS_mode 'PMC_single_token'


# CUDA_VISIBLE_DEVICES=0,1 python inference/mc_inference.py \
#     --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
#     --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
#     --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
#     --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
#     --output_path 'probabilistic_multiple_choice_results/open_ended_double_evidence_baseline'\
#     --input_key 'open_ended_double_evidence_prompt'\

CUDA_VISIBLE_DEVICES=0,1 python inference/mc_inference.py \
    --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'probabilistic_multiple_choice_results/TACS_S_open_ended_double_evidence'\
    --input_key 'open_ended_double_evidence_prompt'\
    --TACS_mode 'PMC_double_sentence'

CUDA_VISIBLE_DEVICES=0,1 python inference/mc_inference.py \
    --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'probabilistic_multiple_choice_results/TACS_T_open_ended_double_evidence'\
    --input_key 'open_ended_double_evidence_prompt'\
    --TACS_mode 'PMC_double_token'
