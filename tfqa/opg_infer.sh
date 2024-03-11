set -e
set -x

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --output_path 'open_ended_generation_results/open_ended_single_evidence_prompt_Llama-2-7b-chat.jsonl'\
    --input_key 'open_ended_single_evidence_prompt'\
    --output_key 'open_ended_single_evidence_prompt_output'\
    --batch_size 16\
    --max_new_tokens 128


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'open_ended_generation_results/TACS_T_open_ended_single_evidence_prompt_Llama-2-7b-chat.jsonl'\
    --input_key 'open_ended_single_evidence_prompt'\
    --output_key 'TACS_T_open_ended_single_evidence_prompt_output'\
    --batch_size 16\
    --TACS_mode 'OPG_single_token'\
    --window 5\
    --max_new_tokens 128

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'open_ended_generation_results/TACS_S_open_ended_single_evidence_prompt_Llama-2-7b-chat.jsonl'\
    --input_key 'open_ended_single_evidence_prompt'\
    --output_key 'TACS_S_open_ended_single_evidence_prompt_output'\
    --batch_size 16\
    --TACS_mode 'OPG_single_sentence'\
    --max_new_tokens 128


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --output_path 'open_ended_generation_results/open_ended_double_evidence_prompt_Llama-2-7b-chat.jsonl'\
    --input_key 'open_ended_double_evidence_prompt'\
    --output_key 'open_ended_double_evidence_prompt_output'\
    --batch_size 16\
    --max_new_tokens 128

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'open_ended_generation_results/TACS_T_open_ended_double_evidence_prompt_Llama-2-7b-chat.jsonl'\
    --input_key 'open_ended_double_evidence_prompt'\
    --output_key 'TACS_T_open_ended_double_evidence_prompt_output'\
    --batch_size 16\
    --TACS_mode 'OPG_double_token'\
    --max_new_tokens 128\
    --window 5

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'open_ended_generation_results/TACS_S_open_ended_double_evidence_prompt_Llama-2-7b-chat.jsonl'\
    --input_key 'open_ended_double_evidence_prompt'\
    --output_key 'TACS_S_open_ended_double_evidence_prompt_output'\
    --batch_size 16\
    --TACS_mode 'OPG_double_sentence'\
    --max_new_tokens 128