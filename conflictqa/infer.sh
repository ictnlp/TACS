set -e
set -x

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/conflictQA-popQA-llama2-7b.jsonl'\
    --output_path 'generative_multiple_choice_results/counter_memory_answer_ab_Llama-2-7b-chat.jsonl'\
    --input_key 'counter_memory_answer_ab'\
    --output_key 'counter_memory_answer_ab_output'\
    --batch_size 24\
    --max_new_tokens 256

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/conflictQA-popQA-llama2-7b.jsonl'\
    --output_path 'generative_multiple_choice_results/counter_memory_answer_ba_Llama-2-7b-chat.jsonl'\
    --input_key 'counter_memory_answer_ba'\
    --output_key 'counter_memory_answer_ba_output'\
    --batch_size 24\
    --max_new_tokens 256


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/conflictQA-popQA-llama2-7b.jsonl'\
    --output_path 'generative_multiple_choice_results/TACS_T_counter_memory_answer_ab_Llama-2-7b-chat.jsonl'\
    --input_key 'counter_memory_answer_ab'\
    --output_key 'counter_memory_answer_ab_output'\
    --batch_size 24\
    --max_new_tokens 256\
    --window 5\
    --svm_path 'svm/svm_popqa_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_popqa_Llama-2-7b-chat-hf'\
    --TACS_mode 'GMC_single_token'

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/conflictQA-popQA-llama2-7b.jsonl'\
    --output_path 'generative_multiple_choice_results/TACS_T_counter_memory_answer_ba_Llama-2-7b-chat.jsonl'\
    --input_key 'counter_memory_answer_ba'\
    --output_key 'counter_memory_answer_ba_output'\
    --batch_size 24\
    --max_new_tokens 256\
    --window 5\
    --svm_path 'svm/svm_popqa_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_popqa_Llama-2-7b-chat-hf'\
    --TACS_mode 'GMC_single_token'


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/conflictQA-popQA-llama2-7b.jsonl'\
    --output_path 'generative_multiple_choice_results/TACS_S_counter_memory_answer_ab_Llama-2-7b-chat.jsonl'\
    --input_key 'counter_memory_answer_ab'\
    --output_key 'counter_memory_answer_ab_output'\
    --batch_size 24\
    --max_new_tokens 256\
    --svm_path 'svm/mean_svm_popqa_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/mean_acc_popqa_Llama-2-7b-chat-hf'\
    --TACS_mode 'GMC_single_sentence'

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/conflictQA-popQA-llama2-7b.jsonl'\
    --output_path 'generative_multiple_choice_results/TACS_S_counter_memory_answer_ba_Llama-2-7b-chat.jsonl'\
    --input_key 'counter_memory_answer_ba'\
    --output_key 'counter_memory_answer_ba_output'\
    --batch_size 24\
    --max_new_tokens 256\
    --svm_path 'svm/mean_svm_popqa_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/mean_acc_popqa_Llama-2-7b-chat-hf'\
    --TACS_mode 'GMC_single_sentence'

