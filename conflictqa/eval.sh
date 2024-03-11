set -e
set -x

python evaluate/evaluate.py \
    --result_name_tf './generative_multiple_choice_results/counter_memory_answer_ab_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'counter_memory_answer_ab'\
    --output_key_tf 'counter_memory_answer_ab_output'\
    --result_name_ft './generative_multiple_choice_results/counter_memory_answer_ba_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'counter_memory_answer_ba'\
    --output_key_ft 'counter_memory_answer_ba_output'

python evaluate/evaluate.py \
    --result_name_tf './generative_multiple_choice_results/TACS_S_counter_memory_answer_ab_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'counter_memory_answer_ab'\
    --output_key_tf 'counter_memory_answer_ab_output'\
    --result_name_ft './generative_multiple_choice_results/TACS_S_counter_memory_answer_ba_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'counter_memory_answer_ba'\
    --output_key_ft 'counter_memory_answer_ba_output'

python evaluate/evaluate.py \
    --result_name_tf './generative_multiple_choice_results/TACS_T_counter_memory_answer_ab_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'counter_memory_answer_ab'\
    --output_key_tf 'counter_memory_answer_ab_output'\
    --result_name_ft './generative_multiple_choice_results/TACS_T_counter_memory_answer_ba_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'counter_memory_answer_ba'\
    --output_key_ft 'counter_memory_answer_ba_output'