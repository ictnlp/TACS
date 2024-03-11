set -e
set -x

python evaluate/evaluate.py \
    --result_name_tf 'generative_multiple_choice_results/single_evidence_true_false_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'single_evidence_true_false'\
    --output_key_tf 'single_evidence_true_false_output'\
    --result_name_ft 'generative_multiple_choice_results/single_evidence_false_true_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'single_evidence_false_true'\
    --output_key_ft 'single_evidence_false_true_output'

python evaluate/evaluate.py \
    --result_name_tf 'generative_multiple_choice_results/TACS_T_single_evidence_true_false_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'single_evidence_true_false'\
    --output_key_tf 'TACS_T_single_evidence_true_false_output'\
    --result_name_ft 'generative_multiple_choice_results/TACS_T_single_evidence_false_true_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'single_evidence_false_true'\
    --output_key_ft 'TACS_T_single_evidence_false_true_output'

python evaluate/evaluate.py \
    --result_name_tf 'generative_multiple_choice_results/TACS_S_single_evidence_true_false_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'single_evidence_true_false'\
    --output_key_tf 'TACS_S_single_evidence_true_false_output'\
    --result_name_ft 'generative_multiple_choice_results/TACS_S_single_evidence_false_true_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'single_evidence_false_true'\
    --output_key_ft 'TACS_S_single_evidence_false_true_output'



python evaluate/evaluate.py \
    --result_name_tf 'generative_multiple_choice_results/double_dif_random_evidence_answer_true_false_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'double_dif_random_evidence_answer_true_false'\
    --output_key_tf 'double_dif_random_evidence_answer_true_false_output'\
    --result_name_ft 'generative_multiple_choice_results/double_dif_random_evidence_answer_false_true_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'double_dif_random_evidence_answer_false_true'\
    --output_key_ft 'double_dif_random_evidence_answer_false_true_output'

python evaluate/evaluate.py \
    --result_name_tf 'generative_multiple_choice_results/TACS_T_double_dif_random_evidence_answer_true_false_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'double_dif_random_evidence_answer_true_false'\
    --output_key_tf 'TACS_T_double_dif_random_evidence_answer_true_false_output'\
    --result_name_ft 'generative_multiple_choice_results/TACS_T_double_dif_random_evidence_answer_false_true_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'double_dif_random_evidence_answer_false_true'\
    --output_key_ft 'TACS_T_double_dif_random_evidence_answer_false_true_output'

python evaluate/evaluate.py \
    --result_name_tf 'generative_multiple_choice_results/TACS_S_double_dif_random_evidence_answer_true_false_Llama-2-7b-chat.jsonl'\
    --input_key_tf 'double_dif_random_evidence_answer_true_false'\
    --output_key_tf 'TACS_S_double_dif_random_evidence_answer_true_false_output'\
    --result_name_ft 'generative_multiple_choice_results/TACS_S_double_dif_random_evidence_answer_false_true_Llama-2-7b-chat.jsonl'\
    --input_key_ft 'double_dif_random_evidence_answer_false_true'\
    --output_key_ft 'TACS_S_double_dif_random_evidence_answer_false_true_output'