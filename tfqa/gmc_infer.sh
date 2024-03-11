set -e
set -x

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --output_path 'generative_multiple_choice_results/single_evidence_true_false_Llama-2-7b-chat.jsonl'\
    --input_key 'single_evidence_true_false'\
    --output_key 'single_evidence_true_false_output'\
    --batch_size 16\


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --output_path 'generative_multiple_choice_results/single_evidence_false_true_Llama-2-7b-chat.jsonl'\
    --input_key 'single_evidence_false_true'\
    --output_key 'single_evidence_false_true_output'\
    --batch_size 16\


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_T_single_evidence_true_false_Llama-2-7b-chat.jsonl'\
    --input_key 'single_evidence_true_false'\
    --output_key 'TACS_T_single_evidence_true_false_output'\
    --TACS_mode 'GMC_single_token'\
    --batch_size 16

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_T_single_evidence_false_true_Llama-2-7b-chat.jsonl'\
    --input_key 'single_evidence_false_true'\
    --output_key 'TACS_T_single_evidence_false_true_output'\
    --TACS_mode 'GMC_single_token'\
    --batch_size 16

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_S_single_evidence_true_false_Llama-2-7b-chat.jsonl'\
    --input_key 'single_evidence_true_false'\
    --output_key 'TACS_S_single_evidence_true_false_output'\
    --TACS_mode 'GMC_single_sentence'\
    --batch_size 16

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_S_single_evidence_false_true_Llama-2-7b-chat.jsonl'\
    --input_key 'single_evidence_false_true'\
    --output_key 'TACS_S_single_evidence_false_true_output'\
    --TACS_mode 'GMC_single_sentence'\
    --batch_size 16

################################################################

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --output_path 'generative_multiple_choice_results/double_dif_random_evidence_answer_true_false_Llama-2-7b-chat.jsonl'\
    --input_key 'double_dif_random_evidence_answer_true_false'\
    --output_key 'double_dif_random_evidence_answer_true_false_output'\
    --batch_size 16\


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --output_path 'generative_multiple_choice_results/double_dif_random_evidence_answer_false_true_Llama-2-7b-chat.jsonl'\
    --input_key 'double_dif_random_evidence_answer_false_true'\
    --output_key 'double_dif_random_evidence_answer_false_true_output'\
    --batch_size 16\


python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_T_double_dif_random_evidence_answer_true_false_Llama-2-7b-chat.jsonl'\
    --input_key 'double_dif_random_evidence_answer_true_false'\
    --output_key 'TACS_T_double_dif_random_evidence_answer_true_false_output'\
    --TACS_mode 'GMC_double_token'\
    --batch_size 16

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_T_double_dif_random_evidence_answer_false_true_Llama-2-7b-chat.jsonl'\
    --input_key 'double_dif_random_evidence_answer_false_true'\
    --output_key 'TACS_T_double_dif_random_evidence_answer_false_true_output'\
    --TACS_mode 'GMC_double_token'\
    --batch_size 16

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_S_double_dif_random_evidence_answer_true_false_Llama-2-7b-chat.jsonl'\
    --input_key 'double_dif_random_evidence_answer_true_false'\
    --output_key 'TACS_S_double_dif_random_evidence_answer_true_false_output'\
    --TACS_mode 'GMC_double_sentence'\
    --batch_size 16

python inference/inference.py \
    --model_name $model_path\
    --data_path 'data/truthfulQA/TruthfulQA.jsonl'\
    --svm_path 'svm/svm_mean_single_evidence_Llama-2-7b-chat-hf'\
    --svm_acc 'svm/acc_mean_single_evidence_Llama-2-7b-chat-hf'\
    --output_path 'generative_multiple_choice_results/TACS_S_double_dif_random_evidence_answer_false_true_Llama-2-7b-chat.jsonl'\
    --input_key 'double_dif_random_evidence_answer_false_true'\
    --output_key 'TACS_S_double_dif_random_evidence_answer_false_true_output'\
    --TACS_mode 'GMC_double_sentence'\
    --batch_size 16

