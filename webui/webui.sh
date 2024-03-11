CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python webui.py\
    --model_name '/data/yutian/llama2/Llama-2-7b-chat-hf'\
    --token_svm_path '../tfqa/svm/svm_single_evidence_Llama-2-7b-chat-hf_fold2.pt'\
    --token_svm_acc '../tfqa/svm/acc_single_evidence_Llama-2-7b-chat-hf_fold2.pt'\
    --sentence_svm_path '../tfqa/svm/svm_mean_single_evidence_Llama-2-7b-chat-hf_fold2.pt'\
    --sentence_svm_acc '../tfqa/svm/acc_mean_single_evidence_Llama-2-7b-chat-hf_fold2.pt'\
    --TACS_mode 'DEMO_token'

# CUDA_VISIBLE_DEVICES=4,5,6,7 python test_webui.py