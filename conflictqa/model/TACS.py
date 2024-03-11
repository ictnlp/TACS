from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import numpy as np
import torch


class TACS_model:
    def __init__(self, 
                 model_path: str,
                 TACS_mode = None,
                 svm_path = None,
                 svm_acc = None,
                 svm_num = 5):

        import model.llama as llama

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        self.model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.padding_side = 'left'
        self.TACS_mode = TACS_mode

        if svm_path is not None and svm_acc is not None:
            self.svm = torch.load(svm_path)
            self.acc = torch.load(svm_acc)
            self.sorted_indices = np.argsort(self.acc)[-svm_num:]  # 获取排序后的索引，最后5个是最大的
            self.layer_indices = self.sorted_indices.numpy()
        self.window = 7

    def truth_detection(self, encodings):
        tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
        start_id = [i.index('Information')+2 for i in tokenized]
        end_id = [i.index('Question')-2 for i in tokenized]

        with torch.no_grad():
            output = self.model(**encodings)

        if self.TACS_mode == 'GMC_single_token':
            result = [torch.zeros(end_id[i]-start_id[i]) for i in range(len(start_id))] 
    
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score

                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]-1:end_id[idx]-1]
                    predict_result = self.svm[layer][0].predict(head_score)
                    result[idx] += predict_result
            result = [torch.where(i>=1, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            for idx in range(len(result)):
                if self.window != 1:
                    for j in range(result[idx].shape[-1]):
                        result[idx][j]=1 if (
                            torch.mean(
                                result[idx][j:min(j+self.window, result[idx].shape[-1])]
                                ) >= 0.2) else 0
                encodings.attention_mask[idx][start_id[idx]:end_id[idx]] = result[idx]
        elif self.TACS_mode == 'GMC_single_sentence':
            result = torch.zeros(len(start_id))

            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score

                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]-1:end_id[idx]-1]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[idx] += predict_result
            result = [torch.where(i>=1, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            for idx in range(len(result)):
                if result[idx] == 0:
                    encodings.attention_mask[idx][start_id[idx]:end_id[idx]] = torch.zeros(end_id[idx]-start_id[idx])
        return encodings

    
    def generate(self, inputs, max_new_tokens=256):
        outputs = self.model.generate(input_ids=inputs.input_ids.cuda(0), attention_mask=inputs.attention_mask.cuda(0), 
                    max_new_tokens=max_new_tokens, 
                    return_dict_in_generate=True, 
                    repetition_penalty=1, 
                    temperature=1.0,
                    top_p=1.0,
                    do_sample  = False
                        )
        return outputs
