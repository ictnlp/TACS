from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import numpy as np
import torch


class TACS_model:
    def __init__(self, 
                 model_path: str,
                 use_iti = False,
                 TACS_mode = None,
                 svm_path = None,
                 svm_acc = None,
                 svm_num = 5,
                 ):
        if use_iti:
            import model.honest_llama as llama
            self.tokenizer = llama.LlamaTokenizer.from_pretrained(model_path, legacy=False)
        else:
            import model.llama as llama
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

        self.model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.padding_side = 'left'
        self.TACS_mode = TACS_mode
        self.threshold = 0.5
        self.svm_num = svm_num

        if svm_path is not None and svm_acc is not None:
            self.svm = torch.load(svm_path)
            self.acc = torch.load(svm_acc)
            self.sorted_indices = np.argsort(self.acc)[-svm_num:]  # 获取排序后的索引，最后5个是最大的
            self.layer_indices = self.sorted_indices.numpy()
        self.window = 7

    def truth_detection(self, encodings):

        tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
        if self.TACS_mode == 'GMC_single_token':
            start_id = []
            end_id = [i.index('Options')-3 for i in tokenized]
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])):
                    if tokenized[i][j] == 'Information':
                        if tokenized[i][j+1] == ':':
                            start_id.append(j+1)
                            break

            with torch.no_grad():
                output = self.model(**encodings)

            result = [torch.zeros(end_id[i]-start_id[i]) for i in range(len(start_id))] 
    
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]:end_id[idx]]
                    predict_result = self.svm[layer][0].predict(head_score)
                    result[idx] += predict_result
            
            result = [torch.where(i>=3, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            print(result)
            for idx in range(len(result)):
                for j in range(result[idx].shape[-1]):
                    result[idx][j]=torch.round(
                        torch.mean(
                            result[idx][j:min(j+self.window, result[idx].shape[-1])]
                            ))
                encodings.attention_mask[idx][start_id[idx]+1:end_id[idx]+1] = result[idx]

        elif self.TACS_mode == 'GMC_double_token':

            information_num = 2
            start_id = [[-1 for x in range(information_num)] for i in range(len(tokenized))]
            end_id = [[-1 for x in range(information_num)] for i in range(len(tokenized))]
            
            for i in range(len(tokenized)):
                current_index = 1
                for j in range(len(tokenized[i])):
                    if tokenized[i][j] == str(current_index):
                        if tokenized[i][j+1] == ":":
                            start_id[i][current_index-1]=j+1
                            if current_index-1 >0:
                                end_id[i][current_index-2]=j-5
                            current_index += 1
                    if tokenized[i][j] == 'Options':
                        end_id[i][current_index-2]=j-3
                        break

            with torch.no_grad():
                output = self.model(**encodings)

            result = [[torch.zeros(end_id[i][j]-start_id[i][j]) for i in range(len(start_id))] for j in range(information_num)] 
            
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for info_id in range(information_num):
                    for idx in range(len(start_id)):
                        head_score = attn_score[idx,start_id[idx][info_id]:end_id[idx][info_id]]
                        predict_result = self.svm[layer][0].predict(head_score)
                        predict_result = torch.tensor(predict_result)
                        result[info_id][idx] += predict_result

            result = [[torch.where(i>=3, torch.ones_like(i), torch.zeros_like(i)) for i in j]for j in result]

            for info_id in range(information_num):
                for idx in range(len(result[0])):
                    for j in range(result[info_id][idx].shape[-1]):
                        result[info_id][idx][j]=torch.round(
                            torch.mean(
                                result[info_id][idx][j:min(j+self.window, result[info_id][idx].shape[-1])]
                                ))
                    encodings.attention_mask[idx][start_id[idx][info_id]+1:end_id[idx][info_id]+1] = result[info_id][idx]
        elif self.TACS_mode == 'GMC_single_sentence':
            start_id = []

            end_id = [i.index('Options')-3 for i in tokenized]
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])):
                    if tokenized[i][j] == 'Information':
                        if tokenized[i][j+1] == ':':
                            start_id.append(j+1)
                            break
            with torch.no_grad():
                output = self.model(**encodings)

            result = torch.zeros(len(start_id))

            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]:end_id[idx]]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[idx] += predict_result
            result = [torch.where(i>=3, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            for idx in range(len(result)):
                if result[idx] == 0:
                    encodings.attention_mask[idx][start_id[idx]+1:end_id[idx]+1] = torch.zeros(end_id[idx]-start_id[idx])
        elif self.TACS_mode == 'GMC_double_sentence':
            information_num = 2
            start_id = [[-1 for x in range(information_num)] for i in range(len(tokenized))]
            end_id = [[-1 for x in range(information_num)] for i in range(len(tokenized))]
            
            for i in range(len(tokenized)):
                current_index = 1
                for j in range(len(tokenized[i])):
                    if tokenized[i][j] == str(current_index):
                        if tokenized[i][j+1] == ":":
                            start_id[i][current_index-1]=j+1
                            if current_index-1 >0:
                                end_id[i][current_index-2]=j-5
                            current_index += 1
                    if tokenized[i][j] == 'Options':
                        end_id[i][current_index-2]=j-3
                        break

            with torch.no_grad():
                output = self.model(**encodings)

            result = torch.zeros((information_num, len(start_id)))
            
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for info_id in range(information_num):
                    for idx in range(len(start_id)):
                        head_score = attn_score[idx,start_id[idx][info_id]:end_id[idx][info_id]]
                        head_score = np.mean(head_score, axis=0)
                        predict_result = self.svm[layer][0].predict([head_score])
                        result[info_id][idx] += predict_result[0]
            result = [[i>=3 for i in j]for j in result]
            for info_id in range(information_num):
                for idx in range(len(result[0])):
                    if result[info_id][idx] == False:
                        encodings.attention_mask[idx][start_id[idx][info_id]+1:end_id[idx][info_id]+1] = torch.zeros(end_id[idx][info_id]-start_id[idx][info_id])
        elif self.TACS_mode == 'OPG_single_token' or self.TACS_mode =='OPG_double_token':

            start_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-1, -1, -1):
                    if tokenized[i][j] == '?':
                        start_id.append(j+1)
                        break
            end_id = [len(tokenized[0])-4 for i in tokenized]

            with torch.no_grad():
                output = self.model(**encodings)

            result = [torch.zeros(end_id[i]-start_id[i]) for i in range(len(start_id))] 
            
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(start_id)):
                    # print(idx, attn_score.shape)
                    head_score = attn_score[idx,start_id[idx]:end_id[idx]]
                    predict_result = self.svm[layer][0].predict(head_score)
                    result[idx] += predict_result
            
            result = [torch.where(i>=3, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            for idx in range(len(result)):
                for j in range(result[idx].shape[-1]):
                    result[idx][j]=torch.round(
                        torch.mean(
                            result[idx][j:min(j+self.window, result[idx].shape[-1])]
                            ))
                encodings.attention_mask[idx][start_id[idx]+1:end_id[idx]+1] = result[idx]
            
        elif self.TACS_mode == 'OPG_single_sentence':
            start_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-1, -1, -1):
                    if '?' in tokenized[i][j]:
                        start_id.append(j+1)
                        break
            end_id = [len(tokenized[0])-4 for i in tokenized]

            with torch.no_grad():
                output = self.model(**encodings)

            result = [0 for i in range(len(end_id))] 
            
            for layer in self.layer_indices:

                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]:end_id[idx]]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[idx] += predict_result[0]

            for i in range(len(result)):
                result[i] = result[i]>=3

            for idx in range(len(result)):
                if result[idx]==0:
                    encodings.attention_mask[idx][start_id[idx]+1:end_id[idx]+1] = torch.zeros(end_id[idx]-start_id[idx])

        elif self.TACS_mode == 'OPG_double_sentence':
            start_id = []
            mid_id = []

            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-1, -1, -1):
                    if '?' in tokenized[i][j]:
                        start_id.append(j+1)
                        for k in range(j+2, len(tokenized[i])):
                            if tokenized[i][k]=='<0x0A>':
                                mid_id.append(k)
                                break
                        break
            end_id = [len(tokenized[0])-4 for i in tokenized]
            
            with torch.no_grad():
                output = self.model(**encodings)

            result = [[0,0] for i in range(len(end_id))] 

            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]:mid_id[idx]-1]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[idx][0] += predict_result[0]

                    head_score = attn_score[idx,mid_id[idx]:end_id[idx]]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[idx][1] += predict_result[0]

            for i in range(len(result)):
                for j in range(2):
                    result[i][j] = result[i][j]>=3

            for idx in range(len(result)):
                if result[idx][0]==0:
                    encodings.attention_mask[idx][start_id[idx]+1:mid_id[idx]] = torch.zeros(mid_id[idx]-start_id[idx]-1)
                if result[idx][1]==0:    
                    encodings.attention_mask[idx][mid_id[idx]+1:end_id[idx]+1] = torch.zeros(end_id[idx]-mid_id[idx])
        elif self.TACS_mode == 'PMC_single_token' or self.TACS_mode == 'PMC_double_token':
            start_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-1, -1, -1):
                    if '?' in tokenized[i][j]:
                            start_id.append(j+1)
                            break
            end_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-2, -1, -1):
                    if tokenized[i][j] == 'A' and tokenized[i][j+1]==':':
                        end_id.append(j-2)
                        break
            with torch.no_grad():
                output = self.model(**encodings)

            result = [torch.zeros(end_id[i]-start_id[i]) for i in range(len(start_id))] 
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for id in range(len(start_id)):
                    head_score = attn_score[id,start_id[id]:end_id[id]]
                    predict_result = self.svm[layer][0].predict(head_score)
                    result[id] += predict_result
            result = [torch.where(i>=3, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            for id in range(len(result)):
                for j in range(result[id].shape[-1]):
                    result[id][j]=torch.round(
                        torch.mean(
                            result[id][j:min(j+self.window, result[id].shape[-1])]
                            ))
            for id in range(len(result)):
                encodings.attention_mask[id][start_id[id]+1:end_id[id]+1] = result[id]
            
        elif self.TACS_mode == 'PMC_single_sentence':
            start_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-1, -1, -1):
                    if '?' in tokenized[i][j]:
                            start_id.append(j+1)
                            break
            end_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-2, -1, -1):
                    if tokenized[i][j] == 'A' and tokenized[i][j+1]==':':
                        end_id.append(j-2)
                        break

            with torch.no_grad():
                output = self.model(**encodings)

            result = [0] 
            
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for id in range(len(start_id)):
                    head_score = attn_score[id,start_id[id]:end_id[id]]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[id] += predict_result[0]
            for i in range(len(result)):
                result[i] = 1 if result[i]>=3 else 0
            for id in range(len(result)):
                if not result[id]:
                    encodings.attention_mask[id][start_id[id]+1:end_id[id]+1] = torch.zeros(end_id[id]-start_id[id])
            
        elif self.TACS_mode == 'PMC_double_sentence':
            start_id = []
            mid_id = []

            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-1, -1, -1):
                    if '?' in tokenized[i][j]:
                        start_id.append(j+1)
                        for k in range(j+2, len(tokenized[i])):
                            if tokenized[i][k]=='<0x0A>':
                                mid_id.append(k)
                                break
                        break

            end_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])-2, -1, -1):
                    if tokenized[i][j] == 'A' and tokenized[i][j+1]==':':
                        end_id.append(j-2)
                        break

            with torch.no_grad():
                output = self.model(**encodings)

            result = [0,0] 
            
            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for id in range(len(start_id)):
                    head_score = attn_score[id,start_id[id]:mid_id[id]-1]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[0] += predict_result[0]

                    head_score = attn_score[id,mid_id[id]:end_id[id]]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[1] += predict_result[0]

            for i in range(len(result)):
                result[i] = 1 if result[i]>=3 else 0

            id = 0
            if not result[0]:
                encodings.attention_mask[0][start_id[0]+1:mid_id[0]] = torch.zeros(mid_id[0]-start_id[0]-1)
            if not result[1]:
                encodings.attention_mask[0][mid_id[0]+1:end_id[0]+1] = torch.zeros(end_id[0]-mid_id[0])
        elif self.TACS_mode == 'DEMO_token':
            start_id = []
            end_id = []
            i=0
            info_begin=0
            for j in range(len(tokenized[i])):
                if tokenized[i][j] == 'Information':
                    if tokenized[i][j+1] == ':':
                        start_id.append(j+1)
                        info_begin=1
                elif info_begin and tokenized[i][j]=='.':
                    end_id.append(j)
                    start_id.append(j)
                elif tokenized[i][j] == 'Answer':
                    # 删去最后一个元素
                    start_id.pop()
                    break
            for i in range(len(end_id)):
                print(start_id[i], end_id[i])
                print(tokenized[0][start_id[i]+1:end_id[i]+1])

            with torch.no_grad():
                output = self.model(**encodings)

            result = [torch.zeros(end_id[i]-start_id[i]) for i in range(len(start_id))] 

            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(start_id)):
                    head_score = attn_score[0,start_id[idx]:end_id[idx]]
                    predict_result = self.svm[layer][0].predict(head_score)
                    result[idx] += predict_result
            
            result = [torch.where(i>=self.svm_num*self.threshold, torch.ones_like(i), torch.zeros_like(i)) for i in result]
            
            for idx in range(len(result)):
                for j in range(result[idx].shape[-1]):
                    result[idx][j]=torch.round(
                        torch.mean(
                            result[idx][j:min(j+self.window, result[idx].shape[-1])]
                            ))
                encodings.attention_mask[0][start_id[idx]+1:end_id[idx]+1] = result[idx]
        elif self.TACS_mode == 'DEMO_sentence':
            start_id = []
            end_id = []
            i=0
            info_begin=0
            for j in range(len(tokenized[i])):
                if tokenized[i][j] == 'Information':
                    if tokenized[i][j+1] == ':':
                        start_id.append(j+1)
                        info_begin=1
                elif info_begin and tokenized[i][j]=='.':
                    end_id.append(j)
                    start_id.append(j)
                elif tokenized[i][j] == 'Answer':
                    # 删去最后一个元素
                    start_id.pop()
                    break
            
            print(tokenized)
            print(start_id)
            print(end_id)
            for i in range(len(end_id)):
                print(start_id[i], end_id[i])
                print(tokenized[0][start_id[i]+1:end_id[i]+1])

            with torch.no_grad():
                output = self.model(**encodings)

            result = torch.zeros(len(end_id))

            for layer in self.layer_indices:
                attn_score = self.model.model.layers[layer].self_attn.attn_score
                for idx in range(len(end_id)):
                    head_score = attn_score[0,start_id[idx]:end_id[idx]]
                    head_score = np.mean(head_score, axis=0)
                    predict_result = self.svm[layer][0].predict([head_score])
                    result[idx] += predict_result
            result = [1 if i>=self.svm_num*self.threshold else 0 for i in result]
            print(result)
            for idx in range(len(result)):
                if result[idx] == 0:
                    encodings.attention_mask[0][start_id[idx]+1:end_id[idx]+1] = torch.zeros(end_id[idx]-start_id[idx])
        return encodings
    
    def generate(self, inputs, max_new_tokens=384):
        outputs = self.model.generate(input_ids=inputs.input_ids.cuda(0), attention_mask=inputs.attention_mask.cuda(0), 
                    max_new_tokens=max_new_tokens, 
                    return_dict_in_generate=True, 
                    repetition_penalty=1, 
                    temperature=1.0,
                    top_p=1.0,
                    do_sample  = False
                        )
        return outputs
