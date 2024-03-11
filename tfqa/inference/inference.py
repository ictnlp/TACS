import torch
import numpy as np
import pickle
import os
import sys
sys.path.append('./')
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from model.TACS import TACS_model
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--use_iti", action="store_true")
    parser.add_argument("--TACS_mode", type=str)
    parser.add_argument("--svm_path", type=str)
    parser.add_argument("--svm_acc", type=str)
    parser.add_argument("--input_key", type=str)
    parser.add_argument("--output_key", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--svm_num", type=int, default=5)
    parser.add_argument("--fold_id", type=str, default='2folds_id.pt')
    args = parser.parse_args()

    print(args)
    data = read_data(args.data_path)

    output_list = []
    TACS_model = TACS_model(model_path=args.model_name,
                                use_iti=args.use_iti,
                                TACS_mode=args.TACS_mode)
    TACS_model.window = args.window
    for fold in [1,2]:

        id_list = torch.load('./2folds_id.pt')['fold_{}'.format(fold)]
        units = [data[j] for j in id_list]
        
        if args.svm_path is not None and args.svm_acc is not None:
            TACS_model.svm = torch.load(args.svm_path+"_fold{}.pt".format(3-fold))
            TACS_model.acc = torch.load(args.svm_acc+"_fold{}.pt".format(3-fold))
            TACS_model.sorted_indices = np.argsort(TACS_model.acc)[-args.svm_num:]
            TACS_model.layer_indices = TACS_model.sorted_indices.numpy()

        for id in tqdm(range(0, len(units), args.batch_size)):
            inputs = TACS_model.tokenizer([i[args.input_key] for i in units[id:min(id+args.batch_size,len(units))]], 
                                        return_tensors='pt', 
                                        padding=True)
            if args.TACS_mode is not None:
                inputs = TACS_model.truth_detection(inputs)

            outputs = TACS_model.generate(inputs, 
                                        max_new_tokens=args.max_new_tokens
                                        )
            sequences = TACS_model.tokenizer.batch_decode(outputs.sequences[:,inputs.input_ids.shape[-1]:], 
                                                        skip_special_tokens=True)
            
            for i in range(len(sequences)):
                print(units[id+i][args.input_key])
                print(sequences[i])
                print("======")
                units[id+i][args.output_key] = sequences[i]
        output_list += units

    output_list = sorted(output_list, key=lambda x: x['Question'])
    with jsonlines.open(args.output_path, mode='w') as writer:
        writer.write_all(output_list)

