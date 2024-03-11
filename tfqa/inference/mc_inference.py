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
import truthfulqa

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


    data = read_data(args.data_path)
    data = pd.DataFrame(data)

    TACS_model = TACS_model(model_path=args.model_name,
                                use_iti=args.use_iti,
                                TACS_mode=args.TACS_mode)
    TACS_model.window = args.window
    TACS_model.tokenizer.padding_side = "right"

    if args.TACS_mode is None:
        run_probs(data, 
                  args.model_name, 
                  preset='qa', 
                  TACS_model=TACS_model,
                  input_key=args.input_key)
    elif args.TACS_mode == 'PMC_single_token':
        run_probs_token_mask(data, 
                            args.model_name, 
                            preset='qa', 
                            TACS_model=TACS_model,
                            input_key=args.input_key,
                            svm_path=args.svm_path,
                            svm_acc=args.svm_acc)
    elif args.TACS_mode == 'PMC_single_sentence':
        run_probs_sentence_mask(data, 
                            args.model_name, 
                            preset='qa', 
                            TACS_model=TACS_model,
                            input_key=args.input_key,
                            svm_path=args.svm_path,
                            svm_acc=args.svm_acc)
    elif args.TACS_mode == 'PMC_double_token':
        run_probs_token_mask_double_info(data, 
                            args.model_name, 
                            preset='qa', 
                            TACS_model=TACS_model,
                            input_key=args.input_key,
                            svm_path=args.svm_path,
                            svm_acc=args.svm_acc)
    elif args.TACS_mode == 'PMC_double_sentence':
        run_probs_sentence_mask_double_info(data, 
                            args.model_name, 
                            preset='qa', 
                            TACS_model=TACS_model,
                            input_key=args.input_key,
                            svm_path=args.svm_path,
                            svm_acc=args.svm_acc)
    truthfulqa.utilities.save_questions(data, args.output_path+'.csv')

    # format and print basic results
    results = format_frame(data)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2', 'MC3',
                                                'bleu acc',
                                                'rouge1 acc',
                                                'BLEURT acc',
                                                'GPT-judge acc',
                                                'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    results.to_csv(args.output_path+'_mc_metrics.csv')


