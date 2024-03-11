import jsonlines
import torch
import sys
sys.path.append('./')
from inference.utils import *
import argparse

def get_cnt(units, a_units, b_units, c_units, no_answer_units, input_key, key):
    for i in units:    
        A_idx = i[input_key].lower().find('A:'.lower())+3
        A_end_idx = i[input_key].lower().find('B:'.lower())-1
        B_idx = i[input_key].lower().find('B:'.lower())+3
        B_end_idx = i[input_key].lower().find('Follow the'.lower())-1
        
        A_answer = i[input_key][A_idx:A_end_idx].strip()
        B_answer = i[input_key][B_idx:B_end_idx].strip()

        a_idx = 100000
        for j in ['option: (a)', 'option: a.','option: a ', 'option: a,', 'option a', '(a)', 'a:', 'cannot choose option b', ':a', ': a', A_answer]:
            a = i[key].lower().find(j.lower())
            a_idx = min(a_idx, a) if a != -1 else a_idx
        b_idx = 100000
        for j in ['option: (b)', 'option: b.','option: b ', 'option: b,', 'option b', '(b)', 'b:', 'cannot choose option a', ':b', ': b',B_answer]:
            b = i[key].lower().find(j.lower())
            b_idx = min(b_idx, b) if b != -1 else b_idx
        c_idx = 100000
        for j in ['option: (c)', 'option: c.','option: c ', 'option: c,', 'option c', '(c)', 'c:', ':c', ': c']:
            c = i[key].lower().find(j.lower())
            c_idx = min(c_idx, c) if c != -1 else c_idx
        if a_idx == 100000 and b_idx == 100000 and c_idx == 100000:
            no_answer_units.append(i)
        else:
            min_idx = min(a_idx, b_idx, c_idx)
            if min_idx == a_idx:
                a_units.append(i)
            elif min_idx == b_idx:
                b_units.append(i)
            elif min_idx == c_idx:
                c_units.append(i)
def print_cnt(a_units, b_units, c_units, no_answer_units, true_false=True):
    if true_false:
        Acc = len(a_units)/(len(a_units)+len(b_units)+len(c_units)+len(no_answer_units))
        print(len(a_units))
        print('True_False Acc: ', Acc)
    else:
        Acc = len(b_units)/(len(a_units)+len(b_units)+len(c_units)+len(no_answer_units))
        print(len(b_units))
        print('False_True Acc: ', Acc)
    return Acc

def getabc(result_name, input_key, output_key):
    # print(result_name)
    units = read_data(result_name)
    a_units = []
    b_units = []
    c_units = []
    no_answer_units = []
    get_cnt(units, a_units, b_units, c_units, no_answer_units, input_key, output_key)
    true_false = 'true_false' in output_key

    return print_cnt(a_units, b_units, c_units, no_answer_units, true_false)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name_tf', type=str)
    parser.add_argument('--input_key_tf', type=str)
    parser.add_argument('--output_key_tf', type=str)
    parser.add_argument('--result_name_ft', type=str)
    parser.add_argument('--input_key_ft', type=str)
    parser.add_argument('--output_key_ft', type=str)

    args = parser.parse_args()
    # print(args)

    Acc_true_false = getabc(args.result_name_tf, 
       args.input_key_tf, 
       args.output_key_tf)
    Acc_false_true = getabc(args.result_name_ft,
         args.input_key_ft, 
         args.output_key_ft)
    Avg_Acc = (Acc_true_false+Acc_false_true)/2
    print('Avg_Acc: ', Avg_Acc)