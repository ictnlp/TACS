import jsonlines
import torch
import sys
sys.path.append('./')
from inference.utils import *
import argparse

def get_cnt(units, right_units, wrong_units, c_units, no_answer_units, input_key, key, answer_ab):

    a_cnt = 0
    b_cnt = 0
    for i in units:    
        A_idx = i[input_key].lower().find('A:'.lower())+3
        A_end_idx = i[input_key].lower().find('B:'.lower())-1
        B_idx = i[input_key].lower().find('B:'.lower())+3
        B_end_idx = i[input_key].lower().find('[/INST]'.lower())-1
        
        A_answer = i[input_key][A_idx:A_end_idx].strip()
        B_answer = i[input_key][B_idx:B_end_idx].strip()

        a_idx = 100000
        for j in ['option: (a)', 'option: a.','option: a ', 'option: a,', 'option a', '(a)', 'a:', 'cannot choose option b', ':a', ': a', A_answer]:
            a = i[key].lower().find(j.lower())
            a_idx = min(a_idx, a) if a != -1 else a_idx
        b_idx = 100000
        for j in ['option: (b)', 'option: b.','option: b ', 'option: b,', 'option b', '(b)', 'b:', 'cannot choose option a', ':b', ': b', B_answer]:
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
                if answer_ab and i['origin_correct']:
                    right_units.append(i)
                elif not answer_ab and not i['origin_correct']:
                    right_units.append(i)
                else:
                    wrong_units.append(i)
                a_cnt += 1
            elif min_idx == b_idx:
                if answer_ab and not i['origin_correct']:
                    right_units.append(i)
                elif not answer_ab and i['origin_correct']:
                    right_units.append(i)
                else:
                    wrong_units.append(i)
                b_cnt += 1
            elif min_idx == c_idx:
                wrong_units.append(i)

def print_cnt(right_units, wrong_units, c_units, no_answer_units, answer_ab=True):
    
    Acc = len(right_units)/(len(right_units)+len(wrong_units)+len(c_units)+len(no_answer_units))
    if answer_ab:
        print('answer_ab Acc: ', Acc)
    else:
        print('answer_ba Acc: ', Acc)
    return Acc

def getabc(result_name, input_key, output_key):
    # print(result_name)
    units = read_data(result_name)
    a_units = []
    b_units = []
    c_units = []
    no_answer_units = []
    answer_ab = 'answer_ab' in output_key
    get_cnt(units, a_units, b_units, c_units, no_answer_units, input_key, output_key, answer_ab)
    

    return print_cnt(a_units, b_units, c_units, no_answer_units, answer_ab)

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