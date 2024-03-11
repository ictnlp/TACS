import torch
import jsonlines
import random
random.seed(0)

data = []
with jsonlines.open('./data/truthfulQA/TruthfulQA.jsonl') as reader:
    for obj in reader:
        data.append(obj)


# torch.random.manual_seed(1)
id_list = [i for i in range(len(data))]
random.shuffle(id_list)

length = len(id_list)
torch.save({'fold_1': id_list[:length//2],
            'fold_2': id_list[length//2:]}, './2folds_id.pt')