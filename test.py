"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""


import os
import torch
from test_api import test_api
from test_api import RobertaEmbeddings
from transformers import RobertaConfig, RobertaTokenizer
from models.modeling_roberta import RobertaModel
import numpy as np
import csv


task_name = 'MRPC'
device = 'cuda:0'
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

pre_str = tokenizer.decode(list(range(1000, 1050))) + ' . '
middle_str = ' ? <mask> .'


for seed in [8, 13, 42, 50, 60]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    best = torch.load(f'./results/{task_name}/{seed}/best.pt').to(device).view(50, -1)

    def sentence_fn(test_data):
        """
        This func can be a little confusing.
        Since there are 2 sentences in MRPC and SNLI each sample, we use the same variable `test_data` to represent both.
        test_data is actually a <dummy_token>. It is then replaced by real data in the wrapped API.
        For other 4 tasks, test_data must be used only once, e.g. pre_str + test_data + post_str
        """
        return pre_str + test_data + middle_str + test_data


    def embedding_and_attention_mask_fn(embedding, attention_mask):
        # res = torch.cat([init_prompt[:-5, :], input_embed, init_prompt[-5:, :]], dim=0)
        prepad = torch.zeros(size=(1, 1024), device=device)
        pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        return embedding + torch.cat([prepad, best, pospad]), attention_mask

    predictions = torch.tensor([], device=device)
    for res, _ in test_api(
        sentence_fn=sentence_fn,
        embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
        test_data_path=f'./test_datasets/{task_name}/encrypted.pth',
        task_name=task_name,
        device=device
    ):

        c0 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
        c1 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]

        pred = torch.stack([c0, c1]).argmax(dim=0)
        predictions = torch.cat([predictions, pred])

    if not os.path.exists(f'./predictions/{task_name}'):
        os.makedirs(f'./predictions/{task_name}')
    with open(f'./predictions/{task_name}/{seed}.csv', 'a+') as f:
        wt = csv.writer(f)
        wt.writerow(['', 'pred'])
        wt.writerows(torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())




