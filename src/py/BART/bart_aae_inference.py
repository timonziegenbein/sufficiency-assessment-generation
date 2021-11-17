from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats


df_aae = pd.read_json('../../../ceph_data/intermediate/corpus-ukp-argument-annotated-essays-v2/aae3.json')
df_split =pd.read_csv('../../../ceph_data/input/UKP-InsufficientArguments_v1.0/data-splitting.tsv', sep='\t', names=['index']+[str(i) for i in range(100)], index_col=False)

def make_sentence(x):
    x = x.capitalize()
    if x[-1] not in ['.','?','!']:
        x += '.'
    return x

df_aae['premises'] = df_aae['premises'].apply(lambda x: ' '.join([make_sentence(x1) for x1 in x]))
premises = df_aae['premises'].values

for j in range(10):
    generated_conclusions = []
    batch_size = 16
    tokenizer = AutoTokenizer.from_pretrained('../../../ceph_data/output/bart-large-AAE-v2-dot-sub/{}/best_tfmr2'.format(j))
    model = AutoModelForSeq2SeqLM.from_pretrained('../../../ceph_data/output/bart-large-AAE-v2-dot-sub/{}/best_tfmr2'.format(j), return_dict=True).cuda()
    pbar = tqdm(range(batch_size,len(premises)+batch_size,batch_size))

    for i in pbar:
        pbar.set_description('|   -running model')
        if i<len(premises):
            premises_batch = premises[i-batch_size:i]
        else:
            premises_batch = premises[i-batch_size:]

        #print(premises_batch)
        input_tokens = tokenizer(list(premises_batch), return_tensors='pt', truncation=True, padding="longest")
        generated_conclusion_batch = model.generate(
            input_tokens['input_ids'].cuda(),
            use_cache=True,
            early_stopping= True,
            length_penalty= 2.0,
            max_length= 70,
            min_length= 0,
            no_repeat_ngram_size= 3,
            num_beams= 4,
        )
        generated_conclusion_batch = [tokenizer.decode(t, skip_special_tokens=True) for t in generated_conclusion_batch]
        generated_conclusions += generated_conclusion_batch

    np.savetxt('../../../ceph_data/output/bart-large-AAE-v2-dot-sub/{}/full_aee.preds'.format(j), generated_conclusions, '%s')
